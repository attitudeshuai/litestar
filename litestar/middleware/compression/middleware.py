# pyright: reportUnnecessaryTypeIgnoreComment=false

from __future__ import annotations

from contextlib import suppress
from io import BytesIO
from typing import TYPE_CHECKING, Any, Literal

from litestar.datastructures import Headers, MutableScopeHeaders
from litestar.enums import CompressionEncoding, ScopeType
from litestar.middleware.base import AbstractMiddleware
from litestar.middleware.compression.gzip_facade import GzipCompression
from litestar.middleware.compression.negotiation import NegotiationResult, select_content_coding
from litestar.response.base import ASGIResponse
from litestar.status_codes import HTTP_406_NOT_ACCEPTABLE
from litestar.utils.empty import value_or_default
from litestar.utils.scope.state import ScopeState

if TYPE_CHECKING:
    from litestar.config.compression import CompressionConfig
    from litestar.middleware.compression.facade import CompressionFacade
    from litestar.types import (
        ASGIApp,
        HTTPResponseBodyEvent,
        HTTPResponseStartEvent,
        Message,
        Receive,
        Scope,
        Send,
    )

    try:
        from brotli import Compressor
    except ImportError:
        Compressor = Any

NO_TRANSFORM = "no-transform"
VARY_ACCEPT_ENCODING = "Accept-Encoding"


class CompressionMiddleware(AbstractMiddleware):
    """Compression Middleware Wrapper.

    This is a wrapper allowing for generic compression configuration / handler middleware
    """

    def __init__(self, app: ASGIApp, config: CompressionConfig) -> None:
        """Initialize ``CompressionMiddleware``

        Args:
            app: The ``next`` ASGI app to call.
            config: An instance of CompressionConfig.
        """
        super().__init__(
            app=app, exclude=config.exclude, exclude_opt_key=config.exclude_opt_key, scopes={ScopeType.HTTP}
        )
        self.config = config

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """ASGI callable.

        Args:
            scope: The ASGI connection scope.
            receive: The ASGI receive function.
            send: The ASGI send function.

        Returns:
            None
        """
        accept_encoding = Headers.from_scope(scope).get("accept-encoding")
        config = self.config
        facades_by_encoding = {facade.encoding: facade for facade in config.backend_facades}

        selection = select_content_coding(
            offered=tuple(facades_by_encoding),
            accept_encoding=accept_encoding,
            select_on_absent_header=config.multi_backend,
            client_order_tie_break=config.multi_backend,
        )

        if selection.result is NegotiationResult.NOT_ACCEPTABLE:
            await self.send_not_acceptable_response(scope, receive, send)
            return

        if selection.result is NegotiationResult.IDENTITY or selection.coding is None:
            await self.app(scope, receive, send)
            return

        compression_encoding = selection.coding
        facade_cls = facades_by_encoding.get(compression_encoding)
        if facade_cls is None:
            # Backwards compatibility for custom subclasses invoking the wrapper directly.
            facade_cls = (
                GzipCompression if compression_encoding == CompressionEncoding.GZIP else config.compression_facade
            )

        compression_send = _ResponseCompressionSend(
            send=send,
            compression_encoding=compression_encoding,
            compression_facade=facade_cls,
            config=config,
            scope=scope,
            stream_minimum_size=config.multi_backend,
        )
        await self.app(scope, receive, compression_send)

    async def send_not_acceptable_response(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Send a ``406 Not Acceptable`` response when neither an offered content coding
        nor ``identity`` is acceptable to the client.

        Args:
            scope: The ASGI connection scope.
            receive: The ASGI receive function.
            send: The ASGI send function.

        Returns:
            None
        """
        is_head_response = scope.get("method") == "HEAD"
        response = ASGIResponse(
            body=b"" if is_head_response else b"Not Acceptable",
            status_code=HTTP_406_NOT_ACCEPTABLE,
            media_type="text/plain",
            headers=[("vary", VARY_ACCEPT_ENCODING)],
            is_head_response=is_head_response,
        )
        await response(scope, receive, send)

    def create_compression_send_wrapper(
        self,
        send: Send,
        compression_encoding: Literal[CompressionEncoding.BROTLI, CompressionEncoding.GZIP, CompressionEncoding.ZSTD]
        | str,
        scope: Scope,
    ) -> Send:
        """Wrap ``send`` to handle compression.

        Args:
            send: The ASGI send function.
            compression_encoding: The compression encoding used.
            scope: The ASGI connection scope

        Returns:
            An ASGI send function.
        """
        facades_by_encoding = {facade.encoding: facade for facade in self.config.backend_facades}
        facade_cls = facades_by_encoding.get(compression_encoding)
        if facade_cls is None:
            # We can't use `self.config.compression_facade` directly if the compression is
            # `gzip` since it may be being used as a fallback.
            facade_cls = (
                GzipCompression if compression_encoding == CompressionEncoding.GZIP else self.config.compression_facade
            )

        return _ResponseCompressionSend(
            send=send,
            compression_encoding=compression_encoding,
            compression_facade=facade_cls,
            config=self.config,
            scope=scope,
            stream_minimum_size=False,
        )


class _ResponseCompressionSend:
    """Stateful, per-response compression wrapper around an ASGI ``send`` callable.

    The compressor and its buffer are created lazily, only when compression actually
    starts, and are finalized exactly once - both on the regular end of the response
    and on mid-send exceptions or an early client disconnect - so no half-finished
    frame is ever emitted and no resources are leaked.
    """

    def __init__(
        self,
        send: Send,
        compression_encoding: str,
        compression_facade: type[CompressionFacade],
        config: CompressionConfig,
        scope: Scope,
        stream_minimum_size: bool,
    ) -> None:
        self._send = send
        self._compression_encoding = compression_encoding
        self._compression_facade = compression_facade
        self._config = config
        self._stream_minimum_size = stream_minimum_size
        self._connection_state = ScopeState.from_scope(scope)

        self._initial_message: HTTPResponseStartEvent | None = None
        self._facade: CompressionFacade | None = None
        self._buffer: BytesIO | None = None
        # Raw chunks held back while buffering a streaming response until it reaches
        # the minimum compression size.
        self._pending: list[bytes] = []

        # awaiting -> buffering -> streaming -> finished
        # awaiting -> streaming -> finished
        # awaiting -> passthrough -> finished
        self._state: Literal["awaiting", "buffering", "streaming", "passthrough", "finished"] = "awaiting"
        self._closed = False
        self._aborted = False

    async def __call__(self, message: Message) -> None:
        """Handle and compress an ASGI HTTP message.

        Args:
            message: An ASGI Message.
        """
        try:
            await self._dispatch(message)
        except BaseException:
            # Covers exceptions raised while compressing as well as failures of the
            # downstream send (e.g. the client disconnecting mid-response).
            self._abort()
            raise

    # ------------------------------------------------------------------
    # Message dispatch
    # ------------------------------------------------------------------
    async def _dispatch(self, message: Message) -> None:
        if message["type"] == "http.response.start":
            if self._initial_message is None:
                self._initial_message = message
            return

        if self._aborted or self._state == "finished":
            # Never forward anything once the response was aborted/finalized.
            return

        if self._initial_message is None:
            await self._send(message)
            return

        if message["type"] == "http.disconnect":
            self._abort()
            return

        if message["type"] != "http.response.body":
            await self._send(message)
            return

        body = message["body"]
        more_body = message["more_body"]

        if self._state == "awaiting":
            await self._handle_first_body(message, body, more_body)
        elif self._state == "passthrough":
            await self._send(message)
            if not more_body:
                self._state = "finished"
        elif self._state == "streaming":
            await self._handle_streaming_body(message, body, more_body)
        else:  # buffering
            await self._handle_buffered_body(message, body, more_body)

    async def _handle_first_body(self, message: HTTPResponseBodyEvent, body: bytes, more_body: bool) -> None:
        initial_message = self._require_initial_message()
        headers = MutableScopeHeaders(initial_message)

        # Per-response declarations: a response that already carries a content
        # coding must not be encoded again, and ``no-transform`` forbids any
        # transformation of the response.
        if "content-encoding" in headers:
            self._enter_passthrough(headers, add_vary=False)
            await self._send(initial_message)
            await self._send(message)
            if not more_body:
                self._state = "finished"
            return

        if _has_no_transform_directive(headers.get("cache-control")):
            self._enter_passthrough(headers, add_vary=True)
            await self._send(initial_message)
            await self._send(message)
            if not more_body:
                self._state = "finished"
            return

        # Cached responses are stored already rendered and must be passed through.
        if value_or_default(self._connection_state.is_cached, False):
            self._enter_passthrough(headers, add_vary=False)
            await self._send(initial_message)
            await self._send(message)
            if not more_body:
                self._state = "finished"
            return

        if not more_body:
            await self._handle_single_body(message, body)
            return

        if self._stream_minimum_size:
            self._state = "buffering"
            self._pending = [body]
            if len(body) >= self._config.minimum_size:
                await self._commit_buffered(message, final=False)
            return

        facade = self._begin_streaming(initial_message, headers)
        facade.write(body, final=False)
        message["body"] = self._drain_buffer()
        await self._send(initial_message)
        await self._send(message)
        self._state = "streaming"

    async def _handle_single_body(self, message: HTTPResponseBodyEvent, body: bytes) -> None:
        initial_message = self._require_initial_message()
        headers = MutableScopeHeaders(initial_message)

        if len(body) < self._config.minimum_size:
            await self._send(initial_message)
            await self._send(message)
            self._state = "finished"
            return

        self._create_compressor()
        compressed = self._finish_compression(body)
        headers["Content-Encoding"] = self._compression_encoding
        headers["Content-Length"] = str(len(compressed))
        headers.extend_header_value("vary", VARY_ACCEPT_ENCODING)
        message["body"] = compressed
        self._connection_state.response_compressed = True

        await self._send(initial_message)
        await self._send(message)
        self._state = "finished"

    async def _handle_streaming_body(self, message: HTTPResponseBodyEvent, body: bytes, more_body: bool) -> None:
        facade = self._create_compressor()
        final = not more_body
        facade.write(body, final=final)
        if final:
            # Closing the facade flushes any remaining data (e.g. the gzip trailer
            # or the final zstd frame); drain after closing, then close the buffer.
            message["body"] = self._finish_compression(b"")
            self._state = "finished"
        else:
            message["body"] = self._drain_buffer()

        await self._send(message)

    async def _handle_buffered_body(self, message: HTTPResponseBodyEvent, body: bytes, more_body: bool) -> None:
        self._pending.append(body)
        total_size = sum(len(chunk) for chunk in self._pending)
        final = not more_body

        if not final and total_size < self._config.minimum_size:
            # Keep holding the response back until the threshold is reached.
            return

        if total_size >= self._config.minimum_size:
            await self._commit_buffered(message, final=final)
            return

        # The stream ended before reaching the minimum size: send it uncompressed.
        payload = b"".join(self._pending)
        self._pending = []
        initial_message = self._require_initial_message()
        headers = MutableScopeHeaders(initial_message)
        headers["Content-Length"] = str(len(payload))
        message["body"] = payload
        message["more_body"] = False

        await self._send(initial_message)
        await self._send(message)
        self._state = "finished"

    async def _commit_buffered(self, message: HTTPResponseBodyEvent, final: bool) -> None:
        """Commit to compressing the buffered streaming response."""
        initial_message = self._require_initial_message()
        headers = MutableScopeHeaders(initial_message)
        facade = self._begin_streaming(initial_message, headers)

        pending = self._pending
        self._pending = []
        last_index = len(pending) - 1
        for index, chunk in enumerate(pending):
            facade.write(chunk, final=final and index == last_index)

        if final:
            message["body"] = self._finish_compression(b"")
            self._state = "finished"
        else:
            message["body"] = self._drain_buffer()
            self._state = "streaming"

        await self._send(initial_message)
        await self._send(message)

    # ------------------------------------------------------------------
    # Compression lifecycle
    # ------------------------------------------------------------------
    def _require_initial_message(self) -> HTTPResponseStartEvent:
        initial_message = self._initial_message
        if initial_message is None:
            raise RuntimeError("Received a response body before the response start event")
        return initial_message

    def _create_compressor(self) -> CompressionFacade:
        facade = self._facade
        if facade is None:
            buffer = BytesIO()
            self._buffer = buffer
            facade = self._compression_facade(
                buffer=buffer,
                compression_encoding=self._compression_encoding,
                config=self._config,
            )
            self._facade = facade
        return facade

    def _begin_streaming(
        self, initial_message: HTTPResponseStartEvent, headers: MutableScopeHeaders
    ) -> CompressionFacade:
        facade = self._create_compressor()
        headers["Content-Encoding"] = self._compression_encoding
        headers.extend_header_value("vary", VARY_ACCEPT_ENCODING)
        del headers["Content-Length"]
        self._connection_state.response_compressed = True
        return facade

    def _drain_buffer(self) -> bytes:
        buffer = self._buffer
        if buffer is None:
            raise RuntimeError("Cannot drain the compression buffer before compression has started")
        data = buffer.getvalue()
        buffer.seek(0)
        buffer.truncate()
        return data

    def _finish_compression(self, body: bytes) -> bytes:
        """Write ``body`` as the final input, close the facade and return the full
        remaining compressed output. Idempotent and safe to call once.
        """
        facade = self._facade
        buffer = self._buffer
        if facade is None or buffer is None or self._closed:
            return b""
        facade.write(body, final=True)
        facade.close()
        data = buffer.getvalue()
        buffer.close()
        self._buffer = None
        self._closed = True
        return data

    def _enter_passthrough(self, headers: MutableScopeHeaders, add_vary: bool) -> None:
        if add_vary:
            vary = headers.get("vary")
            accepted = {value.strip().lower() for value in vary.split(",")} if vary else set()
            if "accept-encoding" not in accepted:
                headers.extend_header_value("vary", VARY_ACCEPT_ENCODING)
        self._state = "passthrough"

    def _abort(self) -> None:
        """Deterministically finalize the compressor and the buffer on abort,
        discarding any un-emitted, half-finished output.
        """
        if self._closed or self._aborted:
            return
        self._aborted = True
        self._state = "finished"
        self._pending = []

        facade = self._facade
        buffer = self._buffer
        if facade is not None:
            with suppress(Exception):
                facade.close()
        if buffer is not None:
            buffer.close()
            self._buffer = None


def _has_no_transform_directive(cache_control: str | None) -> bool:
    if not cache_control:
        return False
    return any(directive.strip().lower() == NO_TRANSFORM for directive in cache_control.split(","))
