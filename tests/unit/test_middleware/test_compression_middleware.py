# pyright: reportUnnecessaryTypeIgnoreComment=false

import gzip
import sys
import zlib
from collections.abc import AsyncIterator, Callable
from io import BytesIO
from typing import Literal, Union, cast
from unittest.mock import MagicMock

import pytest

from litestar import MediaType, WebSocket, get, websocket
from litestar.config.compression import CompressionConfig
from litestar.enums import CompressionEncoding
from litestar.exceptions import ImproperlyConfiguredException
from litestar.handlers import HTTPRouteHandler
from litestar.middleware.compression import (
    CompressionMiddleware,
    NegotiationResult,
    parse_accept_encoding,
    select_content_coding,
)
from litestar.middleware.compression.facade import CompressionFacade
from litestar.middleware.compression.middleware import _ResponseCompressionSend
from litestar.response import Response
from litestar.response.streaming import Stream
from litestar.status_codes import HTTP_200_OK, HTTP_406_NOT_ACCEPTABLE
from litestar.testing import create_test_client
from litestar.types.asgi_types import (
    ASGIApp,
    HTTPResponseBodyEvent,
    HTTPResponseStartEvent,
    Message,
    Receive,
    Scope,
    Send,
)

if sys.version_info >= (3, 14):
    from compression import zstd
else:
    from backports import zstd
zstd_compression_level_upper_bound = zstd.CompressionParameter.compression_level.bounds()[1]

BrotliMode = Literal["text", "generic", "font"]


@pytest.fixture()
def handler() -> HTTPRouteHandler:
    @get(path="/", media_type=MediaType.TEXT)
    def handler_fn() -> str:
        return "_litestar_" * 4000

    return handler_fn


async def streaming_iter(content: bytes, count: int) -> AsyncIterator[bytes]:
    for _ in range(count):
        yield content


def test_compression_disabled_for_unsupported_client(handler: HTTPRouteHandler) -> None:
    with create_test_client(route_handlers=[handler], compression_config=CompressionConfig(backend="brotli")) as client:
        response = client.get("/", headers={"accept-encoding": "deflate"})
        assert response.status_code == HTTP_200_OK
        assert response.text == "_litestar_" * 4000
        assert "Content-Encoding" not in response.headers
        assert int(response.headers["Content-Length"]) == 40000


@pytest.mark.parametrize(
    "backend, compression_encoding",
    (("brotli", CompressionEncoding.BROTLI), ("gzip", CompressionEncoding.GZIP), ("zstd", CompressionEncoding.ZSTD)),
)
def test_regular_compressed_response(
    backend: Literal["gzip", "brotli", "zstd"], compression_encoding: CompressionEncoding, handler: HTTPRouteHandler
) -> None:
    with create_test_client(
        route_handlers=[handler],
        compression_config=CompressionConfig(backend=backend),
        raise_server_exceptions=True,
    ) as client:
        response = client.get("/", headers={"Accept-Encoding": str(compression_encoding.value)})
        assert response.status_code == HTTP_200_OK
        assert response.text == "_litestar_" * 4000
        assert response.headers["Content-Encoding"] == compression_encoding
        assert int(response.headers["Content-Length"]) < 40000


@pytest.mark.parametrize(
    "backend, compression_encoding",
    (("brotli", CompressionEncoding.BROTLI), ("gzip", CompressionEncoding.GZIP), ("zstd", CompressionEncoding.ZSTD)),
)
def test_compression_works_for_streaming_response(
    backend: Literal["gzip", "brotli", "zstd"], compression_encoding: CompressionEncoding
) -> None:
    @get("/streaming-response")
    def streaming_handler() -> Stream:
        return Stream(streaming_iter(content=b"_litestar_" * 400, count=10))

    with create_test_client(
        route_handlers=[streaming_handler], compression_config=CompressionConfig(backend=backend)
    ) as client:
        response = client.get("/streaming-response", headers={"Accept-Encoding": str(compression_encoding.value)})
        assert response.status_code == HTTP_200_OK
        assert response.text == "_litestar_" * 4000
        assert response.headers["Content-Encoding"] == compression_encoding
        assert "Content-Length" not in response.headers


@pytest.mark.parametrize(
    "backend, compression_encoding",
    (("brotli", CompressionEncoding.BROTLI), ("gzip", CompressionEncoding.GZIP), ("zstd", CompressionEncoding.ZSTD)),
)
def test_compression_skips_small_responses(
    backend: Literal["gzip", "brotli", "zstd"], compression_encoding: CompressionEncoding
) -> None:
    @get(path="/no-compression", media_type=MediaType.TEXT)
    def no_compress_handler() -> str:
        return "_litestar_"

    with create_test_client(
        route_handlers=[no_compress_handler], compression_config=CompressionConfig(backend=backend)
    ) as client:
        response = client.get("/no-compression", headers={"Accept-Encoding": str(compression_encoding.value)})
        assert response.status_code == HTTP_200_OK
        assert response.text == "_litestar_"
        assert "Content-Encoding" not in response.headers
        assert int(response.headers["Content-Length"]) == 10


def test_brotli_with_gzip_fallback_enabled(handler: HTTPRouteHandler) -> None:
    with create_test_client(
        route_handlers=[handler], compression_config=CompressionConfig(backend="brotli", brotli_gzip_fallback=True)
    ) as client:
        response = client.get("/", headers={"accept-encoding": CompressionEncoding.GZIP.value})
        assert response.status_code == HTTP_200_OK
        assert response.text == "_litestar_" * 4000
        assert response.headers["Content-Encoding"] == CompressionEncoding.GZIP
        assert int(response.headers["Content-Length"]) < 40000


def test_brotli_gzip_fallback_disabled(handler: HTTPRouteHandler) -> None:
    with create_test_client(
        route_handlers=[handler],
        compression_config=CompressionConfig(backend="brotli", brotli_gzip_fallback=False),
    ) as client:
        response = client.get("/", headers={"accept-encoding": "gzip"})
        assert response.status_code == HTTP_200_OK
        assert response.text == "_litestar_" * 4000
        assert "Content-Encoding" not in response.headers
        assert int(response.headers["Content-Length"]) == 40000


async def test_skips_for_websocket() -> None:
    @websocket("/")
    async def websocket_handler(socket: WebSocket) -> None:
        data = await socket.receive_json()
        await socket.send_json(data)
        await socket.close()

    with (
        create_test_client(
            route_handlers=[websocket_handler],
            compression_config=CompressionConfig(backend="brotli", brotli_gzip_fallback=False),
        ) as client,
        client.websocket_connect("/") as ws,
    ):
        assert b"content-encoding" not in dict(ws.scope["headers"])


@pytest.mark.parametrize("minimum_size, should_raise", ((0, True), (1, False), (-1, True), (100, False)))
def test_config_minimum_size_validation(minimum_size: int, should_raise: bool) -> None:
    if should_raise:
        with pytest.raises(ImproperlyConfiguredException):
            CompressionConfig(backend="brotli", brotli_gzip_fallback=False, minimum_size=minimum_size)
    else:
        CompressionConfig(backend="brotli", brotli_gzip_fallback=False, minimum_size=minimum_size)


@pytest.mark.parametrize(
    "gzip_compress_level, should_raise", ((0, False), (1, False), (-1, True), (10, True), (9, False))
)
def test_config_gzip_compress_level_validation(gzip_compress_level: int, should_raise: bool) -> None:
    if should_raise:
        with pytest.raises(ImproperlyConfiguredException):
            CompressionConfig(backend="gzip", brotli_gzip_fallback=False, gzip_compress_level=gzip_compress_level)
    else:
        CompressionConfig(backend="gzip", brotli_gzip_fallback=False, gzip_compress_level=gzip_compress_level)


@pytest.mark.parametrize(
    "zstd_compress_level, should_raise",
    (
        (-1, True),
        (0, False),
        (1, False),
        (zstd_compression_level_upper_bound, False),
        (zstd_compression_level_upper_bound + 1, True),
    ),
)
def test_config_zstd_compress_level_validation(zstd_compress_level: int, should_raise: bool) -> None:
    if should_raise:
        with pytest.raises(ImproperlyConfiguredException):
            CompressionConfig(backend="zstd", zstd_compress_level=zstd_compress_level)
    else:
        CompressionConfig(backend="zstd", zstd_compress_level=zstd_compress_level)


@pytest.mark.parametrize("brotli_quality, should_raise", ((0, False), (1, False), (-1, True), (12, True), (11, False)))
def test_config_brotli_quality_validation(brotli_quality: int, should_raise: bool) -> None:
    if should_raise:
        with pytest.raises(ImproperlyConfiguredException):
            CompressionConfig(backend="brotli", brotli_gzip_fallback=False, brotli_quality=brotli_quality)
    else:
        CompressionConfig(backend="brotli", brotli_gzip_fallback=False, brotli_quality=brotli_quality)


@pytest.mark.parametrize("brotli_lgwin, should_raise", ((9, True), (10, False), (-1, True), (25, True), (24, False)))
def test_config_brotli_lgwin_validation(brotli_lgwin: int, should_raise: bool) -> None:
    if should_raise:
        with pytest.raises(ImproperlyConfiguredException):
            CompressionConfig(backend="brotli", brotli_gzip_fallback=False, brotli_lgwin=brotli_lgwin)
    else:
        CompressionConfig(backend="brotli", brotli_gzip_fallback=False, brotli_lgwin=brotli_lgwin)


@pytest.mark.parametrize(
    "backend, compression_encoding",
    (
        ("brotli", CompressionEncoding.BROTLI),
        ("gzip", CompressionEncoding.GZIP),
        ("zstd", CompressionEncoding.ZSTD),
    ),
)
async def test_compression_streaming_response_emitted_messages(
    backend: Literal["gzip", "brotli", "zstd"],
    compression_encoding: CompressionEncoding,
    create_scope: Callable[..., Scope],
    mock_asgi_app: ASGIApp,
) -> None:
    mock = MagicMock()

    async def fake_send(message: Message) -> None:
        mock(message)

    wrapped_send = CompressionMiddleware(
        mock_asgi_app, CompressionConfig(backend=backend)
    ).create_compression_send_wrapper(fake_send, compression_encoding, create_scope())

    await wrapped_send(HTTPResponseStartEvent(type="http.response.start", status=200, headers={}))
    # first body message always has compression headers (at least for gzip)
    await wrapped_send(HTTPResponseBodyEvent(type="http.response.body", body=b"abc", more_body=True))
    # second body message with more_body=True will be empty if zlib buffers output and is not flushed
    await wrapped_send(HTTPResponseBodyEvent(type="http.response.body", body=b"abc", more_body=True))
    assert mock.mock_calls[-1].args[0]["body"]
    # send a more_body=False so resources close properly
    await wrapped_send(HTTPResponseBodyEvent(type="http.response.body", body=b"", more_body=False))


@pytest.mark.parametrize(
    "backend, compression_encoding",
    (("brotli", CompressionEncoding.BROTLI), ("gzip", CompressionEncoding.GZIP), ("zstd", CompressionEncoding.ZSTD)),
)
def test_dont_recompress_cached(backend: Literal["gzip", "brotli"], compression_encoding: CompressionEncoding) -> None:
    mock = MagicMock(return_value="_litestar_" * 4000)

    @get(path="/", media_type=MediaType.TEXT, cache=True)
    def handler_fn() -> str:
        return mock()  # type: ignore[no-any-return]

    with create_test_client(
        route_handlers=[handler_fn], compression_config=CompressionConfig(backend=backend)
    ) as client:
        client.get("/", headers={"Accept-Encoding": str(compression_encoding.value)})
        response = client.get("/", headers={"Accept-Encoding": str(compression_encoding.value)})

    assert mock.call_count == 1
    assert response.status_code == HTTP_200_OK
    assert response.text == "_litestar_" * 4000
    assert response.headers["Content-Encoding"] == compression_encoding
    assert int(response.headers["Content-Length"]) < 40000


def test_compression_with_custom_backend(handler: HTTPRouteHandler) -> None:
    class ZlibCompression(CompressionFacade):
        encoding = "deflate"

        def __init__(
            self,
            buffer: BytesIO,
            compression_encoding: Union[Literal[CompressionEncoding.GZIP], str],
            config: CompressionConfig,
        ) -> None:
            self.buffer = buffer
            self.compression_encoding = compression_encoding
            self.config = config

        def write(self, body: Union[bytes, bytearray], final: bool = False) -> None:
            self.buffer.write(zlib.compress(body, level=self.config.backend_config["level"]))

        def close(self) -> None: ...

    zlib_config = {"level": 9}
    config = CompressionConfig(backend="deflate", compression_facade=ZlibCompression, backend_config=zlib_config)
    with create_test_client([handler], compression_config=config) as client:
        response = client.get("/", headers={"Accept-Encoding": "deflate"})
        assert response.status_code == HTTP_200_OK
        assert response.text == "_litestar_" * 4000
        assert response.headers["Content-Encoding"] == "deflate"
        assert int(response.headers["Content-Length"]) < 40000


def test_compression_with_custom_middleware(handler: HTTPRouteHandler) -> None:
    mock = MagicMock()

    class CustomCompressionMiddleware(CompressionMiddleware):
        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            mock()
            await super().__call__(scope, receive, send)
            return

    config = CompressionConfig(backend="gzip", middleware_class=CustomCompressionMiddleware)
    with create_test_client([handler], compression_config=config) as client:
        response = client.get("/", headers={"Accept-Encoding": "gzip"})
        assert response.status_code == HTTP_200_OK
        assert response.text == "_litestar_" * 4000
        assert response.headers["Content-Encoding"] == "gzip"
        assert int(response.headers["Content-Length"]) < 40000
        mock.assert_called_once()


# ---------------------------------------------------------------------------
# Accept-Encoding parsing / weight based negotiation
# ---------------------------------------------------------------------------


def test_parse_accept_encoding_absent() -> None:
    assert parse_accept_encoding(None) == ()
    assert parse_accept_encoding("") == ()


def test_parse_accept_encoding_weights_and_order() -> None:
    entries = parse_accept_encoding(" ZSTD ; q = 0.8 , br;q=0.0,gzip")
    assert [(e.coding, e.quality, e.position) for e in entries] == [
        ("zstd", 0.8, 0),
        ("br", 0.0, 1),
        ("gzip", 1.0, 2),
    ]


def test_parse_accept_encoding_invalid_quality_is_rejected() -> None:
    entries = parse_accept_encoding("gzip;q=2, br;q=not-a-number")
    assert {e.coding: e.quality for e in entries} == {"gzip": 0.0, "br": 0.0}


def test_negotiate_absent_header_modes() -> None:
    selection = select_content_coding(("zstd", "gzip"), None, select_on_absent_header=True, client_order_tie_break=True)
    assert selection.result is NegotiationResult.ENCODE
    assert selection.coding == "zstd"

    selection = select_content_coding(
        ("zstd", "gzip"), None, select_on_absent_header=False, client_order_tie_break=False
    )
    assert selection.result is NegotiationResult.IDENTITY


def test_negotiate_highest_quality_wins() -> None:
    selection = select_content_coding(
        ("zstd", "br", "gzip"),
        "gzip;q=0.5, br;q=0.9, zstd;q=0.8",
        select_on_absent_header=True,
        client_order_tie_break=True,
    )
    assert selection.coding == CompressionEncoding.BROTLI


def test_negotiate_tie_broken_by_client_order() -> None:
    selection = select_content_coding(
        ("zstd", "br", "gzip"),
        "gzip, zstd",
        select_on_absent_header=True,
        client_order_tie_break=True,
    )
    assert selection.coding == CompressionEncoding.GZIP


def test_negotiate_tie_broken_by_server_order_in_legacy_mode() -> None:
    selection = select_content_coding(
        ("br", "gzip"),
        "gzip, br",
        select_on_absent_header=False,
        client_order_tie_break=False,
    )
    assert selection.coding == CompressionEncoding.BROTLI


def test_negotiate_zero_quality_rejects_coding() -> None:
    selection = select_content_coding(
        ("gzip",),
        "gzip;q=0",
        select_on_absent_header=False,
        client_order_tie_break=False,
    )
    assert selection.result is NegotiationResult.IDENTITY


def test_negotiate_explicit_coding_overrides_rejecting_wildcard() -> None:
    selection = select_content_coding(
        ("gzip", "br"),
        "*;q=0, gzip;q=1",
        select_on_absent_header=True,
        client_order_tie_break=True,
    )
    assert selection.coding == CompressionEncoding.GZIP


def test_negotiate_wildcard_quality() -> None:
    selection = select_content_coding(
        ("zstd", "gzip"),
        "br;q=0.5, *;q=0.9",
        select_on_absent_header=True,
        client_order_tie_break=True,
    )
    assert selection.coding == CompressionEncoding.ZSTD


def test_negotiate_unlisted_codings_without_wildcard_unacceptable() -> None:
    selection = select_content_coding(
        ("br", "gzip"),
        "deflate",
        select_on_absent_header=True,
        client_order_tie_break=True,
    )
    assert selection.result is NegotiationResult.IDENTITY


@pytest.mark.parametrize("header", ("identity;q=0", "*;q=0", "gzip;q=0, identity;q=0"))
def test_negotiate_not_acceptable(header: str) -> None:
    selection = select_content_coding(
        ("gzip", "br", "zstd"),
        header,
        select_on_absent_header=True,
        client_order_tie_break=True,
    )
    assert selection.result is NegotiationResult.NOT_ACCEPTABLE
    assert selection.coding is None


def test_negotiate_identity_rejected_but_coding_accepted() -> None:
    selection = select_content_coding(
        ("gzip",),
        "identity;q=0, gzip",
        select_on_absent_header=True,
        client_order_tie_break=True,
    )
    assert selection.coding == CompressionEncoding.GZIP


# ---------------------------------------------------------------------------
# Multi-backend configuration
# ---------------------------------------------------------------------------


def test_config_requires_backend_or_backends() -> None:
    with pytest.raises(ImproperlyConfiguredException):
        CompressionConfig()


def test_config_backends_must_not_be_empty() -> None:
    with pytest.raises(ImproperlyConfiguredException):
        CompressionConfig(backend=None, backends=())


def test_config_unknown_backend_name_raises() -> None:
    with pytest.raises(ImproperlyConfiguredException):
        CompressionConfig(backends=("nope",))


def test_config_backends_deduplicates_by_encoding() -> None:
    config = CompressionConfig(backends=("gzip", "br", "brotli"))
    assert [facade.encoding for facade in config.backend_facades] == [
        CompressionEncoding.GZIP,
        CompressionEncoding.BROTLI,
    ]
    assert config.multi_backend is True


def test_config_multi_backend_validates_all_backend_parameters() -> None:
    with pytest.raises(ImproperlyConfiguredException):
        CompressionConfig(backends=("gzip", "br"), gzip_compress_level=10)
    with pytest.raises(ImproperlyConfiguredException):
        CompressionConfig(backends=("gzip", "br"), brotli_quality=12)
    with pytest.raises(ImproperlyConfiguredException):
        CompressionConfig(backends=("zstd",), zstd_compress_level=zstd_compression_level_upper_bound + 1)


def test_config_backends_accept_custom_facade_classes(handler: HTTPRouteHandler) -> None:
    config = CompressionConfig(backends=(ZlibDeflateFacade, "gzip"), backend_config={"level": 9})

    with create_test_client([handler], compression_config=config) as client:
        response = client.get("/", headers={"Accept-Encoding": "deflate"})
        assert response.headers["Content-Encoding"] == "deflate"
        assert int(response.headers["Content-Length"]) < 40000

        response = client.get("/", headers={"Accept-Encoding": "gzip"})
        assert response.headers["Content-Encoding"] == "gzip"


def test_config_single_backend_remains_legacy_mode() -> None:
    config = CompressionConfig(backend="brotli")
    assert config.multi_backend is False
    assert [facade.encoding for facade in config.backend_facades] == [
        CompressionEncoding.BROTLI,
        CompressionEncoding.GZIP,
    ]


# ---------------------------------------------------------------------------
# Weight based negotiation, end to end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "accept_encoding, expected",
    (
        ("gzip;q=0.5, br;q=0.9, zstd;q=0.8", CompressionEncoding.BROTLI),
        ("zstd;q=0.9, gzip;q=0.9, br;q=0.8", CompressionEncoding.ZSTD),
        ("gzip", CompressionEncoding.GZIP),
        ("br", CompressionEncoding.BROTLI),
        ("zstd", CompressionEncoding.ZSTD),
    ),
)
def test_multi_backend_selects_encoding_by_weight(
    handler: HTTPRouteHandler, accept_encoding: str, expected: CompressionEncoding
) -> None:
    with create_test_client(
        route_handlers=[handler], compression_config=CompressionConfig(backends=("zstd", "br", "gzip"))
    ) as client:
        response = client.get("/", headers={"Accept-Encoding": accept_encoding})
        assert response.status_code == HTTP_200_OK
        assert response.headers["Content-Encoding"] == expected
        assert response.text == "_litestar_" * 4000


def test_multi_backend_identity_when_no_coding_accepted(handler: HTTPRouteHandler) -> None:
    with create_test_client(
        route_handlers=[handler], compression_config=CompressionConfig(backends=("gzip", "br"))
    ) as client:
        response = client.get("/", headers={"Accept-Encoding": "zstd;q=0, gzip;q=0, br;q=0"})
        assert response.status_code == HTTP_200_OK
        assert "Content-Encoding" not in response.headers
        assert int(response.headers["Content-Length"]) == 40000


def test_multi_backend_unlisted_codings_fall_back_to_identity(handler: HTTPRouteHandler) -> None:
    with create_test_client(
        route_handlers=[handler], compression_config=CompressionConfig(backends=("gzip", "br"))
    ) as client:
        response = client.get("/", headers={"Accept-Encoding": "deflate"})
        assert response.status_code == HTTP_200_OK
        assert "Content-Encoding" not in response.headers


@pytest.mark.parametrize("accept_encoding", ("identity;q=0", "*;q=0"))
def test_multi_backend_406_when_nothing_acceptable(handler: HTTPRouteHandler, accept_encoding: str) -> None:
    with create_test_client(
        route_handlers=[handler], compression_config=CompressionConfig(backends=("gzip", "br", "zstd"))
    ) as client:
        response = client.get("/", headers={"Accept-Encoding": accept_encoding})
        assert response.status_code == HTTP_406_NOT_ACCEPTABLE
        assert response.text == "Not Acceptable"
        assert "Accept-Encoding" in response.headers["Vary"]


async def test_406_for_head_request_has_no_body() -> None:
    sent: list[Message] = []
    downstream_called = False

    async def send(message: Message) -> None:
        sent.append(message)

    async def downstream_app(scope: Scope, receive: Receive, send: Send) -> None:
        nonlocal downstream_called
        downstream_called = True

    middleware = CompressionMiddleware(downstream_app, CompressionConfig(backends=("gzip",)))
    scope = cast(
        Scope,
        {"type": "http", "method": "HEAD", "headers": [(b"accept-encoding", b"identity;q=0")]},
    )
    await middleware(scope, MagicMock(), send)

    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == HTTP_406_NOT_ACCEPTABLE
    assert sent[1]["type"] == "http.response.body"
    assert sent[1]["body"] == b""
    # the downstream app is never called when no coding is acceptable
    assert not downstream_called


def test_legacy_backend_rejected_coding_not_used(handler: HTTPRouteHandler) -> None:
    with create_test_client(route_handlers=[handler], compression_config=CompressionConfig(backend="brotli")) as client:
        response = client.get("/", headers={"Accept-Encoding": "br;q=0"})
        assert response.status_code == HTTP_200_OK
        assert "Content-Encoding" not in response.headers
        assert int(response.headers["Content-Length"]) == 40000

        response = client.get("/", headers={"Accept-Encoding": "identity;q=0"})
        assert response.status_code == HTTP_406_NOT_ACCEPTABLE


# ---------------------------------------------------------------------------
# Per-response declarations
# ---------------------------------------------------------------------------


def test_response_no_transform_disables_compression(handler: HTTPRouteHandler) -> None:
    @get(path="/no-transform", media_type=MediaType.TEXT)
    def no_transform_handler() -> Response[str]:
        return Response("_litestar_" * 4000, headers={"cache-control": "no-transform"})

    with create_test_client(
        route_handlers=[no_transform_handler], compression_config=CompressionConfig(backends=("gzip",))
    ) as client:
        response = client.get("/no-transform", headers={"Accept-Encoding": "gzip"})
        assert response.status_code == HTTP_200_OK
        assert "Content-Encoding" not in response.headers
        assert response.headers["Cache-Control"] == "no-transform"
        assert "Accept-Encoding" in response.headers["Vary"]
        assert int(response.headers["Content-Length"]) == 40000
        assert response.text == "_litestar_" * 4000


def test_response_already_content_encoded_is_not_recompressed() -> None:
    encoded = gzip.compress(("_litestar_" * 4000).encode())

    @get(path="/pre-encoded")
    def pre_encoded_handler() -> Response[bytes]:
        return Response(
            encoded,
            media_type="application/octet-stream",
            headers={"content-encoding": "gzip"},
        )

    with create_test_client(
        route_handlers=[pre_encoded_handler], compression_config=CompressionConfig(backends=("zstd", "gzip"))
    ) as client:
        response = client.get("/pre-encoded", headers={"Accept-Encoding": "zstd, gzip"})
        assert response.status_code == HTTP_200_OK
        assert response.headers["Content-Encoding"] == "gzip"
        assert int(response.headers["Content-Length"]) == len(encoded)
        # httpx transparently decodes the single gzip layer; a doubly encoded body
        # would fail this roundtrip
        assert response.content == ("_litestar_" * 4000).encode()


# ---------------------------------------------------------------------------
# Streaming: minimum size threshold and deterministic finalization
# ---------------------------------------------------------------------------


def test_multi_backend_streaming_respects_minimum_size() -> None:
    @get("/small-stream")
    def small_stream() -> Stream:
        return Stream(streaming_iter(content=b"x" * 10, count=5))

    @get("/big-stream")
    def big_stream() -> Stream:
        return Stream(streaming_iter(content=b"x" * 400, count=5))

    with create_test_client(
        route_handlers=[small_stream, big_stream], compression_config=CompressionConfig(backends=("gzip",))
    ) as client:
        response = client.get("/small-stream", headers={"Accept-Encoding": "gzip"})
        assert "Content-Encoding" not in response.headers
        assert int(response.headers["Content-Length"]) == 50
        assert response.text == "x" * 50

        response = client.get("/big-stream", headers={"Accept-Encoding": "gzip"})
        assert response.headers["Content-Encoding"] == "gzip"
        assert "Content-Length" not in response.headers
        assert response.text == "x" * 2000


def test_legacy_streaming_is_compressed_regardless_of_size() -> None:
    @get("/small-stream")
    def small_stream() -> Stream:
        return Stream(streaming_iter(content=b"x" * 10, count=5))

    with create_test_client(
        route_handlers=[small_stream], compression_config=CompressionConfig(backend="gzip")
    ) as client:
        response = client.get("/small-stream", headers={"Accept-Encoding": "gzip"})
        assert response.headers["Content-Encoding"] == "gzip"
        assert response.text == "x" * 50


class _TrackingFacade(CompressionFacade):
    encoding = "tracked"

    def __init__(self, buffer: BytesIO, compression_encoding: str, config: CompressionConfig) -> None:
        self.buffer = buffer
        self.write_calls = 0
        self.close_calls = 0

    def write(self, body: bytes | bytearray, final: bool = False) -> None:
        self.write_calls += 1
        self.buffer.write(b"C(" + bytes(body) + b")")

    def close(self) -> None:
        self.close_calls += 1
        self.buffer.write(b"END")


def _build_tracking_wrapper(send: Send, scope: Scope, *, stream_minimum_size: bool) -> _ResponseCompressionSend:
    config = CompressionConfig(backend="gzip", compression_facade=_TrackingFacade, gzip_fallback=False)
    return _ResponseCompressionSend(
        send=send,
        compression_encoding="tracked",
        compression_facade=_TrackingFacade,
        config=config,
        scope=scope,
        stream_minimum_size=stream_minimum_size,
    )


def _tracking_facade(wrapper: _ResponseCompressionSend) -> _TrackingFacade:
    assert wrapper._facade is not None
    return cast(_TrackingFacade, wrapper._facade)


async def test_send_wrapper_closes_once_on_send_exception(create_scope: Callable[..., Scope]) -> None:
    sent: list[Message] = []
    calls = {"count": 0}

    async def failing_send(message: Message) -> None:
        calls["count"] += 1
        if calls["count"] == 3:
            raise RuntimeError("client disconnected")
        sent.append(message)

    wrapper = _build_tracking_wrapper(failing_send, create_scope(), stream_minimum_size=False)

    await wrapper(HTTPResponseStartEvent(type="http.response.start", status=200, headers=[]))
    await wrapper(HTTPResponseBodyEvent(type="http.response.body", body=b"abc", more_body=True))
    with pytest.raises(RuntimeError, match="client disconnected"):
        await wrapper(HTTPResponseBodyEvent(type="http.response.body", body=b"def", more_body=False))

    facade = _tracking_facade(wrapper)
    assert facade.close_calls == 1
    assert wrapper._buffer is None
    assert wrapper._state == "finished"

    # messages arriving after the abort must not be forwarded
    before = len(sent)
    await wrapper(HTTPResponseBodyEvent(type="http.response.body", body=b"late", more_body=False))
    assert len(sent) == before
    assert facade.close_calls == 1


async def test_send_wrapper_closes_once_on_disconnect(create_scope: Callable[..., Scope]) -> None:
    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    wrapper = _build_tracking_wrapper(send, create_scope(), stream_minimum_size=False)

    await wrapper(HTTPResponseStartEvent(type="http.response.start", status=200, headers=[]))
    await wrapper(HTTPResponseBodyEvent(type="http.response.body", body=b"abc", more_body=True))
    await wrapper({"type": "http.disconnect"})

    facade = _tracking_facade(wrapper)
    assert facade.close_calls == 1
    assert wrapper._state == "finished"

    # the un-finished final chunk is discarded, not emitted as a half frame
    finalized = len(sent)
    await wrapper(HTTPResponseBodyEvent(type="http.response.body", body=b"def", more_body=False))
    assert len(sent) == finalized
    last_sent = cast(HTTPResponseBodyEvent, sent[-1])
    assert last_sent["body"] and not last_sent["body"].endswith(b"END")


async def test_send_wrapper_no_resources_created_when_disconnected_before_body(
    create_scope: Callable[..., Scope],
) -> None:
    async def send(message: Message) -> None:
        raise AssertionError("nothing should be sent")

    wrapper = _build_tracking_wrapper(send, create_scope(), stream_minimum_size=False)
    await wrapper(HTTPResponseStartEvent(type="http.response.start", status=200, headers=[]))
    await wrapper({"type": "http.disconnect"})

    assert wrapper._facade is None
    assert wrapper._buffer is None
    assert wrapper._state == "finished"


async def test_send_wrapper_aborts_when_compression_fails(create_scope: Callable[..., Scope]) -> None:
    async def send(message: Message) -> None:
        raise AssertionError("nothing should be sent after a compression failure")

    wrapper = _build_tracking_wrapper(send, create_scope(), stream_minimum_size=False)

    def failing_write(body: bytes | bytearray, final: bool = False) -> None:
        raise RuntimeError("compression boom")

    await wrapper(HTTPResponseStartEvent(type="http.response.start", status=200, headers=[]))

    original_create = wrapper._create_compressor

    def create_and_break() -> CompressionFacade:
        facade = original_create()
        facade.write = failing_write  # type: ignore[method-assign]
        return facade

    wrapper._create_compressor = create_and_break  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="compression boom"):
        await wrapper(HTTPResponseBodyEvent(type="http.response.body", body=b"abc", more_body=True))

    assert wrapper._state == "finished"
    assert wrapper._buffer is None


def test_multi_backend_streaming_output_is_valid_gzip() -> None:
    @get("/big-stream")
    def big_stream() -> Stream:
        return Stream(streaming_iter(content=b"_litestar_" * 400, count=10))

    with create_test_client(
        route_handlers=[big_stream], compression_config=CompressionConfig(backends=("gzip",))
    ) as client:
        with client.stream("GET", "/big-stream", headers={"Accept-Encoding": "gzip"}) as response:
            assert response.status_code == HTTP_200_OK
            assert response.headers["Content-Encoding"] == "gzip"
            assert "".join(response.iter_text()) == "_litestar_" * 4000


class ZlibDeflateFacade(CompressionFacade):
    encoding = "deflate"

    def __init__(self, buffer: BytesIO, compression_encoding: str, config: CompressionConfig) -> None:
        self.buffer = buffer
        self.compression_encoding = compression_encoding
        self.config = config

    def write(self, body: Union[bytes, bytearray], final: bool = False) -> None:
        self.buffer.write(zlib.compress(bytes(body), level=self.config.backend_config["level"]))

    def close(self) -> None: ...
