from __future__ import annotations

import re
import secrets
from calendar import timegm
from contextlib import suppress
from dataclasses import dataclass
from email.utils import formatdate, parsedate
from mimetypes import encodings_map, guess_type
from typing import TYPE_CHECKING, Literal
from urllib.parse import quote
from zlib import adler32

from litestar.constants import ONE_MEGABYTE
from litestar.exceptions import ImproperlyConfiguredException
from litestar.file_system import (
    BaseFileSystem,
    FileSystemRegistry,
    maybe_wrap_fsspec_file_system,
)
from litestar.response.base import Response
from litestar.response.streaming import ASGIStreamingResponse
from litestar.status_codes import (
    HTTP_200_OK,
    HTTP_206_PARTIAL_CONTENT,
    HTTP_304_NOT_MODIFIED,
    HTTP_400_BAD_REQUEST,
    HTTP_412_PRECONDITION_FAILED,
    HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
)
from litestar.utils.helpers import get_enum_string_value

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterable
    from os import PathLike

    from anyio import Path
    from fsspec import AbstractFileSystem
    from fsspec.asyn import AsyncFileSystem as AbstractAsyncFileSystem

    from litestar.background_tasks import BackgroundTask, BackgroundTasks
    from litestar.connection import Request
    from litestar.datastructures.cookie import Cookie
    from litestar.datastructures.headers import ETag
    from litestar.enums import MediaType
    from litestar.file_system import FileInfo
    from litestar.types import (
        HTTPResponseBodyEvent,
        PathType,
        Receive,
        ResponseCookies,
        ResponseHeaders,
        Scope,
        Send,
        TypeEncodersMap,
    )

__all__ = (
    "ASGIFileResponse",
    "File",
    "create_etag_for_file",
)

# brotli not supported in 'mimetypes.encodings_map' until py 3.9.
encodings_map[".br"] = "br"

#: status codes that must never carry a response body
_NO_BODY_STATUS_CODES = frozenset(
    {
        HTTP_304_NOT_MODIFIED,
        HTTP_400_BAD_REQUEST,
        HTTP_412_PRECONDITION_FAILED,
        HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
    }
)

#: outcome of evaluating the request preconditions / range against the file snapshot
_MODE_FULL = "full"
_MODE_NOT_MODIFIED = "not-modified"
_MODE_PRECONDITION_FAILED = "precondition-failed"
_MODE_MALFORMED_RANGE = "malformed-range"
_MODE_RANGE_NOT_SATISFIABLE = "range-not-satisfiable"
_MODE_PARTIAL = "partial"
_MODE_MULTIPART = "multipart"

_ENTITY_TAG_PATTERN = re.compile(r'\s*(?P<weak>[Ww]/)?("(?P<value>[^"]*)")\s*')
_ENTITY_TAG_VALUE_PATTERN = re.compile(r'(?:[Ww]/)?"(?P<value>[^"]*)"')


def create_etag_for_file(path: PathType, modified_time: float | None, file_size: int) -> str:
    """Create an etag.

    Notes:
        - Function is derived from flask.

    Returns:
        An etag.
    """
    check = adler32(str(path).encode("utf-8")) & 0xFFFFFFFF
    parts = [str(file_size), str(check)]
    if modified_time:
        parts.insert(0, str(modified_time))
    return f'"{"-".join(parts)}"'


def _is_ascii_digits(value: str) -> bool:
    return bool(value) and all("0" <= char <= "9" for char in value)


def _parse_http_date(value: str) -> int | None:
    """Parse an HTTP-date into an integer number of seconds since the epoch.

    HTTP-dates have a resolution of one second. Return ``None`` if the value cannot
    be parsed as an HTTP-date.
    """
    parsed = parsedate(value)
    if parsed is None:
        return None
    return timegm(parsed)


def _etag_opaque_value(etag: str) -> str:
    """Return the opaque value of a quoted entity tag, e.g. ``x`` for ``"x"`` or ``W/"x"``."""
    match = _ENTITY_TAG_VALUE_PATTERN.fullmatch(etag.strip())
    return match.group("value") if match else etag


def _parse_entity_tags(value: str) -> list[tuple[bool, str]] | None:
    """Parse a comma-separated list of entity tags.

    Returns a list of ``(weak, opaque_value)`` tuples or ``None`` if the value is not
    a syntactically valid entity-tag list.
    """
    tags: list[tuple[bool, str]] = []
    position = 0
    length = len(value)
    while position < length:
        match = _ENTITY_TAG_PATTERN.match(value, position)
        if not match:
            return None
        tags.append((bool(match.group("weak")), match.group("value")))
        position = match.end()
        if position < length:
            if value[position] != ",":
                return None
            position += 1
    return tags or None


def _if_match_passes(header_value: str, etag: str) -> bool | None:
    """Evaluate an ``If-Match`` header using strong comparison.

    Returns ``True``/``False`` for the precondition outcome or ``None`` if the header
    cannot be evaluated and therefore must be ignored.
    """
    value = header_value.strip()
    if value == "*":
        return True
    tags = _parse_entity_tags(value)
    if tags is None:
        return None
    return any(not weak and opaque == etag for weak, opaque in tags)


def _if_none_match_matches(header_value: str, etag: str) -> bool | None:
    """Evaluate an ``If-None-Match`` header using weak comparison.

    Returns ``True``/``False`` for the precondition outcome or ``None`` if the header
    cannot be evaluated and therefore must be ignored.
    """
    value = header_value.strip()
    if value == "*":
        return True
    tags = _parse_entity_tags(value)
    if tags is None:
        return None
    return any(opaque == etag for _weak, opaque in tags)


def _if_range_is_fresh(header_value: str, etag: str, modified_time: float | None) -> bool | None:
    """Evaluate an ``If-Range`` header.

    Returns ``True`` when the requested range should be served, ``False`` when the full
    representation should be sent instead, and ``None`` when the header cannot be
    evaluated and therefore must be ignored.
    """
    value = header_value.strip()
    if '"' in value:
        # entity-tag form: a weak validator is invalid in If-Range -> ignore the header
        match = _ENTITY_TAG_PATTERN.fullmatch(value)
        if match is None or match.group("weak"):
            return None
        return match.group("value") == etag

    if modified_time is None:
        return None
    timestamp = _parse_http_date(value)
    if timestamp is None:
        return None
    return int(modified_time) <= timestamp


class MalformedRangeHeaderError(ValueError):
    """Raised when a ``Range`` header is syntactically invalid."""


@dataclass(frozen=True)
class _ByteRange:
    """A closed byte range, i.e. both bounds are inclusive offsets."""

    start: int
    end: int

    def __len__(self) -> int:
        return self.end - self.start + 1


def _parse_byte_spec(spec: str, file_size: int, header_value: str) -> _ByteRange | None:
    """Parse a single byte-range spec (e.g. ``0-499``, ``-100`` or ``500-``).

    Returns ``None`` if the spec is syntactically valid but cannot be satisfied.

    Raises:
        MalformedRangeHeaderError: if the spec is syntactically invalid.
    """
    first, _, last = spec.partition("-")
    first = first.strip()
    last = last.strip()

    if first == "":
        # suffix range: the last N bytes
        if not _is_ascii_digits(last):
            raise MalformedRangeHeaderError(header_value)
        suffix_length = int(last)
        if suffix_length == 0 or file_size == 0:
            return None
        return _ByteRange(start=max(0, file_size - suffix_length), end=file_size - 1)

    if not _is_ascii_digits(first):
        raise MalformedRangeHeaderError(header_value)
    start = int(first)
    if start >= file_size:
        return None

    if last == "":
        return _ByteRange(start=start, end=file_size - 1)

    if not _is_ascii_digits(last):
        raise MalformedRangeHeaderError(header_value)
    end = int(last)
    if end < start:
        # an inverted range cannot be satisfied
        return None
    return _ByteRange(start=start, end=min(end, file_size - 1))


def _coalesce_byte_ranges(ranges: list[_ByteRange]) -> list[_ByteRange]:
    """Sort ranges by start and coalesce duplicates, overlapping and adjacent ranges."""
    ranges.sort(key=lambda byte_range: byte_range.start)
    merged = [ranges[0]]
    for byte_range in ranges[1:]:
        previous = merged[-1]
        if byte_range.start <= previous.end + 1:
            merged[-1] = _ByteRange(start=previous.start, end=max(previous.end, byte_range.end))
        else:
            merged.append(byte_range)
    return merged


def parse_byte_ranges(header_value: str, file_size: int) -> list[_ByteRange] | None:
    """Parse a ``Range`` header value for the ``bytes`` unit against ``file_size``.

    The returned ranges are normalized: clipped to the file size, sorted by start,
    duplicates dropped, and overlapping / adjacent ranges coalesced, giving callers a
    single uniform policy for multiple or repeated ranges.

    Returns:
        - ``None`` if the range unit is not ``bytes`` (the header must be ignored)
        - an empty list if no requested range is satisfiable (``416``)
        - a list of one or more normalized byte ranges

    Raises:
        MalformedRangeHeaderError: if the header is syntactically invalid.
    """
    unit, separator, specs = header_value.partition("=")
    if unit.strip().lower() != "bytes":
        # a range unit this server does not support is not an error: ignore the header
        return None
    if not separator or not specs.strip():
        raise MalformedRangeHeaderError(header_value)

    ranges: list[_ByteRange] = []
    for raw_spec in specs.split(","):
        spec = raw_spec.strip()
        if not spec or "-" not in spec:
            raise MalformedRangeHeaderError(header_value)
        byte_range = _parse_byte_spec(spec, file_size, header_value)
        if byte_range is not None:
            ranges.append(byte_range)

    return _coalesce_byte_ranges(ranges) if ranges else []


def _get_request_header(scope: Scope, name: str) -> str | None:
    """Get a request header value from an ASGI scope.

    Multiple occurrences of the header are joined with ``,`` as HTTP allows combining
    them into one comma-separated list.
    """
    encoded_name = name.encode("latin-1")
    values = [value.decode("latin-1") for header_name, value in scope["headers"] if header_name == encoded_name]
    return ", ".join(values) if values else None


class _FileChangedDuringTransferError(RuntimeError):
    """Raised when the served file changes (e.g. is replaced or truncated) while its
    contents are being streamed.

    Raising aborts the response before the terminating body event is sent, so the
    client never receives a complete response that splices together bytes from
    different versions of the file.
    """


class ASGIFileResponse(ASGIStreamingResponse):
    """A low-level ASGI response, streaming a file as response body."""

    __slots__ = (
        "_base_media_type",
        "_boundary",
        "_file_system",
        "_mode",
        "_ranges",
        "_snapshot",
        "_user_header_names",
        "chunk_size",
        "etag",
        "file_info",
        "file_path",
    )

    def __init__(
        self,
        *,
        background: BackgroundTask | BackgroundTasks | None = None,
        chunk_size: int = ONE_MEGABYTE,
        content_disposition_type: Literal["attachment", "inline"] = "attachment",
        content_length: int | None = None,
        cookies: Iterable[Cookie] | None = None,
        encoding: str = "utf-8",
        etag: ETag | None = None,
        file_info: FileInfo | None = None,
        file_path: str | PathLike | Path,
        file_system: BaseFileSystem,
        filename: str = "",
        headers: dict[str, str] | None = None,
        is_head_response: bool = False,
        media_type: MediaType | str | None = None,
        status_code: int | None = None,
    ) -> None:
        """A low-level ASGI response, streaming a file as response body.

        Args:
            background: A background task or a list of background tasks to be executed after the response is sent.
            chunk_size: The chunk size to use.
            content_disposition_type: The type of the ``Content-Disposition``. Either ``inline`` or ``attachment``.
            content_length: The response content length.
            cookies: The response cookies.
            encoding: The response encoding.
            etag: An etag.
            file_info: A file info.
            file_path: A path to a file.
            file_system: A file system adapter.
            filename: The name of the file.
            headers: A dictionary of headers.
            headers: The response headers.
            is_head_response: A boolean indicating if the response is a HEAD response.
            media_type: The media type of the file.
            status_code: The response status code.
        """
        headers = headers or {}
        self._user_header_names = frozenset(key.lower() for key in headers)
        if not media_type:
            mimetype, content_encoding = guess_type(filename) if filename else (None, None)
            media_type = mimetype or "application/octet-stream"
            if content_encoding is not None:
                headers.update({"content-encoding": content_encoding})

        self._base_media_type = media_type
        self._file_system = file_system
        self._mode = _MODE_FULL
        self._ranges: tuple[_ByteRange, ...] | None = None
        self._boundary: str | None = None
        self._snapshot: FileInfo | None = None

        super().__init__(
            iterator=iter(b""),
            headers=headers,
            media_type=media_type,
            cookies=cookies,
            background=background,
            status_code=status_code,
            content_length=content_length,
            encoding=encoding,
            is_head_response=is_head_response,
        )

        quoted_filename = quote(filename)
        is_utf8 = quoted_filename == filename
        if is_utf8:
            content_disposition = f'{content_disposition_type}; filename="{filename}"'
        else:
            content_disposition = f"{content_disposition_type}; filename*=utf-8''{quoted_filename}"

        self.headers.setdefault("content-disposition", content_disposition)

        self.chunk_size = chunk_size
        self.etag = etag
        self.file_path = file_path
        self.file_info = file_info

    async def _resolve_file_info(self) -> FileInfo:
        """Resolve and cache the file information snapshot for this response.

        All conditional evaluation, range evaluation and reads for a given request are
        based on this single snapshot.
        """
        if self.file_info is None:
            try:
                self.file_info = await self._file_system.info(self.file_path)
            except FileNotFoundError as e:
                raise ImproperlyConfiguredException(f"{self.file_path} does not exist") from e

        if self.file_info["type"] != "file":
            raise ImproperlyConfiguredException(f"{self.file_path} is not a file")

        self._snapshot = self.file_info
        return self.file_info

    def _effective_validators(self, file_info: FileInfo) -> tuple[str, float | None]:
        """Return the effective ``(etag, modified_time)`` validators.

        These are the same values that will be sent as the ``ETag`` response header
        (an explicitly provided ``etag`` header taking precedence) and the file
        modification time used for date based preconditions.
        """
        etag = self.headers.get("etag")
        if etag is None:
            if self.etag:
                etag = self.etag.to_header()
            else:
                etag = create_etag_for_file(
                    path=self.file_path,
                    modified_time=file_info.get("mtime"),
                    file_size=file_info["size"],
                )
        return _etag_opaque_value(etag), file_info.get("mtime")

    @staticmethod
    def _date_precondition_fails(header_value: str | None, modified_time: float | None, *, after: bool) -> bool:
        """Evaluate an ``If-(Un)modified-Since`` precondition.

        ``after=True`` corresponds to ``If-Unmodified-Since`` (failure when the
        modification is later), ``after=False`` to ``If-Modified-Since`` (failure /
        not-modified when the modification is not later).
        """
        if header_value is None or modified_time is None:
            return False
        timestamp = _parse_http_date(header_value)
        if timestamp is None:
            return False
        return int(modified_time) > timestamp if after else int(modified_time) <= timestamp

    def _evaluate_preconditions(
        self,
        scope: Scope,
        etag: str,
        modified_time: float | None,
    ) -> str | None:
        """Evaluate the request preconditions following RFC 9110 precedence.

        Returns the response mode (``not-modified`` / ``precondition-failed``) if a
        precondition short-circuits the response, or ``None`` if processing should
        continue.
        """
        if_match = _get_request_header(scope, "if-match")
        if if_match is not None:
            # 1. If-Match (strong precondition)
            if _if_match_passes(if_match, etag) is False:
                return _MODE_PRECONDITION_FAILED
        elif self._date_precondition_fails(
            _get_request_header(scope, "if-unmodified-since"),
            modified_time,
            after=True,
        ):
            # 2. If-Unmodified-Since (ignored when If-Match is present)
            return _MODE_PRECONDITION_FAILED

        if_none_match = _get_request_header(scope, "if-none-match")
        if if_none_match is not None:
            # 3. If-None-Match (weak comparison, takes precedence over If-Modified-Since)
            if _if_none_match_matches(if_none_match, etag):
                return _MODE_NOT_MODIFIED
        elif self._date_precondition_fails(
            _get_request_header(scope, "if-modified-since"),
            modified_time,
            after=False,
        ):
            # 4. If-Modified-Since (ignored when If-None-Match is present)
            return _MODE_NOT_MODIFIED

        return None

    def _evaluate_range(self, scope: Scope, file_info: FileInfo, etag: str, modified_time: float | None) -> str:
        """Evaluate the ``Range`` header (with an optional ``If-Range`` gate).

        Returns the response mode and populates ``self._ranges`` / ``self._boundary``
        for range responses.
        """
        range_header = _get_request_header(scope, "range")
        if range_header is None:
            return _MODE_FULL

        if_range = _get_request_header(scope, "if-range")
        if if_range is not None and _if_range_is_fresh(if_range, etag, modified_time) is False:
            # representation changed: send the full representation
            self.headers.setdefault("accept-ranges", "bytes")
            return _MODE_FULL

        try:
            ranges = parse_byte_ranges(range_header, file_size=file_info["size"])
        except MalformedRangeHeaderError:
            return _MODE_MALFORMED_RANGE

        if ranges is None:
            # unsupported range unit: ignore the Range header entirely
            return _MODE_FULL

        if not ranges:
            return _MODE_RANGE_NOT_SATISFIABLE

        self.headers.setdefault("accept-ranges", "bytes")
        if len(ranges) == 1:
            self._ranges = (ranges[0],)
            return _MODE_PARTIAL

        self._ranges = tuple(ranges)
        self._boundary = secrets.token_hex(16)
        return _MODE_MULTIPART

    def _evaluate_conditions_and_ranges(self, scope: Scope, file_info: FileInfo) -> None:
        """Evaluate request preconditions and the byte range against the file snapshot.

        The result is stored on the response as ``self._mode`` and, for range
        responses, ``self._ranges`` / ``self._boundary``.
        """
        etag, modified_time = self._effective_validators(file_info)

        precondition_mode = self._evaluate_preconditions(scope, etag, modified_time)
        if precondition_mode is not None:
            self._mode = precondition_mode
            return

        self._mode = self._evaluate_range(scope, file_info, etag, modified_time)

    def _finalize_representation_headers(self, file_info: FileInfo) -> None:
        """Set the default representation headers (length, last-modified, etag).

        Mirrors the headers sent by a regular full file response and is applied for
        every response mode, so validation headers are retained on 304/412/416
        responses as well.
        """
        self.content_length = file_info["size"]

        self.headers.setdefault("content-length", str(self.content_length))
        mtime = file_info.get("mtime")

        if mtime is not None:
            self.headers.setdefault("last-modified", formatdate(mtime, usegmt=True))

        if self.etag:
            self.headers.setdefault("etag", self.etag.to_header())
        else:
            self.headers.setdefault(
                "etag",
                create_etag_for_file(
                    path=self.file_path,
                    modified_time=mtime,
                    file_size=file_info["size"],
                ),
            )

    def _delete_auto_header(self, name: str) -> None:
        """Delete a header that was generated for this response, leaving headers
        explicitly provided by the user untouched.
        """
        if name not in self._user_header_names:
            with suppress(KeyError):
                del self.headers[name]

    def _apply_response_mode(self, file_info: FileInfo) -> None:
        """Transform status code and headers according to the evaluated response mode."""
        file_size = file_info["size"]

        if self._mode in (_MODE_NOT_MODIFIED, _MODE_PRECONDITION_FAILED, _MODE_RANGE_NOT_SATISFIABLE):
            # none of these responses carry a body
            self._delete_auto_header("content-length")
            self._delete_auto_header("content-type")

        if self._mode == _MODE_NOT_MODIFIED:
            self.status_code = HTTP_304_NOT_MODIFIED
            self.headers.setdefault("accept-ranges", "bytes")
            return

        if self._mode == _MODE_PRECONDITION_FAILED:
            self.status_code = HTTP_412_PRECONDITION_FAILED
            return

        if self._mode == _MODE_MALFORMED_RANGE:
            self.status_code = HTTP_400_BAD_REQUEST
            self._delete_auto_header("content-length")
            self._delete_auto_header("content-type")
            return

        if self._mode == _MODE_RANGE_NOT_SATISFIABLE:
            self.status_code = HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE
            self.headers["content-range"] = f"bytes */{file_size}"
            self.headers.setdefault("accept-ranges", "bytes")
            return

        if self._mode == _MODE_PARTIAL:
            byte_range = self._ranges[0]  # type: ignore[index]
            length = len(byte_range)
            self.status_code = HTTP_206_PARTIAL_CONTENT
            self.content_length = length
            self.headers["content-length"] = str(length)
            self.headers["content-range"] = f"bytes {byte_range.start}-{byte_range.end}/{file_size}"
            self.headers.setdefault("accept-ranges", "bytes")
            return

        if self._mode == _MODE_MULTIPART:
            boundary = self._boundary  # type: ignore[assignment]
            self.status_code = HTTP_206_PARTIAL_CONTENT
            self.content_length = self._calculate_multipart_length(file_size)
            self.headers["content-type"] = f"multipart/byteranges; boundary={boundary}"
            self.headers["content-length"] = str(self.content_length)
            self.headers.setdefault("accept-ranges", "bytes")

    def _calculate_multipart_length(self, file_size: int) -> int:
        ranges = self._ranges or ()
        boundary_length = len(self._boundary or "")
        content_length = 0
        for byte_range in ranges:
            # --<boundary>\r\nContent-Type: <media>\r\nContent-Range: bytes s-e/size\r\n\r\n
            content_length += 2 + boundary_length + 2
            content_length += len("Content-Type: ") + len(self._base_media_type.encode("ascii", errors="replace")) + 2
            content_length += len(
                f"Content-Range: bytes {byte_range.start}-{byte_range.end}/{file_size}\r\n\r\n".encode("ascii")
            )
            # the segment itself plus the CRLF after it
            content_length += len(byte_range) + 2
        # closing delimiter: --<boundary>--\r\n
        content_length += 2 + boundary_length + 4
        return content_length

    async def _verify_snapshot_unchanged(self) -> None:
        """Re-check that the file still matches the snapshot taken before the response
        started, aborting the transfer if it was replaced or truncated in the meantime.
        """
        snapshot = self._snapshot
        try:
            current = await self._file_system.info(self.file_path)
        except FileNotFoundError as e:
            raise _FileChangedDuringTransferError(f"{self.file_path} was removed during the transfer") from e
        if current.get("size") != (snapshot or {}).get("size") or current.get("mtime") != (snapshot or {}).get("mtime"):
            raise _FileChangedDuringTransferError(f"{self.file_path} changed during the transfer")

    async def _guarded_iter(
        self, iterator: AsyncGenerator[bytes, None], expected_length: int
    ) -> AsyncGenerator[bytes, None]:
        """Yield chunks from ``iterator`` while ensuring exactly ``expected_length``
        bytes are produced and that the file still matches the snapshot afterwards.
        """
        received = 0
        try:
            async for chunk in iterator:
                received += len(chunk)
                if received > expected_length:
                    raise _FileChangedDuringTransferError(f"{self.file_path} grew during the transfer")
                yield chunk
        finally:
            await iterator.aclose()

        if received != expected_length:
            raise _FileChangedDuringTransferError(
                f"{self.file_path} was truncated during the transfer: expected {expected_length} bytes, got {received}"
            )
        await self._verify_snapshot_unchanged()

    async def _iter_multipart(self) -> AsyncGenerator[bytes, None]:
        """Stream the parts of a ``multipart/byteranges`` response."""
        snapshot_size = self._snapshot["size"]  # type: ignore[index]
        boundary = (self._boundary or "").encode("ascii")
        media_type = self._base_media_type.encode("ascii", errors="replace")
        ranges = self._ranges or ()
        produced = 0

        for byte_range in ranges:
            prologue = (
                b"--"
                + boundary
                + b"\r\nContent-Type: "
                + media_type
                + f"\r\nContent-Range: bytes {byte_range.start}-{byte_range.end}/{snapshot_size}\r\n\r\n".encode(
                    "ascii"
                )
            )
            produced += len(prologue)
            yield prologue

            expected_length = len(byte_range)
            received = 0
            async for chunk in self._guarded_iter(
                self._file_system.iter(
                    self.file_path,
                    chunksize=self.chunk_size,
                    start=byte_range.start,
                    end=byte_range.end + 1,
                ),
                expected_length=expected_length,
            ):
                received += len(chunk)
                produced += len(chunk)
                yield chunk
            if received != expected_length:
                raise _FileChangedDuringTransferError(f"{self.file_path} changed during the transfer")

            produced += 2
            yield b"\r\n"

        closing = b"--" + boundary + b"--\r\n"
        produced += len(closing)
        yield closing

        if produced != self.content_length:
            raise _FileChangedDuringTransferError(f"{self.file_path} changed during the transfer")
        await self._verify_snapshot_unchanged()

    async def send_body(self, send: Send, receive: Receive) -> None:
        """Emit a stream of events correlating with the response body.

        Args:
            send: The ASGI send function.
            receive: The ASGI receive function.

        Returns:
            None
        """
        snapshot_size = self._snapshot["size"] if self._snapshot else (self.content_length or 0)  # type: ignore[index]

        if self._mode == _MODE_MULTIPART:
            self.iterator = self._iter_multipart()
            await super().send_body(send=send, receive=receive)
            return

        if self._mode == _MODE_PARTIAL:
            byte_range = self._ranges[0]  # type: ignore[index]
            start = byte_range.start
            end = byte_range.end + 1  # the file system offsets are exclusive at the end
            length = len(byte_range)
        else:
            start = 0
            end = snapshot_size
            length = snapshot_size

        if length < self.chunk_size:
            # no need to chunk and stream; read and send the whole segment in one go
            body = await self._file_system.read_bytes(self.file_path, start=start, end=end)
            if len(body) != length:
                raise _FileChangedDuringTransferError(
                    f"{self.file_path} changed during the transfer: expected {length} bytes, got {len(body)}"
                )
            # verify the snapshot still holds before emitting any body bytes, so a
            # replaced / truncated file never results in a complete spliced response
            await self._verify_snapshot_unchanged()
            body_event: HTTPResponseBodyEvent = {
                "type": "http.response.body",
                "body": body,
                "more_body": False,
            }
            await send(body_event)
            return

        self.iterator = self._guarded_iter(
            self._file_system.iter(self.file_path, chunksize=self.chunk_size, start=start, end=end),
            expected_length=length,
        )
        await super().send_body(send=send, receive=receive)

    async def start_response(self, send: Send) -> None:
        """Emit the start event of the response. This event includes the headers and status codes.

        Args:
            send: The ASGI send function.

        Returns:
            None
        """
        file_info = await self._resolve_file_info()
        self._finalize_representation_headers(file_info)
        await super().start_response(send=send)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """ASGI callable of the file response.

        Request preconditions (``If-Match``, ``If-None-Match``,
        ``If-Modified-Since``, ``If-Unmodified-Since``) and byte ranges are evaluated
        once, against a single snapshot of the file, before the response starts.

        Args:
            scope: The ASGI connection scope.
            receive: The ASGI receive function.
            send: The ASGI send function.

        Returns:
            None
        """
        file_info = await self._resolve_file_info()

        if scope.get("method", "GET").upper() in {"GET", "HEAD"} and self.status_code == HTTP_200_OK:
            self._evaluate_conditions_and_ranges(scope=scope, file_info=file_info)

        self._finalize_representation_headers(file_info)
        self._apply_response_mode(file_info=file_info)
        await super().start_response(send=send)

        if self.is_head_response or self.status_code in _NO_BODY_STATUS_CODES:
            event: HTTPResponseBodyEvent = {"type": "http.response.body", "body": b"", "more_body": False}
            await send(event)
        else:
            await self.send_body(send=send, receive=receive)

        await self.after_response()


class File(Response):
    """A response, streaming a file as response body."""

    __slots__ = (
        "chunk_size",
        "content_disposition_type",
        "etag",
        "file_info",
        "file_path",
        "file_system",
        "filename",
    )

    def __init__(
        self,
        path: str | PathLike | Path,
        *,
        background: BackgroundTask | BackgroundTasks | None = None,
        chunk_size: int = ONE_MEGABYTE,
        content_disposition_type: Literal["attachment", "inline"] = "attachment",
        cookies: ResponseCookies | None = None,
        encoding: str = "utf-8",
        etag: ETag | None = None,
        file_info: FileInfo | None = None,
        file_system: str | BaseFileSystem | AbstractFileSystem | AbstractAsyncFileSystem | None = None,
        filename: str | None = None,
        headers: ResponseHeaders | None = None,
        media_type: Literal[MediaType.TEXT] | str | None = None,
        status_code: int | None = None,
    ) -> None:
        """Send a file from a file system.

        Args:
            path: A file path in one of the supported formats.
            background: A :class:`BackgroundTask <.background_tasks.BackgroundTask>` instance or
                :class:`BackgroundTasks <.background_tasks.BackgroundTasks>` to execute after the response is finished.
                Defaults to None.
            chunk_size: The chunk sizes to use when streaming the file. Defaults to 1MB.
            content_disposition_type: The type of the ``Content-Disposition``. Either ``inline`` or ``attachment``.
            cookies: A list of :class:`Cookie <.datastructures.Cookie>` instances to be set under the response
                ``Set-Cookie`` header.
            encoding: The encoding to be used for the response headers.
            etag: An optional :class:`ETag <.datastructures.ETag>` instance. If not provided, an etag will be
                generated.
            file_info: The output of calling :meth:`file_system.info <litestar.file_system.BaseFileSystem.info>`
            file_system: The file system to load the file from. Instances of
                :class:`~litestar.file_system.BaseFileSystem`, :class:`fsspec.spec.AbstractFileSystem`,
                :class:`fsspec.asyn.AsyncFileSystem` will be used directly. If passed string, use it to look up the
                corresponding file system from the :class:`~litestar.file_system.FileSystemRegistry`. If not given,
                the file will be loaded from :attr:`~litestar.file_system.FileSystemRegistry.default`
            filename: An optional filename to set in the header.
            headers: A string keyed dictionary of response headers. Header keys are insensitive.
            media_type: A value for the response ``Content-Type`` header. If not provided, the value will be either
                derived from the filename if provided and supported by the stdlib, or will default to
                ``application/octet-stream``.
            status_code: An HTTP status code.
        """

        self.chunk_size = chunk_size
        self.content_disposition_type = content_disposition_type
        self.etag = etag
        self.file_info = file_info
        self.file_path = path
        self.file_system = file_system
        self.filename = filename or ""

        super().__init__(
            content=None,
            status_code=status_code,
            media_type=media_type,
            background=background,
            headers=headers,
            cookies=cookies,
            encoding=encoding,
        )

    def to_asgi_response(
        self,
        request: Request,
        *,
        background: BackgroundTask | BackgroundTasks | None = None,
        cookies: Iterable[Cookie] | None = None,
        headers: dict[str, str] | None = None,
        is_head_response: bool = False,
        media_type: MediaType | str | None = None,
        status_code: int | None = None,
        type_encoders: TypeEncodersMap | None = None,
    ) -> ASGIFileResponse:
        """Create an :class:`ASGIFileResponse <litestar.response.file.ASGIFileResponse>` instance.

        Args:
            background: Background task(s) to be executed after the response is sent.
            cookies: A list of cookies to be set on the response.
            headers: Additional headers to be merged with the response headers. Response headers take precedence.
            is_head_response: Whether the response is a HEAD response.
            media_type: Media type for the response. If ``media_type`` is already set on the response, this is ignored.
            request: The :class:`Request <.connection.Request>` instance.
            status_code: Status code for the response. If ``status_code`` is already set on the response, this is
            type_encoders: A dictionary of type encoders to use for encoding the response content.

        Returns:
            A low-level ASGI file response.
        """

        headers = {**headers, **self.headers} if headers is not None else self.headers

        media_type = self.media_type or media_type
        if media_type is not None:
            media_type = get_enum_string_value(media_type)

        file_system: BaseFileSystem
        if self.file_system is None:
            file_system = request.app.plugins.get(FileSystemRegistry).default
        elif isinstance(self.file_system, str):
            file_system_plugin = request.app.plugins.get(FileSystemRegistry)
            file_system = file_system_plugin[self.file_system]
        else:
            file_system = maybe_wrap_fsspec_file_system(self.file_system)

        return ASGIFileResponse(
            file_path=self.file_path,
            file_system=file_system,
            filename=self.filename,
            background=self.background or background,
            chunk_size=self.chunk_size,
            content_disposition_type=self.content_disposition_type,  # pyright: ignore[reportArgumentType]
            content_length=0,
            cookies=self._merge_cookies(cookies),
            encoding=self.encoding,
            etag=self.etag,
            file_info=self.file_info,
            headers=headers,
            is_head_response=is_head_response,
            media_type=media_type,
            status_code=self.status_code or status_code,
        )
