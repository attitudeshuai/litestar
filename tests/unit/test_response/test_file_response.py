from __future__ import annotations

import asyncio
import os
import pathlib
import secrets
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from email.utils import formatdate
from os import urandom
from pathlib import Path
from typing import Any

import pytest
from fsspec.implementations.local import LocalFileSystem
from pytest_mock import MockerFixture

from litestar import get, head
from litestar.connection.base import empty_send
from litestar.datastructures import ETag
from litestar.exceptions import ImproperlyConfiguredException
from litestar.file_system import BaseFileSystem, BaseLocalFileSystem, FileInfo, FileSystemRegistry
from litestar.response.file import (
    ASGIFileResponse,
    File,
    MalformedRangeHeaderError,
    _ByteRange,
    parse_byte_ranges,
)
from litestar.status_codes import (
    HTTP_200_OK,
    HTTP_206_PARTIAL_CONTENT,
    HTTP_304_NOT_MODIFIED,
    HTTP_400_BAD_REQUEST,
    HTTP_412_PRECONDITION_FAILED,
    HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
    HTTP_500_INTERNAL_SERVER_ERROR,
)
from litestar.testing import create_test_client
from litestar.types import PathType


@pytest.mark.parametrize("content_disposition_type", ("inline", "attachment"))
def test_file_response_default_content_type(tmpdir: Path, content_disposition_type: Any) -> None:
    path = Path(tmpdir / "image.png")
    path.write_bytes(b"")

    @get("/")
    def handler() -> File:
        return File(path=path, content_disposition_type=content_disposition_type)

    with create_test_client(handler, openapi_config=None) as client:
        response = client.get("/")
        assert response.status_code == HTTP_200_OK
        assert response.headers["content-type"] == "application/octet-stream"
        assert response.headers["content-disposition"] == f'{content_disposition_type}; filename=""'


@pytest.mark.parametrize("content_disposition_type", ("inline", "attachment"))
def test_file_response_infer_content_type(tmpdir: Path, content_disposition_type: Any) -> None:
    path = Path(tmpdir / "image.png")
    path.write_bytes(b"")

    @get("/")
    def handler() -> File:
        return File(path=path, filename="image.png", content_disposition_type=content_disposition_type)

    with create_test_client(handler) as client:
        response = client.get("/")
        assert response.status_code == HTTP_200_OK
        assert response.headers["content-type"] == "image/png"
        assert response.headers["content-disposition"] == f'{content_disposition_type}; filename="image.png"'


@pytest.mark.parametrize("filename, expected", (("Jacky Chen", "Jacky%20Chen"), ("成龍", "%E6%88%90%E9%BE%8D")))
def test_filename(tmpdir: Path, filename: str, expected: str) -> None:
    path = Path(tmpdir / f"{filename}.txt")
    path.write_bytes(b"")

    @get("/")
    def handler() -> File:
        return File(path=path, filename=f"{filename}.txt")

    with create_test_client(handler) as client:
        response = client.get("/")
        assert response.status_code == HTTP_200_OK
        assert response.headers["content-disposition"] == f"attachment; filename*=utf-8''{expected}.txt"


def test_file_response_content_length(tmpdir: Path) -> None:
    content = urandom(1024 * 10)
    path = Path(tmpdir / "file.txt")
    path.write_bytes(content)

    @get("/")
    def handler() -> File:
        return File(path=path)

    with create_test_client(handler) as client:
        response = client.get("/")
        assert response.status_code == HTTP_200_OK
        assert response.content == content
        assert response.headers["content-length"] == str(len(content))


def test_file_response_last_modified(tmpdir: Path) -> None:
    path = Path(tmpdir / "file.txt")
    path.write_bytes(b"")

    @get("/")
    def handler() -> File:
        return File(path=path, filename="image.png")

    with create_test_client(handler) as client:
        response = client.get("/")
        assert response.status_code == HTTP_200_OK
        assert response.headers["last-modified"].lower() == formatdate(path.stat().st_mtime, usegmt=True).lower()


@pytest.mark.parametrize(
    "mtime,expected_last_modified",
    [
        pytest.param(
            datetime(2000, 1, 2, 3, 4, 5, tzinfo=UTC).timestamp(),
            "Sun, 02 Jan 2000 03:04:05 GMT",
            id="timestamp",
        ),
        pytest.param(datetime(2000, 1, 2, 3, 4, 5, tzinfo=UTC), "Sun, 02 Jan 2000 03:04:05 GMT", id="datetime"),
        pytest.param(
            datetime(2000, 1, 2, 3, 4, 5, tzinfo=UTC).isoformat(),
            "Sun, 02 Jan 2000 03:04:05 GMT",
            id="isoformat",
        ),
    ],
)
@pytest.mark.parametrize(
    "mtime_key",
    [
        "mtime",
        "ctime",
        "Last-Modified",
        "updated_at",
        "modification_time",
        "last_changed",
        "change_time",
        "last_modified",
        "last_updated",
        "timestamp",
    ],
)
def test_file_response_last_modified_file_info_formats(
    tmpdir: Path,
    mtime: Any,
    mtime_key: str,
    expected_last_modified: str,
    mocker: MockerFixture,
) -> None:
    path = Path(tmpdir / "file.txt")
    path.write_bytes(b"")
    file_info = {"name": "file.txt", "size": 0, "type": "file", mtime_key: mtime}
    fs = LocalFileSystem()
    mocker.patch.object(fs, "info", return_value=file_info)

    @get("/")
    def handler() -> File:
        return File(path=path, filename="image.png", file_system=fs)

    with create_test_client(handler) as client:
        response = client.get("/")
        assert response.status_code == HTTP_200_OK
        assert response.headers["last-modified"].lower() == expected_last_modified.lower()


def test_file_response_last_modified_unsupported_mtime_type(
    tmpdir: Path,
    mocker: MockerFixture,
) -> None:
    path = Path(tmpdir / "file.txt")
    path.write_bytes(b"")
    file_info = {"name": "file.txt", "size": 0, "type": "file", "last_updated": object()}

    fs = LocalFileSystem()
    mocker.patch.object(fs, "info", return_value=file_info)

    @get("/")
    def handler() -> File:
        return File(
            path=path,
            filename="image.png",
            file_system=fs,
        )

    with create_test_client(handler) as client:
        response = client.get("/")
        assert response.status_code == HTTP_500_INTERNAL_SERVER_ERROR
        assert "last-modified" not in response.headers


def test_file_response_last_modified_mtime_not_given(
    tmpdir: Path,
    mocker: MockerFixture,
) -> None:
    path = Path(tmpdir / "file.txt")
    path.write_bytes(b"")
    file_info = {"name": "file.txt", "size": 0, "type": "file"}

    fs = LocalFileSystem()
    mocker.patch.object(fs, "info", return_value=file_info)

    @get("/")
    def handler() -> File:
        return File(
            path=path,
            filename="image.png",
            file_system=fs,
        )

    with create_test_client(handler) as client:
        response = client.get("/")
        assert response.status_code == HTTP_200_OK
        assert "last-modified" not in response.headers


def test_file_response_etag_without_mtime(
    tmpdir: Path,
    mocker: MockerFixture,
) -> None:
    path = Path(tmpdir / "file.txt")
    path.write_bytes(b"")
    file_info = {"name": "file.txt", "size": 0, "type": "file"}

    fs = LocalFileSystem()
    mocker.patch.object(fs, "info", return_value=file_info)

    @get("/")
    def handler() -> File:
        return File(
            path=path,
            filename="image.png",
            file_system=fs,
        )

    with create_test_client(handler) as client:
        response = client.get("/")
        assert response.status_code == HTTP_200_OK
        # we expect etag to only have 2 parts here because no mtime was given
        assert len(response.headers.get("etag", "").split("-")) == 2


async def test_file_response_with_directory_raises_error(tmpdir: Path) -> None:
    with pytest.raises(ImproperlyConfiguredException):
        asgi_response = ASGIFileResponse(file_path=tmpdir, filename="example.png", file_system=BaseLocalFileSystem())
        await asgi_response.start_response(empty_send)


@pytest.mark.parametrize("chunk_size", [4, 8, 16, 256, 512, 1024, 2048])
async def test_file_iterator(tmpdir: Path, chunk_size: int) -> None:
    content = urandom(1024)
    path = Path(tmpdir / "file.txt")
    path.write_bytes(content)
    result = b"".join([chunk async for chunk in BaseLocalFileSystem().iter(path, chunk_size)])
    assert result == content


@pytest.mark.parametrize("size", (1024, 2048, 4096, 1024 * 10, 2048 * 10, 4096 * 10))
def test_large_files(tmpdir: Path, size: int) -> None:
    content = urandom(1024 * size)
    path = Path(tmpdir / "file.txt")
    path.write_bytes(content)

    @get("/")
    def handler() -> File:
        return File(path=path)

    with create_test_client(handler) as client:
        response = client.get("/")
        assert response.status_code == HTTP_200_OK
        assert response.content == content
        assert response.headers["content-length"] == str(len(content))


@pytest.mark.parametrize("file_system", (BaseLocalFileSystem(), LocalFileSystem()))
def test_file_with_different_file_systems(tmpdir: Path, file_system: BaseFileSystem) -> None:
    path = tmpdir / "text.txt"
    path.write_text("content", "utf-8")

    @get("/", media_type="application/octet-stream")
    def handler() -> File:
        return File(
            filename="text.txt",
            path=path,
            file_system=file_system,
        )

    with create_test_client(handler) as client:
        response = client.get("/")
        assert response.status_code == HTTP_200_OK
        assert response.text == "content"
        assert response.headers.get("content-disposition") == 'attachment; filename="text.txt"'


def test_file_with_passed_in_file_info(tmpdir: Path) -> None:
    path = tmpdir / "text.txt"
    path.write_text("content", "utf-8")

    fs = LocalFileSystem()
    fs_info = fs.info(tmpdir / "text.txt")

    assert fs_info

    @get("/", media_type="application/octet-stream")
    def handler() -> File:
        return File(filename="text.txt", path=path, file_system=fs, file_info=fs_info)  # pyright: ignore[reportArgumentType]

    with create_test_client(handler) as client:
        response = client.get("/")
        assert response.status_code == HTTP_200_OK, response.text
        assert response.text == "content"
        assert response.headers.get("content-disposition") == 'attachment; filename="text.txt"'


async def test_file_with_symbolic_link(tmpdir: Path) -> None:
    path = tmpdir / "text.txt"
    path.write_text("content", "utf-8")

    linked = tmpdir / "alt.txt"
    os.symlink(path, linked, target_is_directory=False)

    fs = BaseLocalFileSystem()
    file_info = await fs.info(linked)

    @get("/", media_type="application/octet-stream")
    def handler() -> File:
        return File(filename="alt.txt", path=linked, file_system=fs, file_info=file_info)

    with create_test_client(handler) as client:
        response = client.get("/")
        assert response.status_code == HTTP_200_OK
        assert response.text == "content"
        assert response.headers.get("content-length", "7")
        assert response.headers.get("content-disposition") == 'attachment; filename="alt.txt"'


async def test_file_sets_etag_correctly(tmpdir: Path) -> None:
    path = tmpdir / "file.txt"
    content = b"<file content>"
    Path(path).write_bytes(content)
    etag = ETag(value="special")

    @get("/")
    def handler() -> File:
        return File(path=path, etag=etag)

    with create_test_client(handler) as client:
        response = client.get("/")
        assert response.status_code == HTTP_200_OK
        assert response.headers["etag"] == '"special"'


async def test_file_response_with_missing_file_raises_error(tmpdir: Path) -> None:
    path = tmpdir / "404.txt"
    with pytest.raises(ImproperlyConfiguredException):
        asgi_response = ASGIFileResponse(file_path=path, filename="404.txt", file_system=BaseLocalFileSystem())
        await asgi_response.start_response(empty_send)


class MockFileSystem(BaseFileSystem):
    async def info(self, path: PathType, **kwargs: Any) -> FileInfo:
        return FileInfo(
            is_symlink=False,
            mtime=0.0,
            name=str(path),
            size=len(str(path).encode()),
            type="file",
        )

    async def read_bytes(
        self,
        path: PathType,
        start: int | None = None,
        end: int | None = None,
    ) -> bytes:
        return str(path).encode()

    async def iter(self, path: PathType, chunksize: int, start: int = 0, end: int = -1) -> AsyncGenerator[bytes, None]:
        yield await self.read_bytes(path, start=start, end=end)


def test_file_response_file_system_lookup() -> None:
    @get("/")
    def handler() -> File:
        return File(path="Hello, world!", file_system="custom")

    with create_test_client(handler, plugins=[FileSystemRegistry({"custom": MockFileSystem()})]) as client:
        response = client.get("/")
        assert response.status_code == HTTP_200_OK
        assert response.content == b"Hello, world!"
        assert response.headers.get_list("content-length") == ["13"]


def test_file_response_default_file_system() -> None:
    @get("/")
    def handler() -> File:
        return File(path="Hello, world!")

    with create_test_client(handler, plugins=[FileSystemRegistry(default=MockFileSystem())]) as client:
        response = client.get("/")
        assert response.status_code == HTTP_200_OK
        assert response.content == b"Hello, world!"


def test_file_response_explicit_file_system() -> None:
    @get("/")
    def handler() -> File:
        return File(path="Hello, world!", file_system=MockFileSystem())

    with create_test_client(handler) as client:
        response = client.get("/")
        assert response.status_code == HTTP_200_OK
        assert response.content == b"Hello, world!"


def test_file_response_sync_file_system(tmp_path: pathlib.Path) -> None:
    fs = LocalFileSystem()
    path = tmp_path / "test.txt"
    content = secrets.token_hex()
    path.write_text(content)

    @get("/")
    def handler() -> File:
        return File(path=path, file_system=fs)

    with create_test_client(handler) as client:
        response = client.get("/")
        assert response.status_code == HTTP_200_OK
        assert response.content == content.encode()


@pytest.fixture()
def file(tmpdir: Path) -> Path:
    path = tmpdir / "file.txt"
    content = b"a"
    Path(path).write_bytes(content)
    return path


@pytest.mark.parametrize(
    "header_name",
    [
        "content-length",
        "Content-Length",
        "contenT-leNgTh",  # codespell:ignore
    ],
)
def test_does_not_override_existing_content_length_header(header_name: str, file: Path) -> None:
    @get("/")
    def handler() -> File:
        return File(path=file, headers={header_name: "2"})

    with create_test_client(handler) as client:
        response = client.get("/")
        assert response.status_code == HTTP_200_OK
        assert response.headers.get_list("content-length") == ["2"]


@pytest.mark.parametrize("header_name", ["last-modified", "Last-Modified", "LasT-modiFieD"])
def test_does_not_override_existing_last_modified_header(header_name: str, tmpdir: Path) -> None:
    path = Path(tmpdir / "file.txt")
    path.write_bytes(b"")

    @get("/")
    def handler() -> File:
        return File(path=path, headers={header_name: "foo"})

    with create_test_client(handler) as client:
        response = client.get("/")
        assert response.status_code == HTTP_200_OK
        assert response.headers.get_list("last-modified") == ["foo"]


@pytest.fixture()
def range_file(tmpdir: Path) -> tuple[Path, bytes]:
    content = bytes(range(256)) * 4  # 1024 bytes
    path = Path(tmpdir / "range.bin")
    path.write_bytes(content)
    return path, content


# ---------------------------------------------------------------------------
# conditional responses
# ---------------------------------------------------------------------------


def test_if_none_match_returns_not_modified(range_file: tuple[Path, bytes]) -> None:
    path, content = range_file

    @get("/", sync_to_thread=False)
    def handler() -> File:
        return File(path=path, filename="range.bin")

    with create_test_client(handler) as client:
        etag = client.get("/").headers["etag"]
        response = client.get("/", headers={"if-none-match": etag})
        assert response.status_code == HTTP_304_NOT_MODIFIED
        assert response.content == b""
        # validation headers are retained
        assert response.headers["etag"] == etag
        assert "last-modified" in response.headers
        # a 304 does not carry the representation
        assert "content-length" not in response.headers
        assert "content-type" not in response.headers
        assert response.headers["accept-ranges"] == "bytes"
        # a non-matching etag results in the full content
        response = client.get("/", headers={"if-none-match": '"something-else"'})
        assert response.status_code == HTTP_200_OK
        assert response.content == content


def test_if_none_match_supports_weak_tags_wildcard_and_lists(range_file: tuple[Path, bytes]) -> None:
    path, _ = range_file

    @get("/", sync_to_thread=False)
    def handler() -> File:
        return File(path=path, filename="range.bin")

    with create_test_client(handler) as client:
        etag = client.get("/").headers["etag"]
        assert client.get("/", headers={"if-none-match": f"W/{etag}"}).status_code == HTTP_304_NOT_MODIFIED
        assert client.get("/", headers={"if-none-match": "*"}).status_code == HTTP_304_NOT_MODIFIED
        assert client.get("/", headers={"if-none-match": f'"a", {etag}, "b"'}).status_code == HTTP_304_NOT_MODIFIED
        assert client.get("/", headers={"if-none-match": '"a", "b"'}).status_code == HTTP_200_OK


def test_if_none_match_with_custom_etag(tmpdir: Path) -> None:
    path = Path(tmpdir / "file.txt")
    path.write_bytes(b"content")

    @get("/", sync_to_thread=False)
    def handler() -> File:
        return File(path=path, etag=ETag(value="special"))

    with create_test_client(handler) as client:
        assert client.get("/", headers={"if-none-match": '"special"'}).status_code == HTTP_304_NOT_MODIFIED
        assert client.get("/", headers={"if-none-match": '"other"'}).status_code == HTTP_200_OK


def test_if_modified_since(range_file: tuple[Path, bytes]) -> None:
    path, content = range_file
    mtime = path.stat().st_mtime

    @get("/", sync_to_thread=False)
    def handler() -> File:
        return File(path=path, filename="range.bin")

    with create_test_client(handler) as client:
        last_modified = client.get("/").headers["last-modified"]
        # echoing Last-Modified results in 304
        response = client.get("/", headers={"if-modified-since": last_modified})
        assert response.status_code == HTTP_304_NOT_MODIFIED
        assert response.content == b""
        assert response.headers["last-modified"] == last_modified
        # an earlier date results in the full representation
        response = client.get("/", headers={"if-modified-since": formatdate(mtime - 100, usegmt=True)})
        assert response.status_code == HTTP_200_OK
        assert response.content == content
        # a future date results in 304
        assert (
            client.get("/", headers={"if-modified-since": formatdate(mtime + 100, usegmt=True)}).status_code
            == HTTP_304_NOT_MODIFIED
        )


def test_if_none_match_takes_precedence_over_if_modified_since(range_file: tuple[Path, bytes]) -> None:
    path, content = range_file

    @get("/", sync_to_thread=False)
    def handler() -> File:
        return File(path=path, filename="range.bin")

    with create_test_client(handler) as client:
        last_modified = client.get("/").headers["last-modified"]
        response = client.get(
            "/",
            headers={"if-none-match": '"does-not-match"', "if-modified-since": last_modified},
        )
        assert response.status_code == HTTP_200_OK
        assert response.content == content


def test_if_match_precondition_failed(range_file: tuple[Path, bytes]) -> None:
    path, _ = range_file

    @get("/", sync_to_thread=False)
    def handler() -> File:
        return File(path=path, filename="range.bin")

    with create_test_client(handler) as client:
        response = client.get("/", headers={"if-match": '"does-not-match"'})
        assert response.status_code == HTTP_412_PRECONDITION_FAILED
        assert response.content == b""
        assert response.headers["etag"]
        assert client.get("/", headers={"if-match": "*"}).status_code == HTTP_200_OK


def test_if_unmodified_since_precondition_failed(range_file: tuple[Path, bytes]) -> None:
    path, content = range_file
    mtime = path.stat().st_mtime

    @get("/", sync_to_thread=False)
    def handler() -> File:
        return File(path=path, filename="range.bin")

    with create_test_client(handler) as client:
        response = client.get("/", headers={"if-unmodified-since": formatdate(mtime - 100, usegmt=True)})
        assert response.status_code == HTTP_412_PRECONDITION_FAILED
        assert response.content == b""
        response = client.get("/", headers={"if-unmodified-since": formatdate(mtime + 100, usegmt=True)})
        assert response.status_code == HTTP_200_OK
        assert response.content == content


def test_conditional_headers_ignored_without_mtime(tmpdir: Path, mocker: MockerFixture) -> None:
    path = Path(tmpdir / "file.txt")
    path.write_bytes(b"content")
    fs = LocalFileSystem()
    mocker.patch.object(
        fs,
        "info",
        return_value={"name": "file.txt", "size": 7, "type": "file"},
    )

    @get("/", sync_to_thread=False)
    def handler() -> File:
        return File(path=path, filename="file.txt", file_system=fs)

    with create_test_client(handler) as client:
        response = client.get("/", headers={"if-modified-since": "Wed, 21 Oct 2015 07:28:00 GMT"})
        assert response.status_code == HTTP_200_OK
        assert response.content == b"content"


# ---------------------------------------------------------------------------
# range responses
# ---------------------------------------------------------------------------


def test_single_range_request(range_file: tuple[Path, bytes]) -> None:
    path, content = range_file

    @get("/", sync_to_thread=False)
    def handler() -> File:
        return File(path=path, filename="range.bin")

    with create_test_client(handler) as client:
        response = client.get("/", headers={"range": "bytes=10-19"})
        assert response.status_code == HTTP_206_PARTIAL_CONTENT
        assert response.content == content[10:20]
        assert response.headers["content-range"] == "bytes 10-19/1024"
        assert response.headers["content-length"] == "10"
        assert response.headers["accept-ranges"] == "bytes"
        assert response.headers["etag"]
        assert response.headers["last-modified"]


@pytest.mark.parametrize(
    ("range_header", "expected_slice", "expected_content_range"),
    [
        ("bytes=1000-", slice(1000, None), "bytes 1000-1023/1024"),
        ("bytes=-24", slice(-24, None), "bytes 1000-1023/1024"),
        ("bytes=0-0", slice(0, 1), "bytes 0-0/1024"),
        ("bytes=1000-9999", slice(1000, None), "bytes 1000-1023/1024"),  # end is clipped to size
    ],
)
def test_range_forms(
    range_file: tuple[Path, bytes],
    range_header: str,
    expected_slice: slice,
    expected_content_range: str,
) -> None:
    path, content = range_file

    @get("/", sync_to_thread=False)
    def handler() -> File:
        return File(path=path, filename="range.bin")

    with create_test_client(handler) as client:
        response = client.get("/", headers={"range": range_header})
        assert response.status_code == HTTP_206_PARTIAL_CONTENT
        assert response.content == content[expected_slice]
        assert response.headers["content-range"] == expected_content_range
        assert response.headers["content-length"] == str(len(response.content))


def test_head_range_request(range_file: tuple[Path, bytes]) -> None:
    path, _ = range_file

    @get("/", sync_to_thread=False)
    def get_handler() -> File:
        return File(path=path, filename="range.bin")

    @head("/", sync_to_thread=False)
    def head_handler() -> File:
        return File(path=path, filename="range.bin")

    with create_test_client([get_handler, head_handler]) as client:
        response = client.head("/", headers={"range": "bytes=0-99"})
        assert response.status_code == HTTP_206_PARTIAL_CONTENT
        assert response.content == b""
        assert response.headers["content-length"] == "100"
        assert response.headers["content-range"] == "bytes 0-99/1024"


def test_range_out_of_bounds(range_file: tuple[Path, bytes]) -> None:
    path, _ = range_file

    @get("/", sync_to_thread=False)
    def handler() -> File:
        return File(path=path, filename="range.bin")

    with create_test_client(handler) as client:
        response = client.get("/", headers={"range": "bytes=2000-3000"})
        assert response.status_code == HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE
        assert response.content == b""
        assert response.headers["content-range"] == "bytes */1024"
        assert response.headers["accept-ranges"] == "bytes"


def test_range_on_empty_file(tmpdir: Path) -> None:
    path = Path(tmpdir / "empty.bin")
    path.write_bytes(b"")

    @get("/", sync_to_thread=False)
    def handler() -> File:
        return File(path=path, filename="empty.bin")

    with create_test_client(handler) as client:
        assert client.get("/", headers={"range": "bytes=0-0"}).status_code == HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE
        assert client.get("/", headers={"range": "bytes=-1"}).status_code == HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE


@pytest.mark.parametrize(
    "range_header",
    [
        "bytes=abc",
        "bytes=10",
        "bytes=",
        "bytes=--",
        "bytes=0-10-20",
        "bytes=0-foo",
    ],
)
def test_malformed_range_request(range_file: tuple[Path, bytes], range_header: str) -> None:
    path, _ = range_file

    @get("/", sync_to_thread=False)
    def handler() -> File:
        return File(path=path, filename="range.bin")

    with create_test_client(handler) as client:
        response = client.get("/", headers={"range": range_header})
        assert response.status_code == HTTP_400_BAD_REQUEST
        assert response.content == b""


def test_inverted_range_is_unsatisfiable(range_file: tuple[Path, bytes]) -> None:
    path, _ = range_file

    @get("/", sync_to_thread=False)
    def handler() -> File:
        return File(path=path, filename="range.bin")

    with create_test_client(handler) as client:
        response = client.get("/", headers={"range": "bytes=100-10"})
        assert response.status_code == HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE


def test_unsupported_range_unit_is_ignored(range_file: tuple[Path, bytes]) -> None:
    path, content = range_file

    @get("/", sync_to_thread=False)
    def handler() -> File:
        return File(path=path, filename="range.bin")

    with create_test_client(handler) as client:
        response = client.get("/", headers={"range": "items=0-10"})
        assert response.status_code == HTTP_200_OK
        assert response.content == content


def test_multiple_ranges_return_multipart(range_file: tuple[Path, bytes]) -> None:
    path, content = range_file

    @get("/", sync_to_thread=False)
    def handler() -> File:
        return File(path=path, filename="range.bin")

    with create_test_client(handler) as client:
        response = client.get("/", headers={"range": "bytes=0-9, 100-109"})
        assert response.status_code == HTTP_206_PARTIAL_CONTENT
        content_type = response.headers["content-type"]
        assert content_type.startswith("multipart/byteranges; boundary=")
        boundary = content_type.split("boundary=", 1)[1]
        body = response.content
        assert body.startswith(f"--{boundary}\r\n".encode())
        assert body.endswith(f"--{boundary}--\r\n".encode())
        assert content[0:10] in body and content[100:110] in body
        assert b"Content-Range: bytes 0-9/1024" in body
        assert b"Content-Range: bytes 100-109/1024" in body
        assert response.headers["content-length"] == str(len(body))


def test_duplicate_overlapping_and_adjacent_ranges_are_coalesced(range_file: tuple[Path, bytes]) -> None:
    path, content = range_file

    @get("/", sync_to_thread=False)
    def handler() -> File:
        return File(path=path, filename="range.bin")

    with create_test_client(handler) as client:
        # exact duplicates and overlaps coalesce into a single part -> single 206
        response = client.get("/", headers={"range": "bytes=0-9,0-9,5-20"})
        assert response.status_code == HTTP_206_PARTIAL_CONTENT
        assert response.headers["content-range"] == "bytes 0-20/1024"
        assert response.content == content[0:21]
        # adjacent ranges are coalesced as well
        response = client.get("/", headers={"range": "bytes=0-9,10-19"})
        assert response.status_code == HTTP_206_PARTIAL_CONTENT
        assert response.headers["content-range"] == "bytes 0-19/1024"
        assert response.content == content[0:20]
        # contained ranges
        response = client.get("/", headers={"range": "bytes=0-100,10-20"})
        assert response.headers["content-range"] == "bytes 0-100/1024"
        assert response.content == content[0:101]


def test_streamed_range_across_chunks(tmpdir: Path) -> None:
    content = urandom(5000)
    path = Path(tmpdir / "large.bin")
    path.write_bytes(content)

    @get("/", sync_to_thread=False)
    def handler() -> File:
        return File(path=path, chunk_size=64)

    with create_test_client(handler) as client:
        response = client.get("/", headers={"range": "bytes=100-4999"})
        assert response.status_code == HTTP_206_PARTIAL_CONTENT
        assert response.content == content[100:5000]
        assert response.headers["content-range"] == "bytes 100-4999/5000"


def test_if_range_etag(range_file: tuple[Path, bytes]) -> None:
    path, content = range_file

    @get("/", sync_to_thread=False)
    def handler() -> File:
        return File(path=path, filename="range.bin")

    with create_test_client(handler) as client:
        etag = client.get("/").headers["etag"]
        response = client.get("/", headers={"range": "bytes=0-9", "if-range": etag})
        assert response.status_code == HTTP_206_PARTIAL_CONTENT
        assert response.content == content[0:10]
        # stale If-Range -> the full representation is sent
        response = client.get("/", headers={"range": "bytes=0-9", "if-range": '"stale"'})
        assert response.status_code == HTTP_200_OK
        assert response.content == content


def test_if_range_date(range_file: tuple[Path, bytes]) -> None:
    path, content = range_file
    mtime = path.stat().st_mtime

    @get("/", sync_to_thread=False)
    def handler() -> File:
        return File(path=path, filename="range.bin")

    with create_test_client(handler) as client:
        response = client.get(
            "/",
            headers={"range": "bytes=0-9", "if-range": formatdate(mtime, usegmt=True)},
        )
        assert response.status_code == HTTP_206_PARTIAL_CONTENT
        assert response.content == content[0:10]
        response = client.get(
            "/",
            headers={"range": "bytes=0-9", "if-range": formatdate(mtime - 100, usegmt=True)},
        )
        assert response.status_code == HTTP_200_OK
        assert response.content == content


def test_plain_response_headers_unchanged(range_file: tuple[Path, bytes]) -> None:
    path, content = range_file

    @get("/", sync_to_thread=False)
    def handler() -> File:
        return File(path=path, filename="range.bin")

    with create_test_client(handler) as client:
        response = client.get("/")
        assert response.status_code == HTTP_200_OK
        assert response.content == content
        # no negotiation headers are added when the capabilities are not used
        assert "accept-ranges" not in response.headers
        assert "content-range" not in response.headers
        assert response.headers["content-length"] == "1024"
        assert "etag" in response.headers and "last-modified" in response.headers


# ---------------------------------------------------------------------------
# pure range parsing
# ---------------------------------------------------------------------------


def test_parse_byte_ranges_normalization() -> None:
    assert parse_byte_ranges("bytes=0-9", 100) == [_ByteRange(0, 9)]
    # sorted, overlaps coalesced; the gap between 0-50 and 100-200 keeps them apart
    assert parse_byte_ranges("bytes=100-200,0-50,150-160", 1000) == [
        _ByteRange(0, 50),
        _ByteRange(100, 200),
    ]
    # adjacent ranges coalesce into one
    assert parse_byte_ranges("bytes=100-200,0-99", 1000) == [_ByteRange(0, 200)]
    # disjoint ranges stay separate and are returned in order
    assert parse_byte_ranges("bytes=500-599,0-99", 1000) == [
        _ByteRange(0, 99),
        _ByteRange(500, 599),
    ]
    # suffix ranges
    assert parse_byte_ranges("bytes=-100", 1000) == [_ByteRange(900, 999)]
    # open-ended ranges
    assert parse_byte_ranges("bytes=900-", 1000) == [_ByteRange(900, 999)]
    # nothing satisfiable -> empty list (416)
    assert parse_byte_ranges("bytes=1000-2000", 1000) == []
    assert parse_byte_ranges("bytes=-0", 1000) == []
    # unsupported unit -> None (ignore the Range header)
    assert parse_byte_ranges("items=0-10", 1000) is None
    # malformed syntax raises
    with pytest.raises(MalformedRangeHeaderError):
        parse_byte_ranges("bytes=abc", 1000)
    with pytest.raises(MalformedRangeHeaderError):
        parse_byte_ranges("bytes=", 1000)


# ---------------------------------------------------------------------------
# snapshot consistency
# ---------------------------------------------------------------------------


class _ChangingFileSystem(BaseFileSystem):
    def __init__(self, *, snapshot_size: int, actual_size: int, mtime: float = 1.0) -> None:
        self.snapshot_size = snapshot_size
        self.actual_size = actual_size
        self.mtime = mtime

    async def info(self, path: PathType, **kwargs: Any) -> FileInfo:
        return FileInfo(name=str(path), size=self.snapshot_size, type="file", mtime=self.mtime)

    async def read_bytes(self, path: PathType, start: int | None = None, end: int | None = None) -> bytes:
        return b"x" * max(0, min(self.actual_size, (end if end not in (None, -1) else self.actual_size)) - start)

    async def iter(self, path: PathType, chunksize: int, start: int = 0, end: int = -1) -> AsyncGenerator[bytes, None]:
        limit = self.actual_size - start
        for index in range(0, limit, chunksize):
            yield b"x" * min(chunksize, limit - index)


async def test_truncated_file_aborts_before_body() -> None:
    events: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        events.append(message)

    async def receive() -> Any:
        await asyncio.Event().wait()
        return {"type": "http.disconnect"}

    response = ASGIFileResponse(
        file_path="x",
        file_system=_ChangingFileSystem(snapshot_size=100, actual_size=60),
        chunk_size=1024,
    )
    with pytest.raises(Exception, match="changed during the transfer"):
        await response({"type": "http", "method": "GET", "headers": []}, receive, send)

    assert events[0]["type"] == "http.response.start"
    assert all(event["type"] != "http.response.body" for event in events)


def test_range_uses_file_system_offsets(tmpdir: Path) -> None:
    seen: list[tuple[str, int, int]] = []

    class RecordingFileSystem(BaseFileSystem):
        async def info(self, path: PathType, **kwargs: Any) -> FileInfo:
            return FileInfo(name=str(path), size=100, type="file", mtime=1.0)

        async def read_bytes(self, path: PathType, start: int = 0, end: int = -1) -> bytes:
            seen.append(("read", start, end))
            return b"y" * (end - start)

        async def iter(
            self, path: PathType, chunksize: int, start: int = 0, end: int = -1
        ) -> AsyncGenerator[bytes, None]:
            seen.append(("iter", start, end))
            yield b"y" * (end - start)

    @get("/", sync_to_thread=False)
    def handler() -> File:
        return File(path="x", file_system=RecordingFileSystem())

    with create_test_client(handler) as client:
        response = client.get("/", headers={"range": "bytes=10-19"})
        assert response.status_code == HTTP_206_PARTIAL_CONTENT
        assert response.content == b"y" * 10
        assert seen == [("read", 10, 20)]
