from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from litestar import Litestar


def create_http_scope(
    method: str = "GET",
    path: str = "/",
    *,
    headers: list[tuple[bytes, bytes]] | None = None,
    root_path: str = "",
    query_string: bytes = b"",
) -> dict[str, Any]:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": query_string,
        "root_path": root_path,
        "headers": headers or [],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
        "extensions": {},
    }


def create_websocket_scope(path: str = "/ws") -> dict[str, Any]:
    return {
        "type": "websocket",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "scheme": "ws",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
        "extensions": {},
        "subprotocols": [],
    }


class HTTPExchange:
    """Drive a single HTTP request against an ASGI app in the current event loop."""

    def __init__(self, app: Litestar, **scope_kwargs: Any) -> None:
        self.app = app
        self.scope = create_http_scope(**scope_kwargs)
        self.messages: list[dict[str, Any]] = []
        self.task: asyncio.Task[None] | None = None
        self._request_delivered = False

    async def receive(self) -> dict[str, Any]:
        if not self._request_delivered:
            self._request_delivered = True
            return {"type": "http.request", "body": b"", "more_body": False}
        return {"type": "http.disconnect"}

    async def send(self, message: dict[str, Any]) -> None:
        self.messages.append(message)

    def start(self) -> asyncio.Task[None]:
        self.task = asyncio.create_task(
            self.app(
                self.scope,
                self.receive,
                self.send,
            )
        )
        return self.task

    async def wait(self) -> None:
        assert self.task is not None
        await self.task

    @property
    def response_start(self) -> dict[str, Any]:
        return next(message for message in self.messages if message["type"] == "http.response.start")

    @property
    def status(self) -> int:
        return self.response_start["status"]

    @property
    def headers(self) -> dict[str, str]:
        return {key.decode(): value.decode() for key, value in self.response_start["headers"]}

    @property
    def body(self) -> bytes:
        return b"".join(
            message.get("body", b"")
            for message in self.messages
            if message["type"] == "http.response.body"
        )


class WebSocketExchange:
    def __init__(self, app: Litestar, path: str = "/ws") -> None:
        self.app = app
        self.scope = create_websocket_scope(path)
        self.messages: list[dict[str, Any]] = []
        self.task: asyncio.Task[None] | None = None

    async def receive(self) -> dict[str, Any]:
        return await asyncio.get_running_loop().create_future()

    async def send(self, message: dict[str, Any]) -> None:
        self.messages.append(message)

    def start(self) -> asyncio.Task[None]:
        self.task = asyncio.create_task(self.app(self.scope, self.receive, self.send))
        return self.task

    async def wait(self) -> None:
        assert self.task is not None
        await self.task


class LifespanHarness:
    """Manually drive the ASGI lifespan protocol on a single event loop."""

    def __init__(self, app: Litestar) -> None:
        self.app = app
        self._inbox: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._outbox: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.task: asyncio.Task[None] | None = None

    async def _receive(self) -> dict[str, Any]:
        return await self._inbox.get()

    async def _send(self, message: dict[str, Any]) -> None:
        await self._outbox.put(message)

    async def start(self) -> None:
        self.task = asyncio.create_task(
            self.app(
                {"type": "lifespan"},
                self._receive,
                self._send,
            )
        )
        await self._inbox.put({"type": "lifespan.startup"})
        message = await self._outbox.get()
        assert message["type"] == "lifespan.startup.complete"

    async def shutdown(self) -> dict[str, Any]:
        assert self.task is not None
        await self._inbox.put({"type": "lifespan.shutdown"})
        message = await self._outbox.get()
        await self.task
        return message


async def wait_until(predicate: Any, timeout: float = 1.0) -> None:
    """Poll ``predicate`` until it is true or ``timeout`` elapses."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition not met within timeout")
