from __future__ import annotations

import asyncio
import json
import logging

import pytest

from litestar import Litestar, get
from litestar.config.drain import DrainConfig
from litestar.enums import DrainState
from tests.unit.test_drain.conftest import HTTPExchange, WebSocketExchange, wait_until


def create_app(**drain_kwargs: object) -> tuple[Litestar, asyncio.Event, asyncio.Event]:
    handler_started = asyncio.Event()
    allow_completion = asyncio.Event()

    @get("/")
    async def index() -> dict[str, str]:
        return {"hello": "world"}

    @get("/slow")
    async def slow() -> dict[str, str]:
        handler_started.set()
        await allow_completion.wait()
        return {"ok": "1"}

    @get("/health")
    async def health() -> dict[str, object]:
        return {
            "state": app.drain_state.value,
            "in_flight": app.in_flight_request_count,
        }

    @get("/admin/drain")
    async def admin_drain() -> dict[str, object]:
        result = await app.begin_drain()
        return {"timed_out": result.timed_out, "aborted": len(result.unfinished)}

    app = Litestar(
        route_handlers=[index, slow, health, admin_drain],
        drain_config=DrainConfig(**drain_kwargs),  # type: ignore[arg-type]
    )
    return app, handler_started, allow_completion


async def test_request_processed_normally_when_running() -> None:
    app, _, _ = create_app()

    exchange = HTTPExchange(app)
    exchange.start()
    await exchange.wait()

    assert exchange.status == 200
    assert json.loads(exchange.body) == {"hello": "world"}
    assert app.drain_state is DrainState.RUNNING


async def test_new_request_rejected_while_draining_with_retry_hint() -> None:
    app, started, allow_completion = create_app(grace_period=10)

    slow = HTTPExchange(app, path="/slow")
    slow.start()
    await started.wait()

    drain_task = asyncio.create_task(app.begin_drain())
    await wait_until(lambda: app.drain_state is DrainState.DRAINING)

    rejected = HTTPExchange(app)
    rejected.start()
    await rejected.wait()

    assert rejected.status == 503
    assert rejected.headers["retry-after"] == "30"
    assert rejected.headers["connection"] == "close"
    assert rejected.headers["content-type"] == "application/json"
    assert json.loads(rejected.body) == {
        "status_code": 503,
        "detail": "Service is shutting down",
    }
    assert app.get_drain_status().rejected_requests == 1

    assert not drain_task.done()  # the in-flight request is still running
    allow_completion.set()
    result = await drain_task

    assert result.timed_out is False
    await slow.wait()
    assert slow.status == 200
    assert json.loads(slow.body) == {"ok": "1"}


async def test_probe_paths_remain_available_while_draining() -> None:
    app, started, allow_completion = create_app(grace_period=10, probe_paths=("/health",))

    slow = HTTPExchange(app, path="/slow")
    slow.start()
    await started.wait()

    drain_task = asyncio.create_task(app.begin_drain())
    await wait_until(lambda: app.drain_state is DrainState.DRAINING)

    probe = HTTPExchange(app, path="/health")
    probe.start()
    await probe.wait()
    assert probe.status == 200
    assert json.loads(probe.body) == {"state": "draining", "in_flight": 1}

    allow_completion.set()
    await drain_task
    await slow.wait()

    # after the drain completed, probe paths are rejected as well - shutdown hooks
    # are about to run and resources may already be released
    late_probe = HTTPExchange(app, path="/health")
    late_probe.start()
    await late_probe.wait()
    assert late_probe.status == 503


async def test_custom_rejection_response_without_retry_after() -> None:
    app, _, _ = create_app(
        rejection_status_code=503,
        rejection_detail="please retry elsewhere",
        retry_after=None,
        connection_close=False,
    )
    await app.begin_drain()

    exchange = HTTPExchange(app)
    exchange.start()
    await exchange.wait()

    assert exchange.status == 503
    assert "retry-after" not in exchange.headers
    assert "connection" not in exchange.headers
    assert json.loads(exchange.body) == {
        "status_code": 503,
        "detail": "please retry elsewhere",
    }


async def test_websocket_connections_are_denied_while_draining() -> None:
    app, _, _ = create_app()
    await app.begin_drain()

    exchange = WebSocketExchange(app, path="/ws")
    exchange.start()
    await exchange.wait()

    assert exchange.messages == [{"type": "websocket.close", "code": 1001}]


async def test_in_flight_count_is_reported() -> None:
    app, first_started, _ = create_app(grace_period=10)

    first = HTTPExchange(app, path="/slow")
    first.start()
    await first_started.wait()

    # a second in-flight request using its own synchronization
    second_started = asyncio.Event()
    second_allow = asyncio.Event()

    @get("/other-slow")
    async def other_slow() -> None:
        second_started.set()
        await second_allow.wait()

    app.register(other_slow)  # type: ignore[arg-type]
    second = HTTPExchange(app, path="/other-slow")
    second.start()
    await second_started.wait()

    status = app.get_drain_status()
    assert status.state is DrainState.RUNNING
    assert status.in_flight_requests == 2
    assert app.in_flight_request_count == 2

    drain_task = asyncio.create_task(app.begin_drain())
    await wait_until(lambda: app.drain_state is DrainState.DRAINING)
    assert app.in_flight_request_count == 2

    second_allow.set()
    await second.wait()
    await wait_until(lambda: app.in_flight_request_count == 1)

    assert app.get_drain_status().in_flight_requests == 1
    assert not drain_task.done()


async def test_deadline_aborts_in_flight_request_and_records_it(caplog: pytest.LogCaptureFixture) -> None:
    app, started, _ = create_app(grace_period=0.05, abort_timeout=1)

    slow = HTTPExchange(app, path="/slow")
    slow.start()
    await started.wait()

    with caplog.at_level(logging.WARNING, logger="litestar.drain"):
        result = await app.begin_drain()

    assert result.timed_out is True
    assert len(result.unfinished) == 1
    assert result.unfinished[0].method == "GET"
    assert result.unfinished[0].path == "/slow"
    assert app.drain.result is result
    assert slow.task is not None
    assert slow.task.cancelled()
    assert "aborting in-flight request GET /slow" in caplog.text


async def test_drain_triggered_from_request_is_idempotent_and_does_not_deadlock() -> None:
    app, started, allow_completion = create_app(grace_period=10)

    # state before the trigger is consistent
    assert app.drain_state is DrainState.RUNNING
    assert app.get_drain_status().result is None

    slow = HTTPExchange(app, path="/slow")
    slow.start()
    await started.wait()

    admin = HTTPExchange(app, path="/admin/drain")
    admin.start()
    await wait_until(lambda: app.drain_state is DrainState.DRAINING)

    # the administrative request itself is not counted as work to drain
    assert app.in_flight_request_count == 1

    allow_completion.set()
    await admin.wait()
    await slow.wait()

    assert admin.status == 200
    assert json.loads(admin.body) == {"timed_out": False, "aborted": 0}
    assert slow.status == 200
    assert app.drain_state is DrainState.DRAINED

    # repeated triggers are idempotent and observe the same completed state
    repeated = await app.begin_drain()
    assert repeated is app.drain.result


async def test_root_path_is_stripped_when_matching_probe_paths() -> None:
    app, started, allow_completion = create_app(grace_period=10, probe_paths=("/health",))

    slow = HTTPExchange(app, path="/slow")
    slow.start()
    await started.wait()

    drain_task = asyncio.create_task(app.begin_drain())
    await wait_until(lambda: app.drain_state is DrainState.DRAINING)

    exchange = HTTPExchange(app, path="/api/health", root_path="/api")
    exchange.start()
    await exchange.wait()
    assert exchange.status == 200

    allow_completion.set()
    await drain_task
    await slow.wait()
