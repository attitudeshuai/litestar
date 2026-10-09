from __future__ import annotations

import asyncio
import json

from litestar import Litestar, get
from litestar.config.drain import DrainConfig
from litestar.enums import DrainState
from tests.unit.test_drain.conftest import HTTPExchange, LifespanHarness, wait_until


async def test_lifespan_drains_before_running_shutdown_hooks() -> None:
    events: list[str] = []
    handler_started = asyncio.Event()
    allow_completion = asyncio.Event()

    @get("/slow")
    async def slow() -> dict[str, str]:
        handler_started.set()
        await allow_completion.wait()
        events.append("request-completed")
        return {"ok": "1"}

    async def on_shutdown() -> None:
        events.append("shutdown-hook")

    app = Litestar(
        [slow],
        on_shutdown=[on_shutdown],
        drain_config=DrainConfig(grace_period=10),
    )

    lifespan = LifespanHarness(app)
    await lifespan.start()

    slow_exchange = HTTPExchange(app, path="/slow")
    slow_exchange.start()
    await handler_started.wait()

    shutdown_task = asyncio.create_task(lifespan.shutdown())
    await wait_until(lambda: app.drain_state is DrainState.DRAINING)
    await asyncio.sleep(0)

    # the shutdown hook must not run while the in-flight request is still working
    assert events == []

    allow_completion.set()
    message = await shutdown_task
    await slow_exchange.wait()

    assert message["type"] == "lifespan.shutdown.complete"
    assert slow_exchange.status == 200
    assert events == ["request-completed", "shutdown-hook"]
    assert app.drain.result is not None
    assert app.drain.result.timed_out is False


async def test_lifespan_drain_timeout_is_determinative_and_observable() -> None:
    events: list[str] = []
    handler_started = asyncio.Event()

    @get("/slow")
    async def slow() -> None:
        handler_started.set()
        await asyncio.sleep(3600)

    async def on_shutdown() -> None:
        events.append("shutdown-hook")

    app = Litestar(
        [slow],
        on_shutdown=[on_shutdown],
        drain_config=DrainConfig(grace_period=0.05, abort_timeout=1),
    )

    lifespan = LifespanHarness(app)
    await lifespan.start()

    slow_exchange = HTTPExchange(app, path="/slow")
    slow_exchange.start()
    await handler_started.wait()

    message = await asyncio.wait_for(lifespan.shutdown(), timeout=2)

    assert message["type"] == "lifespan.shutdown.complete"
    assert events == ["shutdown-hook"]

    result = app.drain.result
    assert result is not None
    assert result.timed_out is True
    assert [request.path for request in result.unfinished] == ["/slow"]
    assert slow_exchange.task is not None
    assert slow_exchange.task.cancelled()
    assert app.drain_state is DrainState.DRAINED


async def test_disabled_drain_leaves_lifespan_behaviour_unchanged() -> None:
    events: list[str] = []

    async def on_startup() -> None:
        events.append("startup-hook")

    async def on_shutdown() -> None:
        events.append("shutdown-hook")

    @get("/")
    async def index() -> dict[str, str]:
        return {"hello": "world"}

    app = Litestar([index], on_startup=[on_startup], on_shutdown=[on_shutdown])

    assert app.drain.enabled is False
    assert app.drain_state is DrainState.RUNNING

    lifespan = LifespanHarness(app)
    await lifespan.start()
    assert events == ["startup-hook"]

    exchange = HTTPExchange(app)
    exchange.start()
    await exchange.wait()
    assert exchange.status == 200
    assert json.loads(exchange.body) == {"hello": "world"}

    message = await lifespan.shutdown()
    assert message["type"] == "lifespan.shutdown.complete"
    assert events == ["startup-hook", "shutdown-hook"]
    assert app.drain.result is None


async def test_signal_or_explicit_drain_is_joined_by_lifespan_shutdown() -> None:
    """A drain that already completed before ``lifespan.shutdown`` must not be run
    twice and the shutdown hooks still run afterwards."""
    events: list[str] = []

    async def on_shutdown() -> None:
        events.append("shutdown-hook")

    app = Litestar(
        [],
        on_shutdown=[on_shutdown],
        drain_config=DrainConfig(grace_period=10),
    )

    lifespan = LifespanHarness(app)
    await lifespan.start()

    result = await app.begin_drain()
    assert app.drain_state is DrainState.DRAINED
    assert events == []

    message = await lifespan.shutdown()
    assert message["type"] == "lifespan.shutdown.complete"
    assert events == ["shutdown-hook"]
    assert app.drain.result is result
