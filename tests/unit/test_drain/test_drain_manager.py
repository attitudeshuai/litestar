from __future__ import annotations

import asyncio
import contextlib

import pytest

from litestar.config.drain import DrainConfig
from litestar.drain import DrainManager, DrainResult
from litestar.enums import DrainState
from tests.unit.test_drain.conftest import create_http_scope


def test_disabled_manager_is_inert() -> None:
    manager = DrainManager(None)

    assert manager.enabled is False
    assert manager.state is DrainState.RUNNING
    assert manager.in_flight_request_count == 0
    assert manager.result is None


async def test_begin_drain_raises_when_disabled() -> None:
    manager = DrainManager(None)

    with pytest.raises(RuntimeError, match="not enabled"):
        await manager.begin_drain()


async def test_drain_without_in_flight_completes_immediately() -> None:
    manager = DrainManager(DrainConfig(grace_period=10))

    status_before = manager.get_status()
    assert status_before.state is DrainState.RUNNING
    assert status_before.drain_elapsed is None
    assert status_before.result is None

    result = await manager.begin_drain()

    assert isinstance(result, DrainResult)
    assert result.state is DrainState.DRAINED
    assert result.timed_out is False
    assert result.unfinished == ()
    assert result.in_flight_requests == 0
    assert manager.state is DrainState.DRAINED
    assert manager.get_status().result is result
    assert manager.get_status().drain_elapsed is not None


async def test_begin_drain_is_idempotent_under_concurrent_calls() -> None:
    manager = DrainManager(DrainConfig(grace_period=10))

    hold = asyncio.Event()

    async def tracked_request() -> None:
        record = await manager.admit(create_http_scope(path="/slow"))
        try:
            await hold.wait()
        finally:
            await manager.release(record)

    request_task = asyncio.create_task(tracked_request())
    await wait_admitted(manager)

    drain_tasks = [asyncio.create_task(manager.begin_drain()) for _ in range(3)]
    await asyncio.sleep(0)
    assert manager.state is DrainState.DRAINING
    assert all(not task.done() for task in drain_tasks)

    hold.set()
    results = await asyncio.gather(*drain_tasks)

    assert results[0] is results[1] is results[2]
    assert results[0].timed_out is False
    # after completion, repeated calls return the same stored result
    assert await manager.begin_drain() is results[0]
    await request_task


async def test_admit_decisions_follow_state_and_probe_paths() -> None:
    manager = DrainManager(DrainConfig(grace_period=10, probe_paths=("/health",)))

    business_scope = create_http_scope(path="/orders")
    probe_scope = create_http_scope(path="/health")

    business = await manager.admit(business_scope)
    assert business is not None
    assert manager.in_flight_request_count == 1

    drain_task = asyncio.create_task(manager.begin_drain())
    await asyncio.sleep(0)
    assert manager.state is DrainState.DRAINING

    # new business requests are rejected, probe requests are admitted
    assert await manager.admit(business_scope) is None
    probe = await manager.admit(probe_scope)
    assert probe is not None
    assert manager.rejected_requests == 1
    # probes are observable...
    assert manager.in_flight_request_count == 1
    # ...but the drain still waits for them: releasing the business request while the
    # probe is in flight must not complete the drain
    await manager.release(business)
    assert not drain_task.done()

    await manager.release(probe)
    result = await drain_task
    assert result.timed_out is False

    # once drained, even probe paths are no longer admitted
    assert manager.state is DrainState.DRAINED
    assert await manager.admit(probe_scope) is None
    assert await manager.admit(business_scope) is None
    assert manager.rejected_requests == 3


async def test_deadline_aborts_stragglers_and_records_them() -> None:
    manager = DrainManager(DrainConfig(grace_period=0.05, abort_timeout=1))

    async def tracked_request() -> None:
        record = await manager.admit(create_http_scope(path="/slow"))
        try:
            await asyncio.sleep(3600)
        finally:
            await manager.release(record)

    request_task = asyncio.create_task(tracked_request())
    await wait_admitted(manager)

    result = await manager.begin_drain()

    assert result.timed_out is True
    assert len(result.unfinished) == 1
    unfinished = result.unfinished[0]
    assert unfinished.method == "GET"
    assert unfinished.path == "/slow"
    assert unfinished.elapsed >= 0.05
    with pytest.raises(asyncio.CancelledError):
        await request_task


async def test_drain_proceeds_when_cancelled_request_ignores_cancellation() -> None:
    manager = DrainManager(DrainConfig(grace_period=0.01, abort_timeout=0.05))

    async def stubborn_request() -> None:
        record = await manager.admit(create_http_scope(path="/stubborn"))
        cancellations = 0
        try:
            while True:
                try:
                    await asyncio.sleep(3600)
                except asyncio.CancelledError:
                    # badly behaved handler swallows the first cancellation, but lets
                    # a subsequent one terminate it so the test loop can close
                    cancellations += 1
                    if cancellations >= 2:
                        raise
        finally:
            await manager.release(record)

    request_task = asyncio.create_task(stubborn_request())
    await wait_admitted(manager)

    result = await asyncio.wait_for(manager.begin_drain(), timeout=2)

    # the shutdown sequence stays deterministic even though the task is stuck
    assert result.timed_out is True
    assert len(result.unfinished) == 1
    assert result.in_flight_requests == 1

    request_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await request_task


async def test_trigger_request_is_exempt_from_drain() -> None:
    manager = DrainManager(DrainConfig(grace_period=10))
    triggered = asyncio.Event()

    async def administrative_trigger() -> DrainResult:
        # simulate being admitted like the gate would do, then trigger the drain
        # from within that very request task
        record = await manager.admit(create_http_scope(path="/admin/drain"))
        triggered.set()
        try:
            return await manager.begin_drain()
        finally:
            await manager.release(record)

    trigger_task = asyncio.create_task(administrative_trigger())
    await triggered.wait()
    # the drain must not wait for (or deadlock on) its own trigger
    result = await trigger_task
    assert result.state is DrainState.DRAINED
    assert manager.in_flight_request_count == 0


async def wait_admitted(manager: DrainManager) -> None:
    while manager.in_flight_request_count == 0:
        await asyncio.sleep(0)
