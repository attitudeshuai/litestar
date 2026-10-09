from datetime import timedelta
from typing import Any

import pytest
from anyio import Event, create_task_group

from litestar.di import DependencyCache, DependencyCacheScope, Provide
from litestar.exceptions import ImproperlyConfiguredException
from litestar.types import Empty


async def async_value() -> int:
    return 1


def test_provide_scope_defaults() -> None:
    provide = Provide(async_value)
    assert provide.use_cache is False
    assert provide.cache_scope is None
    assert provide.cache_ttl is None
    assert provide.value is Empty

    provider_scoped = Provide(async_value, use_cache=True)
    assert provider_scoped.use_cache is True
    assert provider_scoped.cache_scope is DependencyCacheScope.PROVIDER

    app_scoped = Provide(async_value, use_cache=DependencyCacheScope.APP)
    assert app_scoped.use_cache is True
    assert app_scoped.cache_scope is DependencyCacheScope.APP

    request_scoped = Provide(async_value, use_cache=DependencyCacheScope.REQUEST)
    assert request_scoped.use_cache is True
    assert request_scoped.cache_scope is DependencyCacheScope.REQUEST


def test_provide_ttl_normalization() -> None:
    provide = Provide(async_value, use_cache=True, cache_ttl=5)
    assert provide.cache_ttl == 5.0

    provide = Provide(async_value, use_cache=DependencyCacheScope.APP, cache_ttl=timedelta(seconds=2))
    assert provide.cache_ttl == 2.0

    with pytest.raises(ImproperlyConfiguredException):
        Provide(async_value, cache_ttl=5)


async def test_cache_computes_once(anyio_backend: str) -> None:
    cache = DependencyCache()
    calls = 0

    async def factory() -> int:
        nonlocal calls
        calls += 1
        return calls

    key = object()
    assert await cache.get_or_compute(key, factory) == 1
    assert await cache.get_or_compute(key, factory) == 1
    assert calls == 1
    assert key in cache
    assert cache.peek(key) == 1


async def test_cache_single_flight_fill(anyio_backend: str) -> None:
    import anyio

    cache = DependencyCache()
    calls = 0
    gate = Event()
    joined = 0
    results: list[int] = []

    async def factory() -> int:
        nonlocal calls
        calls += 1
        await gate.wait()
        return calls

    async def consumer() -> None:
        nonlocal joined
        joined += 1
        results.append(await cache.get_or_compute("key", factory))

    async with create_task_group() as tg:
        for _ in range(5):
            tg.start_soon(consumer)
        # ``joined`` is incremented in the same synchronous block as the cache election,
        # so once all consumers joined exactly one is the leader and the rest are waiting.
        while joined < 5:
            await anyio.sleep(0)
        gate.set()

    assert calls == 1
    assert results == [1, 1, 1, 1, 1]
    assert cache.peek("key") == 1


async def test_cache_failure_propagates_to_waiters(anyio_backend: str) -> None:
    import anyio

    cache = DependencyCache()
    calls = 0
    gate = Event()
    joined = 0

    async def factory() -> int:
        nonlocal calls
        calls += 1
        await gate.wait()
        raise RuntimeError("boom")

    errors: list[BaseException] = []

    async def consumer() -> None:
        nonlocal joined
        joined += 1
        try:
            await cache.get_or_compute("key", factory)
        except BaseException as exc:
            errors.append(exc)

    async with create_task_group() as tg:
        for _ in range(3):
            tg.start_soon(consumer)
        while joined < 3:
            await anyio.sleep(0)
        gate.set()

    assert calls == 1
    assert len(errors) == 3
    assert all(isinstance(exc, RuntimeError) and exc.args == ("boom",) for exc in errors)
    # a failure must never be cached as a value: the next lookup computes again
    assert cache.peek("key") is Empty
    assert "key" not in cache

    async def fixed_factory() -> int:
        nonlocal calls
        calls += 1
        return 42

    assert await cache.get_or_compute("key", fixed_factory) == 42
    assert calls == 2


async def test_cache_base_exception_triggers_retry_for_waiters(anyio_backend: str) -> None:
    import anyio

    cache = DependencyCache()
    calls = 0
    gate = Event()
    joined = 0

    class LeaderAborted(BaseException):
        pass

    async def factory() -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            await gate.wait()
            raise LeaderAborted()
        return 99

    results: list[int] = []
    errors: list[BaseException] = []

    async def consumer() -> None:
        nonlocal joined
        joined += 1
        try:
            results.append(await cache.get_or_compute("key", factory))
        except BaseException as exc:
            errors.append(exc)

    async with create_task_group() as tg:
        for _ in range(3):
            tg.start_soon(consumer)
        while joined < 3:
            await anyio.sleep(0)
        gate.set()

    # only the leader observes its own BaseException; waiters retry and share the new value
    assert len(errors) == 1
    assert isinstance(errors[0], LeaderAborted)
    assert sorted(results) == [99, 99]
    # the retried computation produced a regular cached entry
    assert cache.peek("key") == 99
    assert calls == 2


async def test_cache_invalidate_forces_recompute(anyio_backend: str) -> None:
    cache = DependencyCache()
    calls = 0

    async def factory() -> int:
        nonlocal calls
        calls += 1
        return calls

    key = object()
    assert await cache.get_or_compute(key, factory) == 1
    assert await cache.get_or_compute(key, factory) == 1
    cache.invalidate(key)
    assert key not in cache
    assert await cache.get_or_compute(key, factory) == 2


async def test_cache_invalidate_all(anyio_backend: str) -> None:
    cache = DependencyCache()

    async def factory() -> int:
        return 1

    first = object()
    second = object()
    await cache.get_or_compute(first, factory)
    await cache.get_or_compute(second, factory)
    assert first in cache and second in cache

    cache.invalidate()

    assert first not in cache and second not in cache


async def test_cache_invalidate_during_inflight(anyio_backend: str) -> None:
    import anyio

    cache = DependencyCache()
    calls = 0
    gate = Event()
    joined = 0
    results: list[int] = []
    errors: list[BaseException] = []

    async def factory() -> int:
        nonlocal calls
        calls += 1
        await gate.wait()
        return calls

    async def consumer() -> None:
        nonlocal joined
        joined += 1
        try:
            results.append(await cache.get_or_compute("key", factory))
        except BaseException as exc:
            errors.append(exc)

    async with create_task_group() as tg:
        tg.start_soon(consumer)
        tg.start_soon(consumer)
        while joined < 2:
            await anyio.sleep(0)
        # invalidate while the single computation is running
        cache.invalidate("key")
        gate.set()

    assert not errors
    # requests that joined the in-flight computation receive that complete value
    assert sorted(results) == [1, 1]
    # the invalidated value is not retained, so the next lookup starts a new computation
    assert await cache.get_or_compute("key", factory) == 2
    assert calls == 2


async def test_cache_ttl_expiry(anyio_backend: str) -> None:
    import anyio

    cache = DependencyCache()
    calls = 0

    async def factory() -> int:
        nonlocal calls
        calls += 1
        return calls

    key = object()
    assert await cache.get_or_compute(key, factory, ttl=0.05) == 1
    assert await cache.get_or_compute(key, factory, ttl=0.05) == 1
    assert calls == 1
    await anyio.sleep(0.07)
    assert key not in cache
    assert await cache.get_or_compute(key, factory, ttl=0.05) == 2
    assert calls == 2


async def test_cache_distinct_keys_are_independent(anyio_backend: str) -> None:
    cache = DependencyCache()

    async def factory_one() -> int:
        return 1

    async def factory_two() -> int:
        return 2

    assert await cache.get_or_compute("one", factory_one) == 1
    assert await cache.get_or_compute("two", factory_two) == 2
    assert cache.peek("one") == 1
    assert cache.peek("two") == 2


async def test_provide_direct_call_invalidation(anyio_backend: str) -> None:
    calls = 0

    async def factory() -> int:
        nonlocal calls
        calls += 1
        return calls

    provide = Provide(factory, use_cache=True)
    assert await provide() == 1
    assert await provide() == 1
    assert provide.value == 1
    provide.invalidate()
    assert provide.value is Empty
    assert await provide() == 2


async def test_provide_direct_call_single_flight_failure(anyio_backend: str) -> None:
    calls = 0
    gate = Event()

    async def factory() -> Any:
        nonlocal calls
        calls += 1
        await gate.wait()
        raise ValueError("nope")

    provide = Provide(factory, use_cache=True)
    errors: list[BaseException] = []

    async def consumer() -> None:
        try:
            await provide()
        except BaseException as exc:
            errors.append(exc)

    async with create_task_group() as tg:
        tg.start_soon(consumer)
        tg.start_soon(consumer)
        gate.set()

    assert calls == 1
    assert len(errors) == 2
    assert {type(exc) for exc in errors} == {ValueError}
    assert provide.value is Empty
    # the failed computation is not cached
    with pytest.raises(ValueError, match="nope"):
        await provide()
    assert calls == 2
