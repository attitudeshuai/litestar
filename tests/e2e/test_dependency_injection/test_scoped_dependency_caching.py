import time

import pytest

from litestar import Litestar, Request, get
from litestar.di import DependencyCache, DependencyCacheScope, NamedDependency, Provide
from litestar.status_codes import HTTP_200_OK, HTTP_500_INTERNAL_SERVER_ERROR
from litestar.testing import AsyncTestClient, create_test_client
from litestar.types import Empty
from litestar.utils.scope.state import ScopeState


def test_provider_scope_is_shared_across_requests_and_invalidatable() -> None:
    counter = {"value": 0}

    async def counter_dependency() -> int:
        counter["value"] += 1
        return counter["value"]

    provider = Provide(counter_dependency, use_cache=True)

    @get()
    def route(counter: NamedDependency[int]) -> int:
        return counter

    @get("/invalidate")
    def invalidate_route() -> str:
        provider.invalidate()
        return "ok"

    with create_test_client([route, invalidate_route], dependencies={"counter": provider}) as client:
        response = client.get("/")
        assert response.status_code == HTTP_200_OK
        assert response.content == b"1"

        assert client.get("/").content == b"1"
        # the historical ``value`` attribute stays in sync with the provider-scoped cache
        assert provider.value == 1

        assert client.get("/invalidate").content == b"ok"
        assert provider.value is Empty

        assert client.get("/").content == b"2"
        assert provider.value == 2


def test_app_scope_is_shared_across_requests_and_invalidatable() -> None:
    counter = {"value": 0}

    async def counter_dependency() -> int:
        counter["value"] += 1
        return counter["value"]

    provider = Provide(counter_dependency, use_cache=DependencyCacheScope.APP)

    @get()
    def route(counter: NamedDependency[int]) -> int:
        return counter

    @get("/invalidate")
    def invalidate_route(request: Request) -> str:
        request.app.dependency_cache.invalidate(provider)
        return "ok"

    with create_test_client([route, invalidate_route], dependencies={"counter": provider}) as client:
        assert client.get("/").content == b"1"
        assert client.get("/").content == b"1"

        assert client.get("/invalidate").content == b"ok"

        assert client.get("/").content == b"2"

        client.app.dependency_cache.invalidate()
        assert client.get("/").content == b"3"


def test_app_scope_ttl_forces_recompute_after_expiry() -> None:
    counter = {"value": 0}

    async def counter_dependency() -> int:
        counter["value"] += 1
        return counter["value"]

    provider = Provide(counter_dependency, use_cache=DependencyCacheScope.APP, cache_ttl=0.05)

    @get()
    def route(counter: NamedDependency[int]) -> int:
        return counter

    with create_test_client(route, dependencies={"counter": provider}) as client:
        assert client.get("/").content == b"1"
        assert client.get("/").content == b"1"

        time.sleep(0.07)

        assert client.get("/").content == b"2"


def test_request_scope_dies_with_request_and_is_stored_in_scope() -> None:
    counter = {"value": 0}

    async def first_dependency() -> int:
        counter["value"] += 1
        return counter["value"]

    async def second_dependency(first: NamedDependency[int]) -> int:
        return first + 5

    first_provider = Provide(first_dependency, use_cache=DependencyCacheScope.REQUEST)
    second_provider = Provide(second_dependency, use_cache=DependencyCacheScope.REQUEST)

    @get()
    def route(request: Request, first: NamedDependency[int], second: NamedDependency[int]) -> int:
        # the value is held in the request-scoped cache for the duration of the request
        cache = ScopeState.from_scope(request.scope).dependency_cache
        assert cache is not Empty
        assert first_provider in cache
        return first + second

    with create_test_client(
        route,
        dependencies={"first": first_provider, "second": second_provider},
    ) as client:
        assert client.get("/").content == b"7"  # 1 + (1 + 5)
        # nothing leaks across the request boundary: the value is recomputed next time
        assert client.get("/").content == b"9"  # 2 + (2 + 5)


def test_app_scope_failure_is_not_cached() -> None:
    state = {"fail": True}

    async def flaky_dependency() -> int:
        if state["fail"]:
            raise RuntimeError("boom")
        return 5

    provider = Provide(flaky_dependency, use_cache=DependencyCacheScope.APP)

    @get()
    def route(value: NamedDependency[int]) -> int:
        return value

    with create_test_client(route, dependencies={"value": provider}) as client:
        assert client.get("/").status_code == HTTP_500_INTERNAL_SERVER_ERROR
        # the failed computation must not poison the cache: the request is retried
        assert client.get("/").status_code == HTTP_500_INTERNAL_SERVER_ERROR

        state["fail"] = False
        assert client.get("/").content == b"5"
        assert client.get("/").content == b"5"


class _InstrumentedCache(DependencyCache):
    def __init__(self) -> None:
        super().__init__()
        self.lookups = 0

    async def get_or_compute(self, key, factory, ttl=None):  # type: ignore[override]
        self.lookups += 1
        return await super().get_or_compute(key, factory, ttl)


@pytest.mark.asyncio
async def test_app_scope_concurrent_requests_compute_once() -> None:
    import asyncio

    calls = 0
    entered = asyncio.Event()
    gate = asyncio.Event()

    async def gated_dependency() -> int:
        nonlocal calls
        calls += 1
        entered.set()
        await gate.wait()
        return 7

    provider = Provide(gated_dependency, use_cache=DependencyCacheScope.APP)

    @get()
    def route(value: NamedDependency[int]) -> int:
        return value

    app = Litestar([route], dependencies={"value": provider})
    app.dependency_cache = _InstrumentedCache()
    instrumented_cache = app.dependency_cache

    async with AsyncTestClient(app=app) as client:
        first = asyncio.create_task(client.get("/"))
        await entered.wait()

        second = asyncio.create_task(client.get("/"))
        while instrumented_cache.lookups < 2:
            await asyncio.sleep(0)

        gate.set()
        first_response, second_response = await asyncio.gather(first, second)

    assert first_response.status_code == HTTP_200_OK
    assert second_response.status_code == HTTP_200_OK
    assert first_response.content == b"7"
    assert second_response.content == b"7"
    assert calls == 1
