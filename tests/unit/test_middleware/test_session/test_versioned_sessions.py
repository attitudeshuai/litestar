"""Tests for optimistic concurrency, session-ID rotation and explicit invalidation."""

import asyncio
import time
from typing import Any

import httpx
import pytest

from litestar import Litestar, Request, get
from litestar.exceptions import ImproperlyConfiguredException
from litestar.middleware.session.server_side import (
    _ENVELOPE_MARKER_KEY,
    ENVELOPE_VERSION,
    ServerSideSessionConfig,
    _three_way_merge,
)
from litestar.serialization import decode_json, encode_json
from litestar.status_codes import (
    HTTP_200_OK,
    HTTP_401_UNAUTHORIZED,
    HTTP_409_CONFLICT,
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_503_SERVICE_UNAVAILABLE,
)
from litestar.stores.memory import MemoryStore
from litestar.testing import TestClient


def _make_envelope(data: dict[str, Any], version: int = 2, iat: float | None = None) -> bytes:
    return encode_json(
        {
            _ENVELOPE_MARKER_KEY: ENVELOPE_VERSION,
            "version": version,
            "iat": iat if iat is not None else time.time(),
            "data": data,
        }
    )


def _build_app(store: MemoryStore, config: ServerSideSessionConfig, handlers: list[Any]) -> Litestar:
    return Litestar(handlers, middleware=[config.middleware], stores={"sessions": store})


# -- config validation -----------------------------------------------------------------------


def test_invalid_conflict_policy_raises() -> None:
    with pytest.raises(ImproperlyConfiguredException):
        ServerSideSessionConfig(optimistic_concurrency=True, conflict_policy="nope")  # type: ignore[arg-type]


def test_empty_user_id_key_raises() -> None:
    with pytest.raises(ImproperlyConfiguredException):
        ServerSideSessionConfig(optimistic_concurrency=True, user_id_key="")


# -- storage format ---------------------------------------------------------------------------


def test_legacy_mode_stores_plain_dict(memory_store: MemoryStore) -> None:
    @get("/")
    def handler(request: Request) -> None:
        request.set_session({"foo": "bar"})

    app = _build_app(memory_store, ServerSideSessionConfig(), [handler])
    with TestClient(app) as client:
        res = client.get("/")
        assert res.status_code == HTTP_200_OK
        sid = res.cookies["session"]

    raw = decode_json(memory_store._store[sid].data)
    assert raw == {"foo": "bar"}


def test_strict_mode_stores_versioned_envelope(memory_store: MemoryStore) -> None:
    config = ServerSideSessionConfig(optimistic_concurrency=True)

    @get("/")
    def handler(request: Request) -> dict:
        request.session["counter"] = request.session.get("counter", 0) + 1
        return request.session

    app = _build_app(memory_store, config, [handler])
    with TestClient(app) as client:
        first = client.get("/")
        sid = first.cookies["session"]
        envelope = decode_json(memory_store._store[sid].data)
        assert envelope[_ENVELOPE_MARKER_KEY] == ENVELOPE_VERSION
        assert envelope["version"] == 1
        assert envelope["data"] == {"counter": 1}

        client.get("/")
        envelope = decode_json(memory_store._store[sid].data)
        assert envelope["version"] == 2
        assert envelope["data"] == {"counter": 2}


# -- conflict policies ------------------------------------------------------------------------


def _simulate_remote_write(request: Request, data: dict[str, Any], version: int = 2) -> None:
    sid = request.cookies["session"]
    store = request.app.stores.get("sessions")
    asyncio.get_running_loop().create_task(store.set(sid, _make_envelope(data, version=version)))


def test_reject_policy_conflicting_write_returns_409(memory_store: MemoryStore) -> None:
    @get("/init")
    def init(request: Request) -> None:
        request.set_session({"a": 1})

    @get("/conflict")
    async def conflict(request: Request) -> None:
        await request.app.stores.get("sessions").set(
            request.cookies["session"], _make_envelope({"a": 2}, version=2)
        )
        request.session["a"] = 3

    config = ServerSideSessionConfig(optimistic_concurrency=True, conflict_policy="reject")
    app = _build_app(memory_store, config, [init, conflict])
    with TestClient(app) as client:
        client.get("/init")
        res = client.get("/conflict")

    assert res.status_code == HTTP_409_CONFLICT


def test_read_only_request_does_not_bump_version(memory_store: MemoryStore) -> None:
    @get("/init")
    def init(request: Request) -> None:
        request.set_session({"a": 1})

    @get("/show")
    def show(request: Request) -> dict:
        return request.session

    config = ServerSideSessionConfig(optimistic_concurrency=True)
    app = _build_app(memory_store, config, [init, show])
    with TestClient(app) as client:
        sid = client.get("/init").cookies["session"]
        client.get("/show")
        envelope = decode_json(memory_store._store[sid].data)

    assert envelope["version"] == 1
    assert envelope["data"] == {"a": 1}


def test_reject_policy_accepts_unmodified_base(memory_store: MemoryStore) -> None:
    @get("/init")
    def init(request: Request) -> None:
        request.set_session({"a": 1})

    @get("/ok")
    def ok(request: Request) -> dict:
        request.session["b"] = 2
        return request.session

    config = ServerSideSessionConfig(optimistic_concurrency=True, conflict_policy="reject")
    app = _build_app(memory_store, config, [init, ok])
    with TestClient(app) as client:
        client.get("/init")
        res = client.get("/ok")

    assert res.status_code == HTTP_200_OK
    assert res.json() == {"a": 1, "b": 2}


def test_merge_policy_merges_disjoint_keys(memory_store: MemoryStore) -> None:
    @get("/init")
    def init(request: Request) -> None:
        request.set_session({"a": 1})

    @get("/merge")
    async def merge(request: Request) -> dict:
        await request.app.stores.get("sessions").set(
            request.cookies["session"], _make_envelope({"a": 1, "b": 2}, version=2)
        )
        request.session["c"] = 3
        return request.session

    @get("/show")
    def show(request: Request) -> dict:
        return request.session

    config = ServerSideSessionConfig(optimistic_concurrency=True, conflict_policy="merge")
    app = _build_app(memory_store, config, [init, merge, show])
    with TestClient(app) as client:
        client.get("/init")
        res = client.get("/merge")
        assert res.status_code == HTTP_200_OK
        assert client.get("/show").json() == {"a": 1, "b": 2, "c": 3}


def test_merge_policy_same_key_conflict_returns_409(memory_store: MemoryStore) -> None:
    @get("/init")
    def init(request: Request) -> None:
        request.set_session({"a": 1})

    @get("/conflict")
    async def conflict(request: Request) -> None:
        await request.app.stores.get("sessions").set(
            request.cookies["session"], _make_envelope({"a": 2}, version=2)
        )
        request.session["a"] = 3

    config = ServerSideSessionConfig(optimistic_concurrency=True, conflict_policy="merge")
    app = _build_app(memory_store, config, [init, conflict])
    with TestClient(app) as client:
        client.get("/init")
        res = client.get("/conflict")

    assert res.status_code == HTTP_409_CONFLICT


def test_merge_policy_remote_delete_returns_409(memory_store: MemoryStore) -> None:
    @get("/init")
    def init(request: Request) -> None:
        request.set_session({"a": 1})

    @get("/conflict")
    async def conflict(request: Request) -> None:
        await request.app.stores.get("sessions").delete(request.cookies["session"])
        request.session["a"] = 3

    config = ServerSideSessionConfig(optimistic_concurrency=True, conflict_policy="merge")
    app = _build_app(memory_store, config, [init, conflict])
    with TestClient(app) as client:
        client.get("/init")
        res = client.get("/conflict")

    assert res.status_code == HTTP_409_CONFLICT


def test_overwrite_policy_last_write_wins(memory_store: MemoryStore) -> None:
    @get("/init")
    def init(request: Request) -> None:
        request.set_session({"a": 1})

    @get("/overwrite")
    async def overwrite(request: Request) -> dict:
        await request.app.stores.get("sessions").set(
            request.cookies["session"], _make_envelope({"a": 2}, version=2)
        )
        request.session["a"] = 3
        return request.session

    config = ServerSideSessionConfig(optimistic_concurrency=True, conflict_policy="overwrite")
    app = _build_app(memory_store, config, [init, overwrite])
    with TestClient(app) as client:
        client.get("/init")
        res = client.get("/overwrite")

    assert res.status_code == HTTP_200_OK
    assert res.json() == {"a": 3}
    envelope = decode_json(memory_store._store[res.cookies["session"]].data)
    assert envelope["version"] == 3


def test_clearing_a_concurrently_modified_session_returns_409(memory_store: MemoryStore) -> None:
    @get("/init")
    def init(request: Request) -> None:
        request.set_session({"a": 1})

    @get("/clear")
    async def clear(request: Request) -> None:
        await request.app.stores.get("sessions").set(
            request.cookies["session"], _make_envelope({"a": 2}, version=2)
        )
        request.clear_session()

    config = ServerSideSessionConfig(optimistic_concurrency=True)
    app = _build_app(memory_store, config, [init, clear])
    with TestClient(app) as client:
        client.get("/init")
        res = client.get("/clear")

    assert res.status_code == HTTP_409_CONFLICT


async def test_concurrent_requests_reject_one_write(memory_store: MemoryStore) -> None:
    event = asyncio.Event()
    arrived = {"count": 0}

    @get("/init")
    def init(request: Request) -> None:
        request.set_session({"a": 1})

    @get("/bump")
    async def bump(request: Request) -> dict:
        arrived["count"] += 1
        if arrived["count"] == 2:
            event.set()
        else:
            await asyncio.wait_for(event.wait(), timeout=5)
        request.session["a"] = request.session["a"] + 1
        return request.session

    config = ServerSideSessionConfig(optimistic_concurrency=True, conflict_policy="reject")
    app = _build_app(memory_store, config, [init, bump])

    async with app.lifespan():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            init_res = await client.get("/init")
            sid = init_res.cookies["session"]
            headers = {"Cookie": f"session={sid}"}
            results = await asyncio.gather(
                client.get("/bump", headers=headers),
                client.get("/bump", headers=headers),
            )

    assert sorted(res.status_code for res in results) == [HTTP_200_OK, HTTP_409_CONFLICT]
    envelope = decode_json(memory_store._store[sid].data)
    assert envelope["data"] == {"a": 2}


async def test_concurrent_requests_merge_disjoint_changes(memory_store: MemoryStore) -> None:
    event = asyncio.Event()
    arrived = {"count": 0}

    @get("/init")
    def init(request: Request) -> None:
        request.set_session({"a": 1})

    @get("/show")
    def show(request: Request) -> dict:
        return request.session

    @get("/add")
    async def add(request: Request) -> dict:
        key = request.query_params["key"]
        arrived["count"] += 1
        if arrived["count"] == 2:
            event.set()
        else:
            await asyncio.wait_for(event.wait(), timeout=5)
        request.session[key] = key
        return request.session

    config = ServerSideSessionConfig(optimistic_concurrency=True, conflict_policy="merge")
    app = _build_app(memory_store, config, [init, show, add])

    async with app.lifespan():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            init_res = await client.get("/init")
            sid = init_res.cookies["session"]
            headers = {"Cookie": f"session={sid}"}
            results = await asyncio.gather(
                client.get("/add?key=x", headers=headers),
                client.get("/add?key=y", headers=headers),
            )
            final = await client.get("/show", headers=headers)

    assert [res.status_code for res in results] == [HTTP_200_OK, HTTP_200_OK]
    assert final.status_code == HTTP_200_OK
    assert final.json() == {"a": 1, "x": "x", "y": "y"}
    envelope = decode_json(memory_store._store[sid].data)
    assert envelope["data"] == {"a": 1, "x": "x", "y": "y"}


# -- session-ID rotation ----------------------------------------------------------------------


def test_session_id_rotation_invalidates_old_id(memory_store: MemoryStore) -> None:
    @get("/init")
    def init(request: Request) -> None:
        request.set_session({"k": "v"})

    @get("/show")
    def show(request: Request) -> dict:
        return request.session

    @get("/rotate")
    def rotate(request: Request) -> dict:
        request.session["k2"] = "v2"
        request.regenerate_session_id()
        return request.session

    config = ServerSideSessionConfig(optimistic_concurrency=True)
    app = _build_app(memory_store, config, [init, show, rotate])
    with TestClient(app) as client:
        client.get("/init")
        old_sid = client.cookies["session"]
        res = client.get("/rotate")
        new_sid = res.cookies["session"]

        assert new_sid != old_sid
        assert new_sid in memory_store._store
        assert old_sid not in memory_store._store

        old_cookie = client.get("/show", headers={"Cookie": f"session={old_sid}"})
        assert old_cookie.status_code == HTTP_401_UNAUTHORIZED

        new_cookie = client.get("/show", headers={"Cookie": f"session={new_sid}"})
        assert new_cookie.status_code == HTTP_200_OK
        assert new_cookie.json() == {"k": "v", "k2": "v2"}


# -- explicit invalidation --------------------------------------------------------------------


async def test_invalidate_session_id_via_backend(memory_store: MemoryStore) -> None:
    @get("/init")
    def init(request: Request) -> None:
        request.set_session({"foo": "bar"})

    config = ServerSideSessionConfig(optimistic_concurrency=True)
    app = _build_app(memory_store, config, [init])
    with TestClient(app) as client:
        res = client.get("/init")
        sid = res.cookies["session"]
        await config._backend_class(config=config).invalidate_session_id(sid, store=memory_store)
        follow_up = client.get("/init")

    assert follow_up.status_code == HTTP_401_UNAUTHORIZED


async def test_invalidate_session_id_via_config(memory_store: MemoryStore) -> None:
    @get("/init")
    def init(request: Request) -> None:
        request.set_session({"foo": "bar"})

    config = ServerSideSessionConfig(optimistic_concurrency=True)
    app = _build_app(memory_store, config, [init])
    with TestClient(app) as client:
        sid = client.get("/init").cookies["session"]
        await config.invalidate_session_id(app, sid)
        assert client.get("/init").status_code == HTTP_401_UNAUTHORIZED


async def test_invalidate_user_rejects_old_sessions_only(memory_store: MemoryStore) -> None:
    @get("/login")
    def login(request: Request) -> None:
        request.set_session({"user_id": "user-1"})
        request.regenerate_session_id()

    @get("/whoami")
    def whoami(request: Request) -> dict:
        return request.session

    @get("/mutate")
    def mutate(request: Request) -> None:
        request.session["extra"] = True

    config = ServerSideSessionConfig(optimistic_concurrency=True, user_id_key="user_id")
    app = _build_app(memory_store, config, [login, whoami, mutate])
    with TestClient(app) as client:
        client.get("/login")
        old_sid = client.cookies["session"]
        assert client.get("/whoami").status_code == HTTP_200_OK

        await config.invalidate_user_sessions(app, "user-1")

        # reads are treated as anonymous - the stored identity is not handed out
        anonymous = client.get("/whoami")
        assert anonymous.status_code == HTTP_200_OK
        assert anonymous.json() == {}

        # persisting the invalidated identity without rotation is rejected
        assert client.get("/mutate").status_code == HTTP_401_UNAUTHORIZED

        # re-authentication with rotation establishes a brand new identity
        await asyncio.sleep(0.01)
        client.get("/login")
        new_sid = client.cookies["session"]
        assert new_sid != old_sid
        assert client.get("/whoami").status_code == HTTP_200_OK

        # the old ID is now hard-revoked as well
        assert client.get("/whoami", headers={"Cookie": f"session={old_sid}"}).status_code == (
            HTTP_401_UNAUTHORIZED
        )


async def test_user_revocation_does_not_affect_other_users(memory_store: MemoryStore) -> None:
    @get("/login")
    def login(request: Request) -> None:
        request.set_session({"user_id": request.query_params["user"]})

    @get("/me")
    def me(request: Request) -> dict:
        return request.session

    config = ServerSideSessionConfig(optimistic_concurrency=True, user_id_key="user_id")
    app = _build_app(memory_store, config, [login, me])
    with TestClient(app) as client:
        client.get("/login?user=alice")
        alice_sid = client.cookies["session"]

        await config.invalidate_user_sessions(app, "bob")
        assert client.get("/me", headers={"Cookie": f"session={alice_sid}"}).status_code == HTTP_200_OK


# -- storage failure paths --------------------------------------------------------------------


class _FailingGetStore(MemoryStore):
    async def get(self, key: str, renew_for: Any = None) -> bytes | None:
        raise RuntimeError("storage is down")


class _FailingWriteStore(MemoryStore):
    async def set(self, key: str, value: Any, expires_in: Any = None) -> None:
        raise RuntimeError("storage is down")

    async def compare_and_set(self, *args: Any, **kwargs: Any) -> bool:
        raise RuntimeError("storage is down")

    async def compare_and_delete(self, *args: Any, **kwargs: Any) -> bool:
        raise RuntimeError("storage is down")


def test_storage_read_failure_strict_returns_503() -> None:
    @get("/")
    def handler(request: Request) -> None:
        return

    config = ServerSideSessionConfig(optimistic_concurrency=True)
    store = _FailingGetStore()
    app = _build_app(store, config, [handler])
    with TestClient(app) as client:
        res = client.get("/", headers={"Cookie": "session=deadbeef"})

    assert res.status_code == HTTP_503_SERVICE_UNAVAILABLE


def test_storage_read_failure_legacy_keeps_old_behaviour() -> None:
    @get("/")
    def handler(request: Request) -> None:
        return

    app = _build_app(_FailingGetStore(), ServerSideSessionConfig(), [handler])
    with TestClient(app) as client:
        res = client.get("/", headers={"Cookie": "session=deadbeef"})

    assert res.status_code == HTTP_500_INTERNAL_SERVER_ERROR


def test_no_cookie_with_failing_store_is_normal_empty_session() -> None:
    # without a cookie the store is never consulted, so a backend outage must not be surfaced
    @get("/")
    def handler(request: Request) -> dict:
        return {"ok": True}

    config = ServerSideSessionConfig(optimistic_concurrency=True)
    app = _build_app(_FailingGetStore(), config, [handler])
    with TestClient(app, raise_server_exceptions=False) as client:
        res = client.get("/")

    assert res.status_code == HTTP_200_OK


def test_storage_write_failure_strict_returns_503(memory_store: MemoryStore) -> None:
    @get("/")
    def handler(request: Request) -> None:
        request.set_session({"foo": "bar"})

    config = ServerSideSessionConfig(optimistic_concurrency=True)
    app = _build_app(_FailingWriteStore(), config, [handler])
    with TestClient(app, raise_server_exceptions=False) as client:
        res = client.get("/")

    assert res.status_code == HTTP_503_SERVICE_UNAVAILABLE


# -- store CAS primitives ---------------------------------------------------------------------


async def test_memory_store_compare_and_set() -> None:
    store = MemoryStore()
    assert await store.compare_and_set("k", None, b"a") is True
    assert await store.compare_and_set("k", None, b"b") is False
    assert await store.compare_and_set("k", b"a", b"b") is True
    assert await store.compare_and_set("k", b"a", b"c") is False
    assert await store.get("k") == b"b"


async def test_memory_store_compare_and_delete() -> None:
    store = MemoryStore()
    await store.set("k", b"a")
    assert await store.compare_and_delete("k", b"b") is False
    assert await store.get("k") == b"a"
    assert await store.compare_and_delete("k", b"a") is True
    assert await store.get("k") is None
    assert await store.compare_and_delete("missing", b"a") is False


# -- merge rules ------------------------------------------------------------------------------


def test_three_way_merge_rules() -> None:
    merged, conflict = _three_way_merge(
        base={"shared": 0, "only_base": 1},
        remote={"shared": 0, "only_remote": 2},
        local={"shared": 0, "only_local": 3},
    )
    assert conflict is False
    assert merged == {"shared": 0, "only_remote": 2, "only_local": 3}

    # local deletion is applied when remote did not touch the key
    merged, conflict = _three_way_merge(base={"a": 1}, remote={"a": 1}, local={})
    assert conflict is False
    assert merged == {}

    # remote deletion is kept when local did not touch the key
    merged, conflict = _three_way_merge(base={"a": 1}, remote={}, local={"a": 1})
    assert conflict is False
    assert merged == {}

    # both sides delete - fine
    merged, conflict = _three_way_merge(base={"a": 1}, remote={}, local={})
    assert conflict is False
    assert merged == {}

    # divergent changes to the same key -> deterministic conflict
    _, conflict = _three_way_merge(base={"a": 1}, remote={"a": 2}, local={"a": 3})
    assert conflict is True

    # one deletes while the other modifies -> conflict
    _, conflict = _three_way_merge(base={"a": 1}, remote={}, local={"a": 3})
    assert conflict is True
    _, conflict = _three_way_merge(base={"a": 1}, remote={"a": 2}, local={})
    assert conflict is True
