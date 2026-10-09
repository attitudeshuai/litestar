from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import anyio

from litestar.datastructures import Cookie, MutableScopeHeaders
from litestar.enums import ScopeType
from litestar.exceptions import (
    ConflictException,
    ImproperlyConfiguredException,
    NotAuthorizedException,
    ServiceUnavailableException,
)
from litestar.middleware.session.base import ONE_DAY_IN_SECONDS, BaseBackendConfig, BaseSessionBackend
from litestar.types import Empty, Message, Scopes, ScopeSession
from litestar.utils.dataclass import extract_dataclass_items

__all__ = ("ServerSideSessionBackend", "ServerSideSessionConfig")

#: Version of the versioned session envelope used when optimistic concurrency is enabled.
ENVELOPE_VERSION = 1
#: Distinctive marker key identifying a versioned session envelope.
_ENVELOPE_MARKER_KEY = "_litestar_session"
#: Store key prefix for revoked session-IDs.
_REVOKED_SESSION_PREFIX = "_litestar_revoked_session:"
#: Store key prefix for per-user revocation timestamps.
_REVOKED_USER_PREFIX = "_litestar_revoked_user:"
_CONFLICT_POLICIES = ("reject", "merge", "overwrite")
_MISSING = object()


if TYPE_CHECKING:
    from litestar import Litestar
    from litestar.connection import ASGIConnection
    from litestar.stores.base import Store
    from litestar.types import Scope
    from litestar.utils.scope.state import ScopeState


def _three_way_merge(
    base: dict[str, Any],
    remote: dict[str, Any],
    local: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Deterministically merge ``local`` changes over ``remote`` based on a common ``base``.

    Rules per key:

    - only one side changed relative to ``base`` -> that side wins
    - both sides made the same change -> keep it
    - the sides made different changes (incl. one deleting while the other modifies) -> conflict

    Returns:
        A tuple of the merged dictionary and a boolean indicating a conflict.
    """
    merged = dict(remote)
    for key in set(base) | set(remote) | set(local):
        base_value = base.get(key, _MISSING)
        remote_value = remote.get(key, _MISSING)
        local_value = local.get(key, _MISSING)

        if local_value == base_value:
            # the local request did not touch this key -> the remote state wins
            continue
        if remote_value == base_value:
            # the remote state did not touch this key -> the local change wins
            if local_value is _MISSING:
                merged.pop(key, None)
            else:
                merged[key] = local_value
            continue
        if local_value == remote_value:
            # both sides made the exact same change
            continue
        # both sides changed this key differently - this cannot be merged deterministically
        return {}, True
    return merged, False


class ServerSideSessionBackend(BaseSessionBackend["ServerSideSessionConfig"]):
    """Base class for server-side backends.

    Implements :class:`BaseSessionBackend` and defines and interface which subclasses can
    implement to facilitate the storage of session data.
    """

    def __init__(self, config: ServerSideSessionConfig) -> None:
        """Initialize ``ServerSideSessionBackend``

        Args:
            config: A subclass of ``ServerSideSessionConfig``
        """
        super().__init__(config=config)
        self._key_locks: dict[str, anyio.Lock] = {}

    async def get(self, session_id: str, store: Store) -> bytes | None:
        """Retrieve data associated with ``session_id``.

        Args:
            session_id: The session-ID
            store: Store to retrieve the session data from

        Returns:
            The session data, if existing, otherwise ``None``.
        """
        max_age = int(self.config.max_age) if self.config.max_age is not None else None
        return await store.get(session_id, renew_for=max_age if self.config.renew_on_access else None)

    async def set(self, session_id: str, data: bytes, store: Store) -> None:
        """Store ``data`` under the ``session_id`` for later retrieval.

        If there is already data associated with ``session_id``, replace
        it with ``data`` and reset its expiry time

        Args:
            session_id: The session-ID
            data: Serialized session data
            store: Store to save the session data in

        Returns:
            None
        """
        expires_in = int(self.config.max_age) if self.config.max_age is not None else None
        await store.set(session_id, data, expires_in=expires_in)

    async def delete(self, session_id: str, store: Store) -> None:
        """Delete the data associated with ``session_id``. Fails silently if no such session-ID exists.

        Args:
            session_id: The session-ID
            store: Store to delete the session data from

        Returns:
            None
        """
        await store.delete(session_id)

    def get_session_id(self, connection: ASGIConnection) -> str:
        """Try to fetch session id from the connection. If one does not exist, generate one.

        If a session ID already exists in the cookies, it is returned.
        If there is no ID in the cookies but one in the connection state, then the session exists but has not yet
        been returned to the user.
        Otherwise, a new session must be created.

        Args:
            connection: Originating ASGIConnection containing the scope
        Returns:
            Session id str or None if the concept of a session id does not apply.
        """
        session_id = connection.cookies.get(self.config.key)
        if not session_id or session_id == "null":
            session_id = connection.get_session_id()
            if not session_id:
                session_id = self.generate_session_id()
        return session_id

    def generate_session_id(self) -> str:
        """Generate a new session-ID, with
        n=:attr:`session_id_bytes <ServerSideSessionConfig.session_id_bytes>` random bytes.

        Returns:
            A session-ID
        """
        return secrets.token_hex(self.config.session_id_bytes)

    # -- optimistic concurrency / invalidation helpers -------------------------------------

    @property
    def _strict(self) -> bool:
        return bool(self.config.optimistic_concurrency)

    def _expires_in(self) -> int | None:
        return int(self.config.max_age) if self.config.max_age is not None else None

    @staticmethod
    def _revoked_session_key(session_id: str) -> str:
        return _REVOKED_SESSION_PREFIX + session_id

    @staticmethod
    def _revoked_user_key(user_id: Any) -> str:
        return _REVOKED_USER_PREFIX + str(user_id)

    async def _convert_storage_error(self, exc: Exception) -> Exception:
        """Translate a store failure into an explicit request failure in strict mode.

        In legacy mode the original exception is re-raised unchanged.
        """
        if isinstance(exc, NotAuthorizedException | ConflictException | ServiceUnavailableException):
            return exc
        if self._strict:
            return ServiceUnavailableException(detail="session storage is unavailable")
        return exc

    async def _read_store(self, store: Store, key: str, *, renew: bool) -> bytes | None:
        """Read a key from the store, converting backend failures to explicit errors."""
        try:
            if renew:
                max_age = int(self.config.max_age) if self.config.max_age is not None else None
                return await store.get(key, renew_for=max_age if self.config.renew_on_access else None)
            return await store.get(key)
        except Exception as exc:
            raise await self._convert_storage_error(exc) from exc

    async def _write_store(self, store: Store, key: str, value: bytes) -> None:
        """Write a key to the store, converting backend failures to explicit errors."""
        try:
            await store.set(key, value, expires_in=self._expires_in())
        except Exception as exc:
            raise await self._convert_storage_error(exc) from exc

    def _encode_envelope(
        self, data: dict[str, Any], version: int, iat: float, scope: Scope | None
    ) -> bytes:
        envelope = {
            _ENVELOPE_MARKER_KEY: ENVELOPE_VERSION,
            "version": version,
            "iat": iat,
            "data": data,
        }
        return self.serialize_data(envelope, scope)

    @staticmethod
    def _decode_envelope(raw: bytes) -> tuple[int, float, dict[str, Any]]:
        """Decode a stored payload into ``(version, iat, data)``.

        Payloads written before optimistic concurrency was enabled are treated as version ``1``
        issued at epoch zero, so user-level revocation remains conservative for them.
        """
        parsed = ServerSideSessionBackend.deserialize_data(raw)
        if (
            isinstance(parsed, dict)
            and parsed.get(_ENVELOPE_MARKER_KEY) == ENVELOPE_VERSION
            and isinstance(parsed.get("data"), dict)
        ):
            return int(parsed["version"]), float(parsed["iat"]), parsed["data"]
        if not isinstance(parsed, dict):
            msg = "session data is not a mapping"
            raise ServiceUnavailableException(detail=msg)
        return 1, 0.0, parsed

    async def _assert_session_not_revoked(self, session_id: str, store: Store) -> None:
        if await self._read_store(store, self._revoked_session_key(session_id), renew=False) is not None:
            raise NotAuthorizedException(detail="session has been invalidated")

    async def _user_session_is_revoked(self, user_id: Any, iat: float, store: Store) -> bool:
        """Return whether ``user_id`` revoked sessions issued at/before ``iat``."""
        marker = await self._read_store(store, self._revoked_user_key(user_id), renew=False)
        if marker is None:
            return False
        try:
            revoked_at = float(marker)
        except (TypeError, ValueError) as exc:
            raise ServiceUnavailableException(detail="session revocation data is corrupt") from exc
        return iat <= revoked_at

    async def _revoke_session_id(self, session_id: str, store: Store) -> None:
        """Record a revocation marker for ``session_id`` and delete its data.

        The marker is written first and kept for ``max_age`` seconds, so an in-flight request
        recreating the key still cannot make the old ID usable again.
        """
        await self._write_store(store, self._revoked_session_key(session_id), b"1")
        try:
            await store.delete(session_id)
        except Exception as exc:
            raise await self._convert_storage_error(exc) from exc

    def _get_key_lock(self, session_id: str) -> anyio.Lock:
        lock = self._key_locks.get(session_id)
        if lock is None:
            lock = anyio.Lock()
            self._key_locks[session_id] = lock
        return lock

    def _release_key_lock(self, session_id: str, lock: anyio.Lock) -> None:
        locked = getattr(lock, "locked", None)
        if callable(locked) and not locked() and self._key_locks.get(session_id) is lock:
            self._key_locks.pop(session_id, None)

    async def _clear_existing_session(self, session_id: str, base: bytes | None, store: Store) -> None:
        """Delete an existing session, failing on a conflicting concurrent write."""
        if base is None:
            # nothing was loaded under this ID at request start - best-effort removal only
            try:
                await store.delete(session_id)
            except Exception as exc:
                raise await self._convert_storage_error(exc) from exc
            return
        try:
            deleted = await store.compare_and_delete(session_id, base)
        except Exception as exc:
            raise await self._convert_storage_error(exc) from exc
        if not deleted:
            raise ConflictException(detail="session was modified by another request")

    async def _cas(
        self, store: Store, key: str, old: bytes | None, new: bytes, expires_in: int | None
    ) -> bool:
        """Perform an atomic compare-and-set, converting backend failures to explicit errors."""
        try:
            return await store.compare_and_set(key, old, new, expires_in=expires_in)
        except Exception as exc:
            raise await self._convert_storage_error(exc) from exc

    async def _persist_overwrite(
        self, session_id: str, scope_session: dict[str, Any], iat: float, scope: Scope, store: Store
    ) -> bytes:
        current = await self._read_store(store, session_id, renew=False)
        if current is None:
            version, effective_iat = 1, iat
        else:
            current_version, current_iat, _ = self._decode_envelope(current)
            version, effective_iat = current_version + 1, current_iat
        payload = self._encode_envelope(scope_session, version, effective_iat, scope)
        await self._write_store(store, session_id, payload)
        return payload

    async def _persist_reject(
        self,
        session_id: str,
        scope_session: dict[str, Any],
        base: bytes | None,
        iat: float,
        scope: Scope,
        store: Store,
    ) -> bytes:
        if base is None:
            version, effective_iat = 1, iat
        else:
            base_version, base_iat, _ = self._decode_envelope(base)
            version, effective_iat = base_version + 1, base_iat
        payload = self._encode_envelope(scope_session, version, effective_iat, scope)
        if not await self._cas(store, session_id, base, payload, self._expires_in()):
            raise ConflictException(detail="session was modified by another request")
        return payload

    async def _persist_merge(
        self,
        session_id: str,
        scope_session: dict[str, Any],
        base: bytes | None,
        iat: float,
        scope: Scope,
        store: Store,
    ) -> bytes:
        base_data = self._decode_envelope(base)[2] if base is not None else {}
        for _ in range(2):
            remote = await self._read_store(store, session_id, renew=False)
            if remote is None:
                if base is not None:
                    # the session was deleted by another request - do not resurrect it
                    raise ConflictException(detail="session was deleted by another request")
                remote_data, remote_version, remote_iat = {}, 0, iat
            else:
                remote_version, remote_iat, remote_data = self._decode_envelope(remote)

            merged, conflicting = _three_way_merge(base=base_data, remote=remote_data, local=scope_session)
            if conflicting:
                raise ConflictException(detail="conflicting concurrent session modifications")

            payload = self._encode_envelope(merged, remote_version + 1, remote_iat, scope)
            if await self._cas(store, session_id, remote, payload, self._expires_in()):
                return payload

        raise ConflictException(detail="conflicting concurrent session modifications")

    async def _persist_versioned(
        self,
        session_id: str,
        scope_session: dict[str, Any],
        base: bytes | None,
        iat: float,
        scope: Scope,
        store: Store,
    ) -> bytes:
        """Write the session using the configured conflict policy.

        Must be called while holding the per-key lock.
        """
        policy = self.config.conflict_policy
        if policy == "overwrite":
            return await self._persist_overwrite(session_id, scope_session, iat, scope, store)
        if policy == "reject":
            return await self._persist_reject(session_id, scope_session, base, iat, scope, store)
        return await self._persist_merge(session_id, scope_session, base, iat, scope, store)

    async def _store_cleared_session(
        self,
        session_id: str,
        cookie_id: str | None,
        regenerated: bool,
        state: ScopeState,
        headers: MutableScopeHeaders,
        cookie_params: dict[str, Any],
        store: Store,
    ) -> None:
        if regenerated and cookie_id and cookie_id != "null":
            await self._revoke_session_id(cookie_id, store)
        else:
            base = None if state.session_base is Empty else state.session_base
            await self._clear_existing_session(session_id, base=base, store=store)
        headers.add(
            "Set-Cookie",
            Cookie(value="null", key=self.config.key, expires=0, **cookie_params).to_header(header=""),
        )
        state.session_id_regenerated = False

    async def _store_legacy_session(
        self,
        session_id: str,
        cookie_id: str | None,
        regenerated: bool,
        scope_session: dict[str, Any],
        state: ScopeState,
        scope: Scope,
        headers: MutableScopeHeaders,
        cookie_params: dict[str, Any],
        store: Store,
    ) -> None:
        serialised_data = self.serialize_data(scope_session, scope)
        if regenerated:
            await self._write_store(store, session_id, serialised_data)
            if cookie_id and cookie_id != "null" and cookie_id != session_id:
                await self._revoke_session_id(cookie_id, store)
        else:
            await self.set(session_id=session_id, data=serialised_data, store=store)
        headers.add(
            "Set-Cookie", Cookie(value=session_id, key=self.config.key, **cookie_params).to_header(header="")
        )
        state.session_id_regenerated = False

    async def _store_regenerated_session(
        self,
        session_id: str,
        cookie_id: str | None,
        scope_session: dict[str, Any],
        scope: Scope,
        store: Store,
    ) -> bytes:
        payload = self._encode_envelope(scope_session, version=1, iat=time.time(), scope=scope)
        lock = self._get_key_lock(session_id)
        async with lock:
            try:
                stored = await self._cas(store, session_id, None, payload, self._expires_in())
            finally:
                self._release_key_lock(session_id, lock)
        if not stored:
            # the freshly generated ID collides with existing data - never overwrite it
            raise ServiceUnavailableException(detail="could not issue a new session ID")
        if cookie_id and cookie_id != "null" and cookie_id != session_id:
            await self._revoke_session_id(cookie_id, store)
        return payload

    async def store_in_message(self, scope_session: ScopeSession, message: Message, connection: ASGIConnection) -> None:
        """Store the necessary information in the outgoing ``Message`` by setting a cookie containing the session-ID.

        If the session is empty, a null-cookie will be set. Otherwise, the serialised
        data will be stored using :meth:`set <ServerSideSessionBackend.set>`, under the current session-id. If no session-ID
        exists, a new ID will be generated using :meth:`generate_session_id <ServerSideSessionBackend.generate_session_id>`.

        When session-ID regeneration has been signalled (e.g. after a privilege change), a fresh ID is
        issued, the data is stored under it and the old ID is invalidated.

        Args:
            scope_session: Current session to store
            message: Outgoing send-message
            connection: Originating ASGIConnection containing the scope

        Returns:
            None
        """
        scope = connection.scope
        store = self.config.get_store_from_app(scope["app"])
        headers = MutableScopeHeaders.from_message(message)
        state = connection._connection_state

        cookie_id = connection.cookies.get(self.config.key)
        regenerated = bool(state.session_id_regenerated)
        if regenerated:
            session_id = self.generate_session_id()
            state.session_id = session_id
        else:
            session_id = self.get_session_id(connection)

        cookie_params = dict(extract_dataclass_items(self.config, exclude_none=True, include=Cookie.__dict__.keys()))

        if scope_session is Empty:
            await self._store_cleared_session(
                session_id, cookie_id, regenerated, state, headers, cookie_params, store
            )
            return

        if not self._strict:
            await self._store_legacy_session(
                session_id, cookie_id, regenerated, scope_session, state, scope, headers, cookie_params, store
            )
            return

        if state.session_revoked and not regenerated:
            # the request carried a user-invalidated identity without replacing it
            state.session_id_regenerated = False
            if not scope_session:
                # anonymous read-only request: leave the response untouched
                return
            raise NotAuthorizedException(
                detail="session has been invalidated for this user; "
                "regenerate the session ID after re-authentication"
            )

        base = None if state.session_base is Empty else state.session_base
        if regenerated:
            await self._store_regenerated_session(session_id, cookie_id, scope_session, scope, store)
        elif base is not None and self._decode_envelope(base)[2] == scope_session:
            # the session was not modified by the request - leave stored state untouched
            state.session_id_regenerated = False
            return
        else:
            iat = time.time() if state.session_iat is Empty else float(state.session_iat)
            lock = self._get_key_lock(session_id)
            async with lock:
                try:
                    await self._persist_versioned(
                        session_id=session_id,
                        scope_session=scope_session,
                        base=base,
                        iat=iat,
                        scope=scope,
                        store=store,
                    )
                finally:
                    self._release_key_lock(session_id, lock)

        headers.add(
            "Set-Cookie", Cookie(value=session_id, key=self.config.key, **cookie_params).to_header(header="")
        )
        state.session_id_regenerated = False

    async def load_from_connection(self, connection: ASGIConnection) -> dict[str, Any]:
        """Load session data from a connection and return it as a dictionary to be used in the current application
        scope.

        The session-ID will be gathered from a cookie with the key set in
        :attr:`BaseBackendConfig.key`. If a cookie is found, its value will be used as the session-ID and data associated
        with this ID will be loaded using :meth:`get <ServerSideSessionBackend.get>`.
        If no cookie was found or no data was loaded from the store, this will return an
        empty dictionary.

        When optimistic concurrency is enabled, revocation markers are checked and stored
        payloads that cannot be read result in an explicit failure instead of an empty session.

        Args:
            connection: An ASGIConnection instance

        Returns:
            The current session data
        """
        state = connection._connection_state
        state.session_base = None
        state.session_iat = Empty
        state.session_revoked = False

        session_id = connection.cookies.get(self.config.key)
        if not session_id or session_id == "null":
            return {}

        store = self.config.get_store_from_app(connection.scope["app"])
        data = await self._read_store(store, session_id, renew=True)

        if self._strict:
            await self._assert_session_not_revoked(session_id, store)

        if data is None:
            return {}

        if not self._strict:
            return self.deserialize_data(data)

        try:
            _version, iat, session_data = self._decode_envelope(data)
        except (NotAuthorizedException, ServiceUnavailableException):
            raise
        except Exception as exc:
            raise ServiceUnavailableException(detail="session data is corrupt or unreadable") from exc

        state.session_base = data
        state.session_iat = iat

        if self.config.user_id_key is not None:
            user_id = session_data.get(self.config.user_id_key)
            if user_id is not None and await self._user_session_is_revoked(user_id, iat, store):
                # the authenticated identity was invalidated: present an anonymous session
                # instead of the stored one. Protected resources must reject via the auth
                # layer; re-authentication with ID rotation establishes a new identity.
                state.session_revoked = True
                return {}

        return session_data

    # -- explicit invalidation API ---------------------------------------------------------

    async def invalidate_session_id(
        self,
        session_id: str,
        *,
        app: Litestar | None = None,
        store: Store | None = None,
    ) -> None:
        """Immediately invalidate a single session-ID.

        Further requests presenting the ID are rejected while optimistic concurrency is enabled,
        even if an in-flight request recreates data under it.

        Args:
            session_id: The session-ID to invalidate
            app: The Litestar application used to resolve the store, if ``store`` is not given
            store: Explicit store to use, taking precedence over ``app``
        """
        store = store or self.config.get_store_from_app(app)
        await self._revoke_session_id(session_id, store)

    async def invalidate_user(
        self,
        user_id: Any,
        *,
        app: Litestar | None = None,
        store: Store | None = None,
    ) -> None:
        """Immediately invalidate all sessions of a user issued before now.

        Requires :attr:`ServerSideSessionConfig.user_id_key` so the user identity can be
        located in a session. Affected requests are treated as anonymous (so auth-protected
        resources reject them), attempts to persist the invalidated identity without an ID
        rotation are rejected with ``401``, and sessions created (with a new ID) after this
        call are not affected.

        Args:
            user_id: The user identifier, matching the value stored under ``user_id_key``
            app: The Litestar application used to resolve the store, if ``store`` is not given
            store: Explicit store to use, taking precedence over ``app``
        """
        store = store or self.config.get_store_from_app(app)
        await self._write_store(store, self._revoked_user_key(user_id), str(time.time()).encode())


@dataclass
class ServerSideSessionConfig(BaseBackendConfig[ServerSideSessionBackend]):
    """Base configuration for server side backends."""

    _backend_class = ServerSideSessionBackend

    session_id_bytes: int = field(default=32)
    """Number of bytes used to generate a random session-ID."""
    renew_on_access: bool = field(default=False)
    """Renew expiry times of sessions when they're being accessed"""
    key: str = field(default="session")
    """Key to use for the cookie inside the header, e.g. ``session=<data>`` where ``session`` is the cookie key and
    ``<data>`` is the session data.

    Notes:
        - If a session cookie exceeds 4KB in size it is split. In this case the key will be of the format
          ``session-{segment number}``.

    """
    max_age: int = field(default=ONE_DAY_IN_SECONDS * 14)
    """Maximal age of the cookie before its invalidated."""
    scopes: Scopes = field(default_factory=lambda: {ScopeType.HTTP, ScopeType.WEBSOCKET})
    """Scopes for the middleware - options are ``http`` and ``websocket`` with the default being both"""
    path: str = field(default="/")
    """Path fragment that must exist in the request url for the cookie to be valid.

    Defaults to ``'/'``.
    """
    domain: str | None = field(default=None)
    """Domain for which the cookie is valid."""
    secure: bool = field(default=False)
    """Https is required for the cookie."""
    httponly: bool = field(default=True)
    """Forbids javascript to access the cookie via 'Document.cookie'."""
    samesite: Literal["lax", "strict", "none"] = field(default="lax")
    """Controls whether or not a cookie is sent with cross-site requests. Defaults to ``lax``."""
    exclude: str | list[str] | None = field(default=None)
    """A pattern or list of patterns to skip in the session middleware."""
    exclude_opt_key: str = field(default="skip_session")
    """An identifier to use on routes to disable the session middleware for a particular route."""
    store: str = "sessions"
    """Name of the :class:`Store <.stores.base.Store>` to use"""
    optimistic_concurrency: bool = field(default=False)
    """Store sessions as versioned envelopes and enforce explicit concurrency and failure semantics.

    When enabled:

    - every write is guarded by :attr:`conflict_policy` instead of silently overwriting concurrent changes
    - revoked session-IDs / users are rejected on load
    - failures of the storage backend result in a ``503`` response instead of an empty session being used

    When disabled (the default), behaviour is identical to a plain server-side session backend.
    """
    conflict_policy: Literal["reject", "merge", "overwrite"] = field(default="reject")
    """Policy applied when a session is written but another request has modified it in the meantime.

    - ``reject`` (default): fail the write with a ``409 Conflict`` response
    - ``merge``: deterministically merge non-conflicting key changes; fail with ``409`` when both
      requests changed the same key differently
    - ``overwrite``: last write wins, while still storing a comparable version

    Only takes effect when :attr:`optimistic_concurrency` is enabled.
    """
    user_id_key: str | None = field(default=None)
    """Key under which the user identifier is stored in the session data.

    Required for per-user invalidation (:meth:`ServerSideSessionBackend.invalidate_user`). When set,
    sessions whose issue time predates a user-level invalidation are rejected.
    """

    def __post_init__(self) -> None:
        if len(self.key) < 1 or len(self.key) > 256:
            raise ImproperlyConfiguredException("key must be a string with a length between 1-256")
        if self.max_age < 1:
            raise ImproperlyConfiguredException("max_age must be greater than 0")
        if self.conflict_policy not in _CONFLICT_POLICIES:
            raise ImproperlyConfiguredException(
                f"conflict_policy must be one of {', '.join(_CONFLICT_POLICIES)}"
            )
        if self.user_id_key is not None and not isinstance(self.user_id_key, str):
            raise ImproperlyConfiguredException("user_id_key must be a string or None")
        if isinstance(self.user_id_key, str) and not self.user_id_key:
            raise ImproperlyConfiguredException("user_id_key must not be empty")

    def get_store_from_app(self, app: Litestar) -> Store:
        """Get the store defined in :attr:`store` from an :class:`Litestar <.app.Litestar>` instance"""
        return app.stores.get(self.store)

    async def invalidate_session_id(self, app: Litestar, session_id: str) -> None:
        """Convenience wrapper around :meth:`ServerSideSessionBackend.invalidate_session_id`."""
        await self._backend_class(config=self).invalidate_session_id(session_id, app=app)

    async def invalidate_user_sessions(self, app: Litestar, user_id: Any) -> None:
        """Convenience wrapper around :meth:`ServerSideSessionBackend.invalidate_user`."""
        await self._backend_class(config=self).invalidate_user(user_id, app=app)
