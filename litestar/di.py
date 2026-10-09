# pyright: reportUnnecessaryTypeIgnoreComment=false

from __future__ import annotations

from datetime import timedelta
from enum import StrEnum
from inspect import isasyncgenfunction, isclass, isfunction, isgeneratorfunction, ismethod
from time import monotonic
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeAlias, TypeVar

from anyio import Event

from litestar.exceptions import ImproperlyConfiguredException
from litestar.plugins import DIPlugin, PluginRegistry
from litestar.types import Empty, TypeDecodersSequence
from litestar.utils import ensure_async_callable
from litestar.utils.helpers import unwrap_partial
from litestar.utils.predicates import is_async_callable
from litestar.utils.signature import ParsedSignature
from litestar.utils.warnings import (
    warn_implicit_sync_to_thread,
    warn_sync_to_thread_with_async_callable,
    warn_sync_to_thread_with_generator,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from litestar._signature import SignatureModel
    from litestar.dto import AbstractDTO
    from litestar.types import AnyCallable


__all__ = (
    "DependencyCache",
    "DependencyCacheScope",
    "NamedDependency",
    "Provide",
)


T = TypeVar("T")


class Dependency:
    def __init__(self, kind: Literal["named"] = "named") -> None:
        # currently only 'named' kind is supported, and this value isn't used. 3.0 will
        # introduce the 'type' kind
        self.kind = kind


NamedDependency: TypeAlias = Annotated[T, Dependency(kind="named")]
"""
Mark a parameter for name-based dependency injection.

The name of the function parameter will be used as the name for the dependency to inject.
"""


class DependencyCacheScope(StrEnum):
    """Enumeration of the possible lifetime scopes of a cached dependency value.

    - ``PROVIDER``: the value is stored on the :class:`Provide` instance and lives for as
      long as the provider object itself. This is the scope selected by ``use_cache=True``
      and matches the historical caching behaviour.
    - ``APP``: the value is stored on the assembled :class:`Litestar <litestar.Litestar>`
      application and is shared across requests, until it is invalidated through
      ``app.dependency_cache``.
    - ``REQUEST``: the value is stored in the request scope and is discarded as soon as the
      request finishes. Concurrent resolutions of the same provider within a request share
      a single value.
    """

    PROVIDER = "provider"
    APP = "app"
    REQUEST = "request"


class _ReadyCacheEntry:
    """An immutable, fully computed cache entry."""

    __slots__ = ("expires_at", "value")

    def __init__(self, value: Any, expires_at: float | None) -> None:
        self.value = value
        self.expires_at = expires_at

    def is_expired(self, now: float) -> bool:
        return self.expires_at is not None and now >= self.expires_at


class _InFlightCacheEntry:
    """A marker for a value that is currently being computed.

    Every coroutine that encounters an in-flight entry waits on ``event`` and, once it is
    set, observes either a complete ``value`` or a complete ``error``. A partially computed
    value can therefore never be observed.

    ``base_error`` marks a failure with a non-shareable ``BaseException`` (e.g. a task
    cancellation), in which case waiters discard the generation and run a new election
    instead of receiving an exception that belongs to another task.
    """

    __slots__ = ("base_error", "error", "event", "value")

    def __init__(self, event: Event) -> None:
        self.event = event
        self.value: Any = Empty
        self.error: BaseException | None = None
        self.base_error = False


class DependencyCache:
    """Cache for dependency values with scoped lifetime, TTL, single-flight fill and
    explicit invalidation.

    A cache instance is owned by a lifetime scope:

    - each :class:`Provide` lazily creates one for :attr:`DependencyCacheScope.PROVIDER`
      values,
    - the :class:`Litestar <litestar.Litestar>` application owns one for
      :attr:`DependencyCacheScope.APP` values (``app.dependency_cache``),
    - each request owns one for :attr:`DependencyCacheScope.REQUEST` values.

    All public, non-async helpers are safe to call from synchronous code. The cache relies
    on the cooperative scheduling of the running event loop: every state inspection and
    update happens in a synchronous, non-awaiting section and is therefore atomic with
    respect to other cache operations.
    """

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        # Entries are keyed by the object identity of the (possibly unhashable) cache
        # key. The owning scope always keeps :class:`Provide` instances alive for at least
        # as long as their entries, so ``id`` reuse cannot collide with a live entry.
        self._entries: dict[int, _ReadyCacheEntry | _InFlightCacheEntry] = {}

    async def get_or_compute(
        self,
        key: Any,
        factory: Callable[[], Awaitable[Any]],
        ttl: float | None = None,
    ) -> Any:
        """Return the cached value for ``key`` or compute it via ``factory``.

        If another coroutine is already computing the value, the coroutine awaits the same
        result instead of invoking ``factory`` a second time. When the computation fails,
        every waiting coroutine receives the same exception and the next lookup computes
        the value again.

        Args:
            key: Cache key, in practice the :class:`Provide` instance. Keys are matched by
                object identity.
            factory: A zero-argument async callable that produces the value.
            ttl: Optional lifetime in seconds. When given, the value is recomputed once it
                has expired.

        Returns:
            The cached or the freshly computed value.
        """
        storage_key = id(key)
        while True:
            # This whole election section contains no ``await`` and is therefore atomic:
            # callers either join an existing computation or become its single leader.
            entry = self._entries.get(storage_key)
            now = monotonic()
            if isinstance(entry, _ReadyCacheEntry) and not entry.is_expired(now):
                return entry.value

            if isinstance(entry, _InFlightCacheEntry):
                flight = entry
                is_leader = False
            else:
                flight = _InFlightCacheEntry(Event())
                self._entries[storage_key] = flight
                is_leader = True

            if not is_leader:
                await flight.event.wait()
                # the leader was cancelled (or hit another non-shareable ``BaseException``);
                # discard this generation and run a new election instead of adopting its error
                if flight.base_error:
                    continue
                if flight.error is not None:
                    raise flight.error
                return flight.value

            try:
                value = await factory()
            except BaseException as exc:
                # ``Exception`` failures are shared verbatim with the in-flight waiters, and
                # the entry is detached so the next lookup computes again. Non-shareable
                # ``BaseException`` failures (cancellations etc.) instead trigger a retry.
                if isinstance(exc, Exception):
                    flight.error = exc
                else:
                    flight.base_error = True
                if self._entries.get(storage_key) is flight:
                    del self._entries[storage_key]
                flight.event.set()
                raise

            # ``value`` is published before the event is set, and the entry is replaced as one
            # immutable object, so every observer sees either the old or the new value whole.
            flight.value = value
            if self._entries.get(storage_key) is flight:
                self._entries[storage_key] = _ReadyCacheEntry(value, expires_at=monotonic() + ttl if ttl else None)
            flight.event.set()
            return value

    def peek(self, key: Any) -> Any:
        """Return the cached value for ``key`` without computing or waiting, or ``Empty``."""
        entry = self._entries.get(id(key))
        if isinstance(entry, _ReadyCacheEntry):
            return entry.value
        return Empty

    def invalidate(self, key: Any | None = None) -> None:
        """Invalidate cached values.

        When ``key`` is given, only that entry is removed; otherwise every entry is
        removed. An in-flight computation is allowed to complete for the requests already
        waiting on it, but its result is discarded and any subsequent lookup recomputes
        the value.

        Args:
            key: An optional cache key, in practice the :class:`Provide` instance. Keys are
                matched by object identity.

        Returns:
            None
        """
        if key is None:
            self._entries.clear()
        else:
            self._entries.pop(id(key), None)

    def __contains__(self, key: Any) -> bool:
        entry = self._entries.get(id(key))
        return isinstance(entry, _ReadyCacheEntry) and not entry.is_expired(monotonic())


def _normalize_cache_config(
    use_cache: bool | DependencyCacheScope,
    cache_ttl: int | float | timedelta | None,
) -> tuple[bool, DependencyCacheScope | None, float | None]:
    """Normalize the ``use_cache`` / ``cache_ttl`` constructor arguments."""
    if isinstance(use_cache, DependencyCacheScope):
        enabled = True
        scope: DependencyCacheScope | None = use_cache
    elif use_cache:
        enabled = True
        scope = DependencyCacheScope.PROVIDER
    else:
        enabled = False
        scope = None

    if cache_ttl is not None:
        if not enabled:
            raise ImproperlyConfiguredException("cache_ttl requires caching to be enabled via 'use_cache'")
        if isinstance(cache_ttl, timedelta):
            cache_ttl = cache_ttl.total_seconds()
        cache_ttl = float(cache_ttl)

    return enabled, scope, cache_ttl


class Provide:
    """Wrapper class for dependency injection"""

    __slots__ = (
        "_parsed_fn_signature",
        "_provider_cache",
        "_signature_model",
        "cache_scope",
        "cache_ttl",
        "dependency",
        "has_async_generator_dependency",
        "has_sync_callable",
        "has_sync_generator_dependency",
        "sync_to_thread",
        "use_cache",
        "value",
    )

    dependency: AnyCallable
    cache_scope: DependencyCacheScope | None
    cache_ttl: float | None

    def __init__(
        self,
        dependency: AnyCallable | type[Any],
        use_cache: bool | DependencyCacheScope = False,
        sync_to_thread: bool | None = None,
        cache_ttl: int | float | timedelta | None = None,
    ) -> None:
        """Initialize ``Provide``

        Args:
            dependency: Callable to call or class to instantiate. The result is then injected as a dependency.
            use_cache: Cache the dependency return value. ``True`` caches the value on the provider itself
                (:attr:`DependencyCacheScope.PROVIDER`, the historical behaviour). A
                :class:`DependencyCacheScope` member selects a different lifetime scope. Defaults to False.
            sync_to_thread: Run sync code in an async thread. Defaults to False.
            cache_ttl: If given, the cached value expires after this amount of time and is recomputed on the
                next lookup. Accepts seconds or a :class:`datetime.timedelta`. Requires caching to be enabled.
        """
        if not callable(dependency):
            raise ImproperlyConfiguredException("Provider dependency must be a callable value")

        self.use_cache, self.cache_scope, self.cache_ttl = _normalize_cache_config(use_cache, cache_ttl)

        is_class_dependency = isclass(dependency)
        is_function = isfunction(dependency)
        is_method = ismethod(dependency)
        is_callable_instance = not is_class_dependency and not is_function and not is_method
        if is_class_dependency or is_callable_instance:
            check_target = dependency.__call__  # type: ignore[operator]
        else:
            check_target = dependency
        self.has_sync_generator_dependency = isgeneratorfunction(check_target)
        self.has_async_generator_dependency = isasyncgenfunction(check_target)

        has_generator_dependency = self.has_sync_generator_dependency or self.has_async_generator_dependency

        if has_generator_dependency and self.use_cache:
            raise ImproperlyConfiguredException(
                "Cannot cache generator dependency, consider using Lifespan Context instead."
            )

        has_sync_callable = is_class_dependency or not is_async_callable(dependency)
        if sync_to_thread is not None:
            if has_generator_dependency:
                warn_sync_to_thread_with_generator(dependency, stacklevel=3)  # type: ignore[arg-type]
            elif not has_sync_callable:
                warn_sync_to_thread_with_async_callable(dependency, stacklevel=3)
        elif has_sync_callable and not has_generator_dependency:
            warn_implicit_sync_to_thread(dependency, stacklevel=3)
        if sync_to_thread and has_sync_callable:
            self.dependency = ensure_async_callable(dependency)
            self.has_sync_callable = False
        else:
            self.dependency = dependency
            self.has_sync_callable = has_sync_callable

        self.sync_to_thread = bool(sync_to_thread)
        self.value: Any = Empty
        self._provider_cache: DependencyCache | None = None
        self._parsed_fn_signature: ParsedSignature | None = None
        self._signature_model: type[SignatureModel] | None = None

    @property
    def signature_model(self) -> type[SignatureModel]:
        if self._signature_model is None:
            raise ValueError(f"Cannot access signature model of Provider {self} because it is not finalized")
        return self._signature_model

    @property
    def parsed_fn_signature(self) -> ParsedSignature:
        if self._parsed_fn_signature is None:
            raise ValueError(f"Cannot access parsed signature of Provider {self} because it is not finalized")
        return self._parsed_fn_signature

    def get_provider_cache(self) -> DependencyCache:
        """Return the lazily created :class:`DependencyCache` backing the provider scope."""
        if self._provider_cache is None:
            self._provider_cache = DependencyCache()
        return self._provider_cache

    def invalidate(self) -> None:
        """Invalidate the value cached on this provider.

        Only affects values stored in the :attr:`DependencyCacheScope.PROVIDER` scope, i.e.
        those cached by ``use_cache=True`` or by calling the provider directly. Values
        cached in the app or request scope must be invalidated through the owning cache
        (``app.dependency_cache``).
        """
        self.value = Empty
        if self._provider_cache is not None:
            self._provider_cache.invalidate(self)

    def finalize(
        self,
        *,
        plugins: PluginRegistry,
        signature_namespace: dict[str, Any],
        dependency_keys: set[str],
        data_dto: type[AbstractDTO] | None,
        type_decoders: TypeDecodersSequence,
    ) -> None:
        if self._parsed_fn_signature is None:
            dependency = unwrap_partial(self.dependency)
            plugin = next(
                (p for p in plugins.di if isinstance(p, DIPlugin) and p.has_typed_init(dependency)),
                None,
            )
            if plugin:
                signature, init_type_hints = plugin.get_typed_init(dependency)
                self._parsed_fn_signature = ParsedSignature.from_signature(signature, init_type_hints)
            else:
                self._parsed_fn_signature = ParsedSignature.from_fn(dependency, signature_namespace)

        if self._signature_model is None:
            from litestar._signature import SignatureModel

            self._signature_model = SignatureModel.create(
                dependency_name_set=dependency_keys,
                fn=self.dependency,
                parsed_signature=self.parsed_fn_signature,
                data_dto=data_dto,
                type_decoders=type_decoders,
            )

    async def _invoke_dependency(self, kwargs: dict[str, Any]) -> Any:
        """Invoke the wrapped dependency with the resolved kwargs."""
        if self.has_sync_callable:
            return self.dependency(**kwargs)
        return await self.dependency(**kwargs)

    async def resolve_value(self, kwargs: dict[str, Any], cache: DependencyCache | None) -> Any:
        """Resolve the dependency value, using ``cache`` when one is bound to the scope.

        When ``cache`` is ``None`` the dependency is invoked directly, preserving the
        non-cached behaviour.

        Args:
            kwargs: Resolved keyword arguments for the dependency.
            cache: The scope-bound :class:`DependencyCache` or ``None``.

        Returns:
            The dependency value.
        """
        if cache is None:
            return await self._invoke_dependency(kwargs)
        value = await cache.get_or_compute(
            self,
            lambda: self._invoke_dependency(kwargs),
            self.cache_ttl,
        )
        # keep the historical ``value`` attribute in sync for provider-scoped caches
        if self._provider_cache is not None and cache is self._provider_cache:
            self.value = cache.peek(self)
        return value

    async def __call__(self, **kwargs: Any) -> Any:
        """Call the provider's dependency.

        Direct calls cache the value in the :attr:`DependencyCacheScope.PROVIDER` scope when
        caching is enabled, which matches the historical ``use_cache`` behaviour.
        """
        if not self.use_cache:
            return await self._invoke_dependency(kwargs)

        cache = self.get_provider_cache()
        value = await cache.get_or_compute(
            self,
            lambda: self._invoke_dependency(kwargs),
            self.cache_ttl,
        )
        self.value = cache.peek(self)
        return value

    def __eq__(self, other: Any) -> bool:
        # check if memory address is identical, otherwise compare attributes
        return other is self or (
            isinstance(other, self.__class__)
            and other.dependency == self.dependency
            and other.use_cache == self.use_cache
            and other.value == self.value
        )
