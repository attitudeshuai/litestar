from __future__ import annotations

from typing import TYPE_CHECKING, Any

from litestar.di import DependencyCache, DependencyCacheScope
from litestar.types import Empty
from litestar.utils.scope.state import ScopeState

__all__ = ("DependencyContainer", "create_dependency_batches", "map_dependencies_recursively", "resolve_dependency")


if TYPE_CHECKING:
    from litestar._kwargs.cleanup import DependencyCleanupGroup
    from litestar.connection import ASGIConnection
    from litestar.di import Provide


class DependencyContainer:
    """Dependency graph of a given combination of ``Route`` + ``RouteHandler``"""

    __slots__ = ("dependencies", "key", "provide")

    def __init__(self, key: str, provide: Provide, dependencies: list[DependencyContainer]) -> None:
        """Initialize a dependency.

        Args:
            key: The dependency key
            provide: Provider
            dependencies: List of child nodes
        """
        self.key = key
        self.provide = provide
        self.dependencies = dependencies

    def __eq__(self, other: Any) -> bool:
        # check if memory address is identical, otherwise compare attributes
        return other is self or (isinstance(other, self.__class__) and other.key == self.key)

    def __hash__(self) -> int:
        return hash(self.key)


def get_dependency_cache(provide: Provide, connection: ASGIConnection) -> DependencyCache | None:
    """Return the cache bound to the provider's configured scope, or ``None``.

    Args:
        provide: The dependency provider.
        connection: The current connection, used to reach the app and request scopes.

    Returns:
        A :class:`DependencyCache` when caching is enabled, otherwise ``None``.
    """
    if not provide.use_cache:
        return None

    if provide.cache_scope == DependencyCacheScope.APP:
        return connection.app.dependency_cache

    if provide.cache_scope == DependencyCacheScope.REQUEST:
        connection_state = ScopeState.from_scope(connection.scope)
        cache = connection_state.dependency_cache
        if cache is Empty:
            cache = DependencyCache()
            connection_state.dependency_cache = cache
        return cache

    return provide.get_provider_cache()


async def resolve_dependency(
    dependency: DependencyContainer,
    connection: ASGIConnection,
    kwargs: dict[str, Any],
    cleanup_group: DependencyCleanupGroup,
) -> None:
    """Resolve a given instance of :class:`Dependency <litestar._kwargs.Dependency>`.

    All required sub dependencies must already
    be resolved into the kwargs. The result of the dependency will be stored in the kwargs.

    Args:
        dependency: An instance of :class:`Dependency <litestar._kwargs.Dependency>`
        connection: An instance of :class:`Request <litestar.connection.Request>` or
            :class:`WebSocket <litestar.connection.WebSocket>`.
        kwargs: Any kwargs to pass to the dependency, the result will be stored here as well.
        cleanup_group: DependencyCleanupGroup to which generators returned by ``dependency`` will be added
    """
    provide = dependency.provide
    signature_model = provide.signature_model
    dependency_kwargs = (
        signature_model.parse_values_from_connection_kwargs(connection=connection, kwargs=kwargs)
        if signature_model._fields
        else {}
    )
    value = await provide.resolve_value(dependency_kwargs, cache=get_dependency_cache(provide, connection))

    if provide.has_sync_generator_dependency:
        cleanup_group.add(value)
        value = next(value)
    elif provide.has_async_generator_dependency:
        cleanup_group.add(value)
        value = await anext(value)

    kwargs[dependency.key] = value


def create_dependency_batches(expected_dependencies: set[DependencyContainer]) -> list[set[DependencyContainer]]:
    """Calculate batches for all dependencies, recursively.

    Args:
        expected_dependencies: A set of all direct :class:`Dependencies <litestar._kwargs.Dependency>`.

    Returns:
        A list of batches.
    """
    dependencies_to: dict[DependencyContainer, set[DependencyContainer]] = {}
    for dependency in expected_dependencies:
        if dependency not in dependencies_to:
            map_dependencies_recursively(dependency, dependencies_to)

    batches = []
    while dependencies_to:
        current_batch = {
            dependency
            for dependency, remaining_sub_dependencies in dependencies_to.items()
            if not remaining_sub_dependencies
        }

        for dependency in current_batch:
            del dependencies_to[dependency]
            for others_dependencies in dependencies_to.values():
                others_dependencies.discard(dependency)

        batches.append(current_batch)

    return batches


def map_dependencies_recursively(
    dependency: DependencyContainer, dependencies_to: dict[DependencyContainer, set[DependencyContainer]]
) -> None:
    """Recursively map dependencies to their sub dependencies.

    Args:
        dependency: The current dependency to map.
        dependencies_to: A map of dependency to its sub dependencies.
    """
    dependencies_to[dependency] = set(dependency.dependencies)
    for sub in dependency.dependencies:
        if sub not in dependencies_to:
            map_dependencies_recursively(sub, dependencies_to)
