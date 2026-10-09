from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING

from litestar.exceptions import ImproperlyConfiguredException
from litestar.status_codes import HTTP_503_SERVICE_UNAVAILABLE, WS_1001_GOING_AWAY
from litestar.utils import normalize_path

if TYPE_CHECKING:
    from collections.abc import Sequence
    from signal import Signals

__all__ = ("DrainConfig",)


def _to_seconds(value: float | timedelta, name: str) -> float:
    if isinstance(value, timedelta):
        value = value.total_seconds()
    if value < 0:
        raise ImproperlyConfiguredException(f"'{name}' must not be negative, got {value!r}")
    return value


@dataclass(frozen=True)
class DrainConfig:
    """Configuration of the shutdown drain behaviour.

    Passing an instance of this config to :class:`Litestar <litestar.Litestar>` via the
    ``drain_config`` parameter enables the drain capability. When it is not passed,
    request handling and the lifespan hooks behave exactly as without the capability.

    When a shutdown notification is received (the ASGI ``lifespan.shutdown`` event, an
    operating system signal configured through :attr:`signals`, or an explicit
    ``await app.begin_drain()`` call), the application transitions to the
    :class:`DrainState.DRAINING <litestar.enums.DrainState>` state:

    - New HTTP requests are rejected with :attr:`rejection_status_code` (``503`` by
      default) and a configurable ``Retry-After`` hint, unless their path matches
      :attr:`probe_paths`.
    - New websocket connections are closed with :attr:`websocket_close_code`, unless
      their path matches :attr:`probe_paths`.
    - Requests already being processed continue running to completion.
    - If all in-flight requests finish within :attr:`grace_period`, the application
      transitions to :class:`DrainState.DRAINED <litestar.enums.DrainState>` and the
      shutdown hooks are executed.
    - If the deadline elapses first, the remaining requests are cancelled, recorded on
      the returned :class:`DrainResult <litestar.drain.DrainResult>`, logged, and only
      then are the shutdown hooks executed.
    """

    grace_period: float | timedelta = 30.0
    """Maximum time in seconds (or a :class:`~datetime.timedelta`) to wait for
    in-flight requests to complete after the drain has begun. When the deadline
    elapses, still-running requests are aborted."""
    probe_paths: Sequence[str] = field(default_factory=tuple)
    """HTTP / websocket paths that remain serviceable while the application is
    :class:`DrainState.DRAINING <litestar.enums.DrainState>`, e.g. health and
    readiness probe paths. Paths are normalized and matched exactly. Once the drain
    completes, no new requests - probe paths included - are admitted."""
    rejection_status_code: int = HTTP_503_SERVICE_UNAVAILABLE
    """Status code returned for HTTP requests that arrive while the application is
    draining and do not match :attr:`probe_paths`."""
    rejection_detail: str = "Service is shutting down"
    """Human readable detail included in the JSON body of a rejection response."""
    retry_after: int | float | timedelta | None = 30
    """Value of the ``Retry-After`` response header (in seconds) attached to rejection
    responses. Set to ``None`` to omit the header. Fractional and
    :class:`~datetime.timedelta` values are rounded up to whole seconds."""
    connection_close: bool = True
    """Whether to attach a ``Connection: close`` header to rejection responses, which
    causes HTTP clients (including keep-alive connections) to reconnect instead of
    retrying on the draining instance."""
    reject_websockets: bool = True
    """Whether to deny new websocket connections that arrive while draining."""
    websocket_close_code: int = WS_1001_GOING_AWAY
    """Close code sent when a new websocket connection is denied while draining."""
    abort_timeout: float | timedelta = 5.0
    """Maximum time in seconds (or a :class:`~datetime.timedelta`) to wait for requests
    to finish after they have been cancelled because the :attr:`grace_period` elapsed.
    Requests that do not stop within this time are logged and the drain proceeds so that
    the shutdown sequence remains deterministic."""
    signals: Sequence[int | Signals] | None = None
    """Optional operating system signals that trigger the drain (POSIX only, e.g.
    ``signal.SIGUSR1``). This allows starting the drain before the server sends the
    ``lifespan.shutdown`` event. Signal handlers are registered once the lifespan has
    started and removed when it ends. On platforms without signal handler support
    (e.g. Windows) a warning is logged and the option is ignored."""

    def __post_init__(self) -> None:
        """Normalize and validate the configuration.

        Raises:
            ImproperlyConfiguredException: If any value is invalid.
        """
        if not 400 <= self.rejection_status_code <= 599:
            raise ImproperlyConfiguredException(
                f"'rejection_status_code' must be in the 4xx-5xx range, got {self.rejection_status_code}"
            )
        if not self.rejection_detail:
            raise ImproperlyConfiguredException("'rejection_detail' must be a non-empty string")

        object.__setattr__(self, "grace_period", _to_seconds(self.grace_period, "grace_period"))
        object.__setattr__(self, "abort_timeout", _to_seconds(self.abort_timeout, "abort_timeout"))
        object.__setattr__(
            self,
            "probe_paths",
            tuple(dict.fromkeys(normalize_path(path) for path in self.probe_paths)),
        )

        retry_after: int | None = None
        if self.retry_after is not None:
            raw_value = (
                self.retry_after.total_seconds()
                if isinstance(self.retry_after, timedelta)
                else float(self.retry_after)
            )
            if raw_value < 0:
                raise ImproperlyConfiguredException(
                    f"'retry_after' must not be negative, got {self.retry_after!r}"
                )
            retry_after = max(0, math.ceil(raw_value))
        object.__setattr__(self, "retry_after", retry_after)

        if self.signals is not None:
            normalized_signals = tuple(self.signals)
            if not normalized_signals:
                raise ImproperlyConfiguredException(
                    "'signals' must contain at least one signal or be set to None"
                )
            object.__setattr__(self, "signals", normalized_signals)
