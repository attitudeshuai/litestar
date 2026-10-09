from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from litestar.enums import DrainState
from litestar.serialization import encode_json
from litestar.utils import normalize_path

if TYPE_CHECKING:
    from litestar.config.drain import DrainConfig
    from litestar.types import Receive, Scope, Send
    from litestar.types.asgi_types import ASGIApp

__all__ = (
    "DrainManager",
    "DrainResult",
    "DrainStatus",
    "InFlightRequest",
    "UnfinishedRequest",
)

logger = logging.getLogger("litestar.drain")


@dataclass(slots=True, eq=False)
class InFlightRequest:
    """Internal representation of a request that has been admitted and is being
    processed.
    """

    method: str
    """HTTP method, or ``"WEBSOCKET"`` for websocket connections."""
    path: str
    """Normalized request path (with ``root_path`` stripped)."""
    started_at: float
    """``time.monotonic()`` timestamp of admission."""
    task: asyncio.Task[Any] | None
    """Asyncio task running the request, if running in one."""
    await_completion: bool = True
    """Whether the request counts as business work reported by
    :attr:`DrainManager.in_flight_request_count`. ``False`` for the request that
    triggered the drain (waiting for it would deadlock) and for probe requests
    admitted while draining (infrastructure observations must not masquerade as
    business traffic). Such requests still keep the drain active until they finish
    and are aborted at the deadline."""
    abortable: bool = True
    """Whether the drain waits for the request and may cancel it when the deadline
    elapses. Only ``False`` for the request that triggered the drain."""


@dataclass(frozen=True, slots=True)
class UnfinishedRequest:
    """Description of a request that had not completed when the drain deadline
    elapsed.
    """

    method: str
    """HTTP method, or ``"WEBSOCKET"`` for websocket connections."""
    path: str
    """Normalized request path."""
    elapsed: float
    """Seconds elapsed between the admission of the request and the abort."""


@dataclass(frozen=True, slots=True)
class DrainResult:
    """Result of a completed drain.

    Returned (or awaited on) by :meth:`DrainManager.begin_drain` and
    :meth:`Litestar.begin_drain <litestar.Litestar.begin_drain>`. Repeated triggers
    receive the exact same instance.
    """

    state: DrainState
    """Always :class:`DrainState.DRAINED <litestar.enums.DrainState>`."""
    timed_out: bool
    """``True`` if the drain ended because the configured deadline elapsed rather than
    because all in-flight requests finished."""
    elapsed: float
    """Seconds the drain took."""
    unfinished: tuple[UnfinishedRequest, ...]
    """Requests still in-flight when the deadline elapsed, in an unspecified order."""
    admitted_requests: int
    """Total number of requests admitted since the manager was created."""
    rejected_requests: int
    """Total number of requests rejected while draining."""
    completed_requests: int
    """Total number of admitted requests that have completed."""
    in_flight_requests: int
    """Number of requests still in-flight when the drain completed."""


@dataclass(frozen=True, slots=True)
class DrainStatus:
    """Point-in-time, internally consistent snapshot of the drain state."""

    state: DrainState
    """Current drain state."""
    in_flight_requests: int
    """Number of requests currently being processed (excluding the drain trigger)."""
    admitted_requests: int
    """Total number of requests admitted since the manager was created."""
    rejected_requests: int
    """Total number of requests rejected while draining."""
    completed_requests: int
    """Total number of admitted requests that have completed."""
    drain_elapsed: float | None
    """Seconds since the drain began, or ``None`` if it has not begun yet."""
    result: DrainResult | None
    """The drain result once the drain has completed, otherwise ``None``."""


class DrainManager:
    """Owns the shutdown drain state machine, in-flight request accounting, the ASGI
    admission gate and the optional signal trigger.

    A manager is always attached to a :class:`Litestar <litestar.Litestar>` instance,
    but it is inert unless constructed with a :class:`DrainConfig
    <litestar.config.drain.DrainConfig>`. When inert, the gate is never installed and
    :meth:`begin_drain` raises, leaving request handling and lifespan hooks unchanged.
    """

    __slots__ = (
        "_drain_future",
        "_drain_started_at",
        "_idle",
        "_inflight",
        "_lock",
        "_result",
        "_signal_handlers",
        "admitted_requests",
        "completed_requests",
        "config",
        "enabled",
        "rejected_requests",
        "state",
    )

    def __init__(self, config: DrainConfig | None = None) -> None:
        """Initialize the manager.

        Args:
            config: The drain configuration, or ``None`` to keep the capability
                disabled.
        """
        self.config = config
        self.enabled = config is not None
        self.state: DrainState = DrainState.RUNNING
        self._inflight: set[InFlightRequest] = set()
        self._idle = asyncio.Event()
        self._idle.set()
        self._lock = asyncio.Lock()
        self._drain_future: asyncio.Future[DrainResult] | None = None
        self._drain_started_at: float | None = None
        self._result: DrainResult | None = None
        self._signal_handlers: dict[int, Any] = {}
        self.admitted_requests = 0
        self.rejected_requests = 0
        self.completed_requests = 0

    # ------------------------------------------------------------------
    # observability
    # ------------------------------------------------------------------

    @property
    def in_flight_request_count(self) -> int:
        """Number of business requests currently being processed.

        Excludes the drain trigger request and probe requests admitted while
        draining.
        """
        return sum(1 for request in self._inflight if request.await_completion)

    @property
    def result(self) -> DrainResult | None:
        """The :class:`DrainResult` once the drain has completed, else ``None``."""
        return self._result

    def get_status(self) -> DrainStatus:
        """Return a consistent snapshot of the current state.

        All state transitions happen in the running event loop thread inside
        uninterruptible critical sections, so the snapshot is consistent when called
        from that thread.
        """
        started_at = self._drain_started_at
        return DrainStatus(
            state=self.state,
            in_flight_requests=self.in_flight_request_count,
            admitted_requests=self.admitted_requests,
            rejected_requests=self.rejected_requests,
            completed_requests=self.completed_requests,
            drain_elapsed=None if started_at is None else time.monotonic() - started_at,
            result=self._result,
        )

    # ------------------------------------------------------------------
    # admission gate
    # ------------------------------------------------------------------

    def _resolve_path(self, scope: Scope) -> str:
        path = scope["path"]
        if root_path := scope.get("root_path", ""):
            path = path.split(root_path, maxsplit=1)[-1]
        return normalize_path(path)

    def _is_probe_path(self, path: str) -> bool:
        return path in cast("DrainConfig", self.config).probe_paths

    async def admit(self, scope: Scope) -> InFlightRequest | None:
        """Admit a request for normal processing.

        The request is counted *before* the state is checked. This closes the race
        with a concurrently starting drain: a request admitted immediately before the
        drain is guaranteed to be tracked and awaited, while a request observing the
        draining state is rejected.

        Probe paths are admitted while :class:`DrainState.DRAINING
        <litestar.enums.DrainState>`. Nothing is admitted once
        :class:`DrainState.DRAINED <litestar.enums.DrainState>` is reached.

        Returns:
            The tracking record if the request may be processed, ``None`` if it must
            be rejected.
        """
        config = cast("DrainConfig", self.config)
        path = self._resolve_path(scope)
        async with self._lock:
            is_probe_path = self._is_probe_path(path)
            admitted = self.state is DrainState.RUNNING or (
                self.state is DrainState.DRAINING
                and (is_probe_path or (scope["type"] == "websocket" and not config.reject_websockets))
            )
            if not admitted:
                self.rejected_requests += 1
                return None
            record = InFlightRequest(
                method=scope.get("method") or "WEBSOCKET",
                path=path,
                started_at=time.monotonic(),
                task=asyncio.current_task(),
                # probe requests admitted *while draining* still get served and the
                # drain still waits for them (they are abortable), but they are not
                # reported as business work. Everything admitted while RUNNING is
                # regular traffic.
                await_completion=self.state is DrainState.RUNNING or not is_probe_path,
            )
            self._inflight.add(record)
            self.admitted_requests += 1
            # every admitted request is abortable, hence keeps the drain active
            self._idle.clear()
            return record

    async def release(self, record: InFlightRequest) -> None:
        """Remove a completed (or aborted) request from the in-flight set."""
        async with self._lock:
            self._inflight.discard(record)
            self.completed_requests += 1
            if not any(request.abortable for request in self._inflight):
                self._idle.set()

    async def gate(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        next_app: ASGIApp,
    ) -> None:
        """ASGI entry point installed in front of the regular application when the
        drain capability is enabled.
        """
        record = await self.admit(scope)
        if record is None:
            await self.reject(scope=scope, send=send)
            return
        try:
            await next_app(scope, receive, send)
        finally:
            await self.release(record)

    async def reject(self, scope: Scope, send: Send) -> None:
        """Send the configured rejection response without entering business
        logic.
        """
        config = cast("DrainConfig", self.config)
        if scope["type"] == "websocket":
            if config.reject_websockets:
                await send({"type": "websocket.close", "code": config.websocket_close_code})
            return

        body = encode_json(
            {
                "status_code": config.rejection_status_code,
                "detail": config.rejection_detail,
            }
        )
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ]
        if config.retry_after is not None:
            headers.append((b"retry-after", str(config.retry_after).encode()))
        if config.connection_close:
            headers.append((b"connection", b"close"))
        await send(
            {
                "type": "http.response.start",
                "status": config.rejection_status_code,
                "headers": headers,
            }
        )
        await send({"type": "http.response.body", "body": body, "more_body": False})

    # ------------------------------------------------------------------
    # drain trigger / state machine
    # ------------------------------------------------------------------

    async def begin_drain(self) -> DrainResult:
        """Transition the application to the draining state and wait for the drain to
        complete.

        Idempotent: the first call performs the transition and runs the drain; any
        subsequent call - including concurrent ones - awaits the same outcome and
        receives the exact same :class:`DrainResult`.

        Raises:
            RuntimeError: If the drain capability is not enabled.
        """
        if not self.enabled:
            raise RuntimeError(
                "shutdown drain is not enabled. Pass a 'DrainConfig' instance via the "
                "'drain_config' parameter of 'Litestar' to enable it."
            )

        loop = asyncio.get_running_loop()
        async with self._lock:
            if self.state is DrainState.DRAINED:
                return cast("DrainResult", self._result)
            if self._drain_future is None:
                future: asyncio.Future[DrainResult] = loop.create_future()
                self._drain_future = future
                self.state = DrainState.DRAINING
                self._drain_started_at = time.monotonic()
                self._exempt_trigger_request_locked()
                run_drain = True
            else:
                future = self._drain_future
                run_drain = False

        if not run_drain:
            return await future

        try:
            result = await self._run_drain()
        except BaseException as exc:  # propagate to every concurrent waiter
            if not future.done():
                future.set_exception(exc)
            raise
        else:
            if not future.done():
                future.set_result(result)
            return result

    def _exempt_trigger_request_locked(self) -> None:
        """Exempt the request that triggered the drain from being awaited / aborted.

        Without this, ``await app.begin_drain()`` called from within a route handler
        (e.g. an administrative endpoint) would wait for itself.
        """
        trigger_task = asyncio.current_task()
        if trigger_task is None:
            return
        for request in self._inflight:
            if request.task is trigger_task:
                request.await_completion = False
                request.abortable = False
                logger.info(
                    "drain triggered from in-flight request %s %s, excluding it from the drain",
                    request.method,
                    request.path,
                )
                break
        if not any(request.abortable for request in self._inflight):
            self._idle.set()

    async def _run_drain(self) -> DrainResult:
        """Wait for in-flight requests to finish within the deadline, abort stragglers
        and finalize the state.
        """
        config = cast("DrainConfig", self.config)
        unfinished: tuple[UnfinishedRequest, ...] = ()
        timed_out = False
        try:
            await asyncio.wait_for(self._idle.wait(), timeout=config.grace_period)
        except TimeoutError:
            timed_out = True
            unfinished = await self._abort_inflight()
        return await self._finalize(timed_out=timed_out, unfinished=unfinished)

    async def _abort_inflight(self) -> tuple[UnfinishedRequest, ...]:
        """Cancel all abortable in-flight requests and return descriptions of them."""
        async with self._lock:
            records = [request for request in self._inflight if request.abortable]
            now = time.monotonic()
            unfinished = tuple(
                UnfinishedRequest(
                    method=request.method,
                    path=request.path,
                    elapsed=now - request.started_at,
                )
                for request in records
            )
            tasks = [request.task for request in records if request.task is not None]

        for request in unfinished:
            logger.warning(
                "drain deadline elapsed, aborting in-flight request %s %s after %.3fs",
                request.method,
                request.path,
                request.elapsed,
            )

        for task in tasks:
            task.cancel()

        if tasks:
            gather_task = asyncio.gather(*tasks, return_exceptions=True)
            try:
                await asyncio.wait_for(
                    asyncio.shield(gather_task),
                    timeout=cast("DrainConfig", self.config).abort_timeout,
                )
            except TimeoutError:
                # Tasks ignoring cancellation cannot be force-stopped (e.g. code
                # blocked in a worker thread). The shutdown sequence must still be
                # deterministic, so we proceed and leave the task GC/loop to reap them.
                logger.error(
                    "%d request(s) did not stop within %ss after cancellation; "
                    "proceeding with shutdown",
                    len(tasks),
                    cast("DrainConfig", self.config).abort_timeout,
                )
                gather_task.add_done_callback(lambda _: None)

        return unfinished

    async def _finalize(
        self,
        *,
        timed_out: bool,
        unfinished: tuple[UnfinishedRequest, ...],
    ) -> DrainResult:
        async with self._lock:
            started_at = self._drain_started_at or time.monotonic()
            in_flight = [request for request in self._inflight if request.abortable]
            result = DrainResult(
                state=DrainState.DRAINED,
                timed_out=timed_out,
                elapsed=time.monotonic() - started_at,
                unfinished=unfinished,
                admitted_requests=self.admitted_requests,
                rejected_requests=self.rejected_requests,
                completed_requests=self.completed_requests,
                in_flight_requests=len(in_flight),
            )
            self.state = DrainState.DRAINED
            self._result = result

        if timed_out:
            logger.warning(
                "drain completed after the deadline with %d aborted request(s)", len(unfinished)
            )
        else:
            logger.info("drain completed, all in-flight requests finished")
        return result

    # ------------------------------------------------------------------
    # operating-system signal trigger
    # ------------------------------------------------------------------

    def install_signal_handlers(self, loop: asyncio.AbstractEventLoop) -> None:
        """Register the configured POSIX signal handlers on ``loop``.

        On platforms without loop signal support (e.g. Windows) a warning is logged
        and the option is ignored.
        """
        config = cast("DrainConfig", self.config)
        signals = config.signals or ()
        if not signals:
            return
        add_handler = getattr(loop, "add_signal_handler", None)
        if add_handler is None:
            logger.warning(
                "signal handlers are not supported on this platform; "
                "drain signals %s are not registered",
                tuple(int(sig) for sig in signals),
            )
            return
        for sig in signals:
            signum = int(sig)
            add_handler(signum, self._on_signal, loop, sig)
            self._signal_handlers[signum] = loop
            logger.info("drain will be triggered by signal %s", sig)

    def remove_signal_handlers(self) -> None:
        """Remove all signal handlers previously installed by
        :meth:`install_signal_handlers`.
        """
        for signum, loop in self._signal_handlers.items():
            with contextlib.suppress(NotImplementedError, RuntimeError, OSError):
                loop.remove_signal_handler(signum)
        self._signal_handlers.clear()

    def _on_signal(self, loop: asyncio.AbstractEventLoop, sig: int) -> None:
        logger.warning("received signal %s, beginning drain", sig)
        task = loop.create_task(self.begin_drain())
        task.add_done_callback(_log_drain_trigger_exception)


def _log_drain_trigger_exception(task: asyncio.Task[DrainResult]) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("drain triggered by signal failed", exc_info=exc)
