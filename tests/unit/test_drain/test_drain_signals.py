from __future__ import annotations

import asyncio
import os
import signal
import sys

import pytest

from litestar import Litestar
from litestar.config.drain import DrainConfig
from litestar.enums import DrainState
from tests.unit.test_drain.conftest import LifespanHarness, wait_until

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or not hasattr(signal, "SIGUSR1"),
    reason="POSIX signals are required",
)


async def test_posix_signal_triggers_drain() -> None:
    loop = asyncio.get_running_loop()
    if not hasattr(loop, "add_signal_handler"):
        pytest.skip("the running event loop does not support signal handlers")

    app = Litestar(drain_config=DrainConfig(grace_period=10, signals=(signal.SIGUSR1,)))

    lifespan = LifespanHarness(app)
    await lifespan.start()
    assert app.drain_state is DrainState.RUNNING

    os.kill(os.getpid(), signal.SIGUSR1)

    await wait_until(lambda: app.drain_state is DrainState.DRAINING)
    result = await app.begin_drain()

    assert result.timed_out is False
    assert app.drain_state is DrainState.DRAINED

    # a second signal after completion must not raise or trigger another drain
    os.kill(os.getpid(), signal.SIGUSR1)
    await wait_until(lambda: True)

    message = await lifespan.shutdown()
    assert message["type"] == "lifespan.shutdown.complete"
    assert app.drain.result is result
