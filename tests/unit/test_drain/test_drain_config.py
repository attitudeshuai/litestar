from __future__ import annotations

from datetime import timedelta

import pytest

from litestar.config.drain import DrainConfig
from litestar.exceptions import ImproperlyConfiguredException
from litestar.status_codes import HTTP_503_SERVICE_UNAVAILABLE, WS_1001_GOING_AWAY


def test_drain_config_defaults() -> None:
    config = DrainConfig()

    assert config.grace_period == 30.0
    assert config.probe_paths == ()
    assert config.rejection_status_code == HTTP_503_SERVICE_UNAVAILABLE
    assert config.rejection_detail == "Service is shutting down"
    assert config.retry_after == 30
    assert config.connection_close is True
    assert config.reject_websockets is True
    assert config.websocket_close_code == WS_1001_GOING_AWAY
    assert config.abort_timeout == 5.0
    assert config.signals is None


def test_drain_config_normalizes_timedeltas() -> None:
    config = DrainConfig(
        grace_period=timedelta(seconds=2),
        abort_timeout=timedelta(milliseconds=500),
        retry_after=timedelta(seconds=1.5),
    )

    assert config.grace_period == 2.0
    assert config.abort_timeout == 0.5
    assert config.retry_after == 2  # rounded up, never understating the wait


def test_drain_config_retry_after_fractional_rounds_up() -> None:
    assert DrainConfig(retry_after=0.01).retry_after == 1
    assert DrainConfig(retry_after=1.2).retry_after == 2
    assert DrainConfig(retry_after=0).retry_after == 0
    assert DrainConfig(retry_after=None).retry_after is None


def test_drain_config_normalizes_and_deduplicates_probe_paths() -> None:
    config = DrainConfig(probe_paths=("health/", "/readiness", "/health", "liveness//"))

    assert config.probe_paths == ("/health", "/readiness", "/liveness")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"grace_period": -1}, "grace_period"),
        ({"abort_timeout": -0.1}, "abort_timeout"),
        ({"retry_after": -3}, "retry_after"),
        ({"rejection_status_code": 200}, "rejection_status_code"),
        ({"rejection_status_code": 399}, "rejection_status_code"),
        ({"rejection_detail": ""}, "rejection_detail"),
        ({"signals": ()}, "signals"),
    ],
)
def test_drain_config_rejects_invalid_values(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ImproperlyConfiguredException, match=match):
        DrainConfig(**kwargs)  # type: ignore[arg-type]
