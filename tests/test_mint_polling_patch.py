from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import mint  # noqa: E402
from mint import mint as mint_impl  # noqa: E402

import tinker  # noqa: E402
import tinker.lib.telemetry as telemetry_module  # noqa: E402
import tinker.resources.futures as futures_module  # noqa: E402


EVENT_NAME = "APIFuture.result_async.api_status_error"


async def _raise_status_error(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("test stub should be replaced")


def _make_status_error(
    status_code: int,
    *,
    payload: object | None = None,
    json_error: Exception | None = None,
) -> tinker.APIStatusError:
    request = SimpleNamespace(method="POST", url="https://example.com/api/v1/retrieve_future")

    if json_error is None:
        response = SimpleNamespace(
            request=request,
            status_code=status_code,
            json=lambda: {} if payload is None else payload,
        )
    else:
        def _raise_json_error() -> object:
            raise json_error

        response = SimpleNamespace(
            request=request,
            status_code=status_code,
            json=_raise_json_error,
        )

    return tinker.APIStatusError(
        f"status {status_code}",
        response=response,
        body=payload,
    )


async def _run_retrieve_once() -> None:
    await futures_module.AsyncFuturesResource.retrieve(object(), request=object())


@pytest.mark.parametrize(
    ("payload", "expected_delay"),
    [
        ({"queue_state": "active"}, 0.3),
        ({"queue_state": "paused_capacity"}, 1.0),
        ({"queue_state": "paused_rate_limit"}, 2.0),
        ({}, 1.0),
        ({"queue_state": "unexpected"}, 1.0),
    ],
)
def test_retrieve_future_polling_uses_graded_delays(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, str],
    expected_delay: float,
) -> None:
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    async def fake_retrieve(*args: Any, **kwargs: Any) -> Any:
        raise _make_status_error(408, payload=payload)

    monkeypatch.setattr(mint_impl._asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(futures_module.AsyncFuturesResource, "retrieve", fake_retrieve)

    mint_impl._patch_retrieve_future_polling()

    with pytest.raises(tinker.APIStatusError):
        asyncio.run(_run_retrieve_once())

    assert delays == [pytest.approx(expected_delay)]


def test_retrieve_future_polling_falls_back_when_response_json_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    async def fake_retrieve(*args: Any, **kwargs: Any) -> Any:
        raise _make_status_error(408, json_error=ValueError("bad json"))

    monkeypatch.setattr(mint_impl._asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(futures_module.AsyncFuturesResource, "retrieve", fake_retrieve)

    mint_impl._patch_retrieve_future_polling()

    with pytest.raises(tinker.APIStatusError):
        asyncio.run(_run_retrieve_once())

    assert delays == [1.0]


def test_retrieve_future_polling_respects_env_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    async def fake_retrieve(*args: Any, **kwargs: Any) -> Any:
        raise _make_status_error(408, payload={"queue_state": "paused_rate_limit"})

    monkeypatch.setenv("TINKER_RETRIEVE_POLL_PAUSED_RATE_LIMIT_MS", "2500")
    monkeypatch.setattr(mint_impl._asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(futures_module.AsyncFuturesResource, "retrieve", fake_retrieve)

    mint_impl._patch_retrieve_future_polling()

    with pytest.raises(tinker.APIStatusError):
        asyncio.run(_run_retrieve_once())

    assert delays == [2.5]


def test_retrieve_future_polling_does_not_sleep_for_non_408(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    async def fake_retrieve(*args: Any, **kwargs: Any) -> Any:
        raise _make_status_error(500)

    monkeypatch.setattr(mint_impl._asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(futures_module.AsyncFuturesResource, "retrieve", fake_retrieve)

    mint_impl._patch_retrieve_future_polling()

    with pytest.raises(tinker.APIStatusError):
        asyncio.run(_run_retrieve_once())

    assert delays == []


def test_retrieve_patch_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_retrieve(*args: Any, **kwargs: Any) -> Any:
        return {"ok": True}

    monkeypatch.setattr(futures_module.AsyncFuturesResource, "retrieve", fake_retrieve)

    mint_impl._patch_retrieve_future_polling()
    first = futures_module.AsyncFuturesResource.retrieve
    mint_impl._patch_retrieve_future_polling()
    second = futures_module.AsyncFuturesResource.retrieve

    assert first is second
    assert getattr(second, "_mint_busy_poll_patch", False) is True


def test_apply_mint_patches_is_idempotent() -> None:
    retrieve = futures_module.AsyncFuturesResource.retrieve
    telemetry_log = telemetry_module.Telemetry.log

    mint_impl.apply_mint_patches()

    assert futures_module.AsyncFuturesResource.retrieve is retrieve
    assert telemetry_module.Telemetry.log is telemetry_log


def test_telemetry_408_events_are_sampled(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, object | None, str]] = []
    timestamps = iter((100.0, 120.0, 161.0))

    def fake_log(self: object, event_name: str, event_data: object = None, severity: str = "INFO") -> bool:
        calls.append((event_name, event_data, severity))
        return True

    monkeypatch.setattr(telemetry_module.Telemetry, "log", fake_log)
    monkeypatch.setattr(mint_impl._time, "monotonic", lambda: next(timestamps))

    mint_impl._patch_telemetry_408_sampling()

    telemetry = SimpleNamespace()

    assert telemetry_module.Telemetry.log(
        telemetry,
        EVENT_NAME,
        event_data={"status_code": 408},
        severity="WARNING",
    ) is True
    assert telemetry_module.Telemetry.log(
        telemetry,
        EVENT_NAME,
        event_data={"status_code": 408},
        severity="WARNING",
    ) is False
    assert telemetry_module.Telemetry.log(
        telemetry,
        EVENT_NAME,
        event_data={"status_code": 408},
        severity="WARNING",
    ) is True
    assert calls == [
        (EVENT_NAME, {"status_code": 408}, "WARNING"),
        (EVENT_NAME, {"status_code": 408}, "WARNING"),
    ]


def test_telemetry_sampling_does_not_affect_non_408_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object | None, str]] = []

    def fake_log(self: object, event_name: str, event_data: object = None, severity: str = "INFO") -> bool:
        calls.append((event_name, event_data, severity))
        return True

    monkeypatch.setattr(telemetry_module.Telemetry, "log", fake_log)
    mint_impl._patch_telemetry_408_sampling()

    telemetry = SimpleNamespace()

    assert telemetry_module.Telemetry.log(
        telemetry,
        EVENT_NAME,
        event_data={"status_code": 500},
        severity="ERROR",
    ) is True
    assert telemetry_module.Telemetry.log(
        telemetry,
        "different.event",
        event_data={"status_code": 408},
        severity="INFO",
    ) is True
    assert calls == [
        (EVENT_NAME, {"status_code": 500}, "ERROR"),
        ("different.event", {"status_code": 408}, "INFO"),
    ]
