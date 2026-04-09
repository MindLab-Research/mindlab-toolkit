from __future__ import annotations

import asyncio
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import mint  # noqa: E402
from mint import mint as mint_impl  # noqa: E402
from mint.mint import _mintx  # noqa: E402
import tinker  # noqa: E402


class _FakeAPIFuture:
    def __init__(self, model_cls, holder, future, **kwargs):
        self.model_cls = model_cls
        self.holder = holder
        self.future = future
        self.kwargs = kwargs


class _FakeClient:
    def __init__(self, calls: list[dict[str, Any]]) -> None:
        self.calls = calls

    async def post(self, path: str, *, body: dict[str, Any], cast_to: Any) -> Any:
        self.calls.append({"path": path, "body": body, "cast_to": cast_to})
        return tinker.types.UntypedAPIFuture(request_id="req-123")


class _FakeHolder:
    def __init__(self, calls: list[dict[str, Any]]) -> None:
        self.calls = calls
        self.run_coroutine_threadsafe = lambda coro: SimpleNamespace(result=lambda: asyncio.run(coro))

    @contextmanager
    def aclient(self, _pool_type):
        yield _FakeClient(self.calls)

    async def execute_with_retries(self, fn):
        return await fn()


def test_mint_namespace_exports_mintx_symbols() -> None:
    assert hasattr(mint_impl, "ReverseKLDatum")
    assert hasattr(mint_impl, "interpolate_checkpoints")
    assert hasattr(mint_impl, "forward_backward_reverse_kl")
    assert "ReverseKLDatum" in mint_impl.__all__
    assert "interpolate_checkpoints" in mint_impl.__all__
    assert "forward_backward_reverse_kl" in mint_impl.__all__


def test_interpolate_checkpoints_async_posts_to_mint_endpoint(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []
    holder = _FakeHolder(calls)
    service_client = SimpleNamespace(holder=holder)
    monkeypatch.setattr(_mintx, "_APIFuture", _FakeAPIFuture)

    future = asyncio.run(
        _mintx.interpolate_checkpoints_async(
            service_client,
            source_paths=["mint://teacher", "mint://student"],
            coefficients=[0.9, 0.1],
            output_path="ema-step-0010",
        )
    )

    assert isinstance(future, _FakeAPIFuture)
    assert future.model_cls is _mintx.InterpolateCheckpointsResponse
    assert calls == [
        {
            "path": "/api/v1/mint/checkpoints/interpolate",
            "body": {
                "source_paths": ["mint://teacher", "mint://student"],
                "coefficients": [0.9, 0.1],
                "output_path": "ema-step-0010",
                "output_checkpoint_type": "sampler",
                "type": "mint_interpolate_checkpoints",
            },
            "cast_to": tinker.types.UntypedAPIFuture,
        }
    ]


def test_forward_backward_reverse_kl_async_posts_to_mint_endpoint(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []
    holder = _FakeHolder(calls)
    training_client = SimpleNamespace(
        holder=holder,
        model_id="model-123",
        _queue_state_logger=object(),
    )
    monkeypatch.setattr(_mintx, "_APIFuture", _FakeAPIFuture)

    datum = mint_impl.ReverseKLDatum(
        student_input=tinker.types.ModelInput.from_ints([1, 2, 3]),
        reference_input=tinker.types.ModelInput.from_ints([4, 5, 6]),
        target_tokens=tinker.types.TensorData(data=[7, 8], shape=[2], dtype="int64"),
        weights=tinker.types.TensorData(data=[1.0, 1.0], shape=[2], dtype="float32"),
    )
    future = asyncio.run(
        _mintx.forward_backward_reverse_kl_async(
            training_client,
            reference_model_path="mint://teacher-step-0010",
            data=[datum],
            temperature=1.5,
            seq_id=11,
        )
    )

    assert isinstance(future, _FakeAPIFuture)
    assert future.model_cls is _mintx.ForwardBackwardReverseKLResponse
    assert future.kwargs["queue_state_observer"] is training_client._queue_state_logger
    assert calls == [
        {
            "path": "/api/v1/mint/forward_backward_reverse_kl",
            "body": {
                "model_id": "model-123",
                "reference_model_path": "mint://teacher-step-0010",
                "data": [
                    {
                        "student_input": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]},
                        "reference_input": {"chunks": [{"type": "encoded_text", "tokens": [4, 5, 6]}]},
                        "target_tokens": {"data": [7, 8], "shape": [2], "dtype": "int64"},
                        "weights": {"data": [1.0, 1.0], "shape": [2], "dtype": "float32"},
                    }
                ],
                "temperature": 1.5,
                "seq_id": 11,
                "type": "mint_forward_backward_reverse_kl",
            },
            "cast_to": tinker.types.UntypedAPIFuture,
        }
    ]
