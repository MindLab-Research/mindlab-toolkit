from __future__ import annotations

import asyncio
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import mint  # noqa: E402
from mint import mint as mintx  # noqa: E402


def test_openpi_helpers_live_only_in_mint_namespace() -> None:
    assert hasattr(mintx, "create_openpi_training_client")
    assert hasattr(mintx, "create_openpi_training_client_async")
    assert hasattr(mintx, "build_openpi_fast_datum")
    assert not hasattr(mint, "create_openpi_training_client")
    assert not hasattr(mint.TrainingClient, "train_step")


def test_build_openpi_fast_datum_keeps_camera_order_and_shapes() -> None:
    datum = mintx.build_openpi_fast_datum(
        prefix_tokens=[11, 12, 13],
        image_bytes_by_camera={
            "base_0_rgb": b"base-image",
            "left_wrist_0_rgb": b"left-image",
            "right_wrist_0_rgb": b"right-image",
        },
        state=[0.1] * 7,
        target_tokens=[21, 22],
        weights=[1.0, 1.0],
        token_ar_mask=[1, 1],
    )

    chunks = datum.model_input.chunks
    assert [type(chunk).__name__ for chunk in chunks] == ["ImageChunk", "ImageChunk", "ImageChunk", "EncodedTextChunk"]
    assert [chunk.data for chunk in chunks[:3]] == [b"base-image", b"left-image", b"right-image"]
    assert chunks[-1].tokens == [11, 12, 13]
    assert datum.loss_fn_inputs["state"].shape == [7]
    assert datum.loss_fn_inputs["state"].dtype == "float32"
    assert datum.loss_fn_inputs["token_ar_mask"].dtype == "int64"


def test_build_openpi_fast_datum_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="share one length"):
        mintx.build_openpi_fast_datum(
            prefix_tokens=[11, 12],
            image_bytes_by_camera={camera_name: b"img" for camera_name in mintx.CAMERA_LAYOUT},
            state=[0.1] * 7,
            target_tokens=[21, 22],
            weights=[1.0],
            token_ar_mask=[1, 1],
        )


def test_create_openpi_training_client_wraps_private_submit_helper() -> None:
    captured: dict[str, Any] = {}

    class _FakeFuture:
        def result(self, timeout: float | None = None):
            captured["timeout"] = timeout
            return SimpleNamespace(model_id="model-123")

    class _FakeServiceClient:
        def _create_lora_training_client_submit(self, **kwargs):
            captured["kwargs"] = kwargs
            return _FakeFuture()

    client = mintx.create_openpi_training_client(
        _FakeServiceClient(),
        create_timeout_seconds=9.5,
        user_metadata={"example": "openpi-test"},
    )

    assert isinstance(client, mintx.OpenPITrainingClient)
    assert client.model_id == "model-123"
    assert captured["timeout"] == 9.5
    assert captured["kwargs"] == {
        "base_model": mintx.OPENPI_FAST_MODEL,
        "rank": mintx.OPENPI_FAST_LORA_RANK,
        "seed": None,
        "train_attn": True,
        "train_mlp": True,
        "train_unembed": True,
        "user_metadata": {"example": "openpi-test"},
    }


def test_create_openpi_training_client_rejects_non_openpi_config() -> None:
    class _FakeServiceClient:
        pass

    with pytest.raises(ValueError, match="pinned"):
        mintx.create_openpi_training_client(_FakeServiceClient(), base_model="other/model")

    with pytest.raises(ValueError, match="requires rank"):
        mintx.create_openpi_training_client(_FakeServiceClient(), rank=8)


def test_openpi_training_client_posts_train_step_without_global_patch() -> None:
    captured: dict[str, Any] = {}

    async def fake_post(path: str, *, body: dict[str, Any], options: dict[str, Any], cast_to: object):
        captured["path"] = path
        captured["body"] = body
        captured["options"] = options
        captured["cast_to"] = cast_to
        return "future-object"

    class _FakeTrainingClient:
        def __init__(self) -> None:
            self.model_id = "model-123"
            self._queue_state_logger = object()
            self.holder = self

        def _guaranteed_model_id(self) -> str:
            return self.model_id

        @contextmanager
        def aclient(self, _pool_type):
            yield SimpleNamespace(training=SimpleNamespace(_post=fake_post))

    wrapper = mintx.OpenPITrainingClient(_FakeTrainingClient())
    datum = mintx.build_openpi_fast_datum(
        prefix_tokens=[11, 12, 13],
        image_bytes_by_camera={camera_name: b"img" for camera_name in mintx.CAMERA_LAYOUT},
        state=[0.1] * 7,
        target_tokens=[21, 22],
        weights=[1.0, 1.0],
        token_ar_mask=[1, 1],
    )

    result = asyncio.run(
        wrapper._send_train_step_request(
            0,
            [datum],
            "cross_entropy",
            adam_params=mint.types.AdamParams(learning_rate=0.003),
        )
    )

    assert result == "future-object"
    assert captured["path"] == "/api/v1/train_step"
    assert captured["body"]["type"] == "train_step"
    assert captured["body"]["model_id"] == "model-123"
    assert captured["body"]["seq_id"] == 1
    assert captured["body"]["forward_backward_input"]["loss_fn"] == "cross_entropy"
    assert captured["body"]["forward_backward_input"]["data"][0]["loss_fn_inputs"]["token_ar_mask"]["dtype"] == "int64"
    assert captured["options"] == {}
