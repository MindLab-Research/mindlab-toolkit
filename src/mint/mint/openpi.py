"""MinT-only OpenPI / VLA helpers exposed through ``mint.mint`` / ``mintx``."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import tinker
from pydantic import BaseModel
from tinker._base_client import make_request_options
from tinker._compat import model_dump
from tinker.lib.api_future_impl import _APIFuture, _CombinedAPIFuture
from tinker.lib.chunked_fwdbwd_helpers import combine_fwd_bwd_output_results
from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
from tinker.lib.public_interfaces.api_future import APIFuture
from tinker.lib.public_interfaces.service_client import ServiceClient
from tinker.lib.public_interfaces.training_client import TrainingClient
from tinker.types.shared.untyped_api_future import UntypedAPIFuture

OPENPI_FAST_MODEL = "openpi/pi0-fast-libero-low-mem-finetune"
OPENPI_FAST_LORA_RANK = 16
CAMERA_LAYOUT = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


class OpenPITrainStepRequest(BaseModel):
    forward_backward_input: tinker.types.ForwardBackwardInput
    adam_params: tinker.types.AdamParams | None = None
    model_id: str
    seq_id: int | None = None
    type: Literal["train_step"] = "train_step"


OpenPITrainStepRequest.model_rebuild()


class OpenPITrainingClient:
    """MinT-only wrapper around a standard TrainingClient for OpenPI VLA flows.

    The wrapped ``TrainingClient`` stays a normal Tinker-compatible object. This
    wrapper adds only the OpenPI-specific ``train_step`` surface and leaves the
    rest of the public methods delegated to the underlying client.
    """

    def __init__(self, training_client: TrainingClient):
        self._training_client = training_client

    @property
    def base_training_client(self) -> TrainingClient:
        return self._training_client

    @property
    def model_id(self) -> tinker.types.ModelID:
        return self._training_client.model_id

    def __getattr__(self, name: str) -> Any:
        return getattr(self._training_client, name)

    async def _send_train_step_request(
        self,
        request_id: int,
        data: list[tinker.types.Datum],
        loss_fn: tinker.types.LossFnType,
        loss_fn_config: dict[str, float] | None = None,
        adam_params: tinker.types.AdamParams | None = None,
    ) -> UntypedAPIFuture:
        request = OpenPITrainStepRequest(
            forward_backward_input=tinker.types.ForwardBackwardInput(
                data=data,
                loss_fn=loss_fn,
                loss_fn_config=loss_fn_config,
            ),
            adam_params=adam_params or tinker.types.AdamParams(),
            model_id=self._training_client._guaranteed_model_id(),
            seq_id=request_id + 1,
        )
        options = make_request_options()
        with self._training_client.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
            return await client.training._post(
                "/api/v1/train_step",
                body=model_dump(request, mode="json"),
                options=options,
                cast_to=UntypedAPIFuture,
            )

    def train_step(
        self,
        data: list[tinker.types.Datum],
        loss_fn: tinker.types.LossFnType,
        loss_fn_config: dict[str, float] | None = None,
        adam_params: tinker.types.AdamParams | None = None,
    ) -> APIFuture[tinker.types.ForwardBackwardOutput]:
        requests = self._training_client._chunked_requests(data)

        async def _train_step_async():
            futures = []
            start_time = time.time()

            for request_id, chunk in requests:
                async with self._training_client._take_turn(request_id):
                    untyped_future = await self._training_client.holder.execute_with_retries(
                        self._send_train_step_request,
                        request_id,
                        chunk,
                        loss_fn,
                        loss_fn_config,
                        adam_params,
                    )
                api_future = _APIFuture(
                    tinker.types.ForwardBackwardOutput,
                    self._training_client.holder,
                    untyped_future,
                    request_start_time=start_time,
                    request_type="TrainStep",
                    queue_state_observer=self._training_client._queue_state_logger,
                )
                futures.append(api_future)

            return await _CombinedAPIFuture(
                futures,
                combine_fwd_bwd_output_results,
                self._training_client.holder,
            )

        return self._training_client.holder.run_coroutine_threadsafe(_train_step_async())

    async def train_step_async(
        self,
        data: list[tinker.types.Datum],
        loss_fn: tinker.types.LossFnType,
        loss_fn_config: dict[str, float] | None = None,
        adam_params: tinker.types.AdamParams | None = None,
    ) -> APIFuture[tinker.types.ForwardBackwardOutput]:
        return self.train_step(
            data,
            loss_fn,
            loss_fn_config=loss_fn_config,
            adam_params=adam_params,
        )


def _tensor(
    data: Sequence[int] | Sequence[float], *, dtype: Literal["int64", "float32"]
) -> tinker.types.TensorData:
    return tinker.types.TensorData(data=list(data), dtype=dtype, shape=[len(data)])


def build_openpi_fast_datum(
    *,
    prefix_tokens: Sequence[int],
    image_bytes_by_camera: Mapping[str, bytes],
    state: Sequence[float],
    target_tokens: Sequence[int],
    weights: Sequence[float],
    token_ar_mask: Sequence[int],
    logprobs: Sequence[float] | None = None,
    advantages: Sequence[float] | None = None,
) -> tinker.types.Datum:
    if set(image_bytes_by_camera) != set(CAMERA_LAYOUT):
        raise ValueError(f"image_bytes_by_camera must contain exactly {CAMERA_LAYOUT}")
    if len(target_tokens) != len(weights) or len(target_tokens) != len(token_ar_mask):
        raise ValueError("target_tokens, weights, and token_ar_mask must share one length")
    if (logprobs is None) != (advantages is None):
        raise ValueError("logprobs and advantages must be provided together")
    if logprobs is not None and len(logprobs) != len(target_tokens):
        raise ValueError("logprobs must share one length with target_tokens")
    if advantages is not None and len(advantages) != len(target_tokens):
        raise ValueError("advantages must share one length with target_tokens")

    loss_fn_inputs: dict[str, Any] = {
        "state": _tensor(state, dtype="float32"),
        "target_tokens": _tensor(target_tokens, dtype="int64"),
        "weights": _tensor(weights, dtype="float32"),
        "token_ar_mask": _tensor(token_ar_mask, dtype="int64"),
    }
    if logprobs is not None and advantages is not None:
        loss_fn_inputs["logprobs"] = _tensor(logprobs, dtype="float32")
        loss_fn_inputs["advantages"] = _tensor(advantages, dtype="float32")

    return tinker.types.Datum(
        model_input=tinker.types.ModelInput(
            chunks=[
                *[
                    tinker.types.ImageChunk(
                        data=image_bytes_by_camera[camera_name],
                        format="png",
                        expected_tokens=256,
                    )
                    for camera_name in CAMERA_LAYOUT
                ],
                tinker.types.EncodedTextChunk(tokens=list(prefix_tokens)),
            ]
        ),
        loss_fn_inputs=loss_fn_inputs,
    )


def create_openpi_training_client(
    service_client: ServiceClient,
    *,
    base_model: str = OPENPI_FAST_MODEL,
    rank: int = OPENPI_FAST_LORA_RANK,
    create_timeout_seconds: float | None = None,
    user_metadata: dict[str, str] | None = None,
) -> OpenPITrainingClient:
    if base_model != OPENPI_FAST_MODEL:
        raise ValueError(f"OpenPI helper is currently pinned to {OPENPI_FAST_MODEL}")
    if rank != OPENPI_FAST_LORA_RANK:
        raise ValueError(f"OpenPI FAST create_model currently requires rank={OPENPI_FAST_LORA_RANK}")
    if not hasattr(service_client, "_create_lora_training_client_submit"):
        raise RuntimeError("ServiceClient does not expose _create_lora_training_client_submit")

    future = service_client._create_lora_training_client_submit(
        base_model=base_model,
        rank=rank,
        seed=None,
        train_attn=True,
        train_mlp=True,
        train_unembed=True,
        user_metadata=user_metadata,
    )
    if create_timeout_seconds is None:
        training_client = future.result()
    else:
        training_client = future.result(timeout=create_timeout_seconds)
    return OpenPITrainingClient(training_client)


async def create_openpi_training_client_async(
    service_client: ServiceClient,
    *,
    base_model: str = OPENPI_FAST_MODEL,
    rank: int = OPENPI_FAST_LORA_RANK,
    create_timeout_seconds: float | None = None,
    user_metadata: dict[str, str] | None = None,
) -> OpenPITrainingClient:
    if base_model != OPENPI_FAST_MODEL:
        raise ValueError(f"OpenPI helper is currently pinned to {OPENPI_FAST_MODEL}")
    if rank != OPENPI_FAST_LORA_RANK:
        raise ValueError(f"OpenPI FAST create_model currently requires rank={OPENPI_FAST_LORA_RANK}")
    if not hasattr(service_client, "_create_lora_training_client_submit"):
        raise RuntimeError("ServiceClient does not expose _create_lora_training_client_submit")

    future = service_client._create_lora_training_client_submit(
        base_model=base_model,
        rank=rank,
        seed=None,
        train_attn=True,
        train_mlp=True,
        train_unembed=True,
        user_metadata=user_metadata,
    )
    if create_timeout_seconds is None:
        training_client = await future.result_async()
    else:
        training_client = await asyncio.wait_for(future.result_async(), timeout=create_timeout_seconds)
    return OpenPITrainingClient(training_client)


__all__ = [
    "CAMERA_LAYOUT",
    "OPENPI_FAST_LORA_RANK",
    "OPENPI_FAST_MODEL",
    "OpenPITrainingClient",
    "build_openpi_fast_datum",
    "create_openpi_training_client",
    "create_openpi_training_client_async",
]
