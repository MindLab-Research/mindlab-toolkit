from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

import tinker
from tinker._compat import model_dump
from tinker.lib.api_future_impl import _APIFuture
from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
from tinker.lib.public_interfaces.api_future import APIFuture
from tinker.lib.telemetry import capture_exceptions


class MintXBaseModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class ReverseKLDatum(MintXBaseModel):
    student_input: tinker.types.ModelInput
    reference_input: tinker.types.ModelInput
    target_tokens: tinker.types.TensorData
    weights: tinker.types.TensorData


class InterpolateCheckpointsRequest(MintXBaseModel):
    source_paths: list[str]
    coefficients: list[float]
    output_path: str | None = None
    output_checkpoint_type: Literal["sampler"] = "sampler"
    type: Literal["mint_interpolate_checkpoints"] = "mint_interpolate_checkpoints"


class InterpolateCheckpointsResponse(MintXBaseModel):
    path: str
    checkpoint_type: Literal["sampler"] = "sampler"
    source_paths: list[str]
    coefficients: list[float]
    has_rank_shards: bool = False
    type: Literal["mint_interpolate_checkpoints"] = "mint_interpolate_checkpoints"


class ForwardBackwardReverseKLRequest(MintXBaseModel):
    model_id: str
    reference_model_path: str
    data: list[ReverseKLDatum]
    temperature: float = 1.0
    seq_id: int | None = None
    type: Literal["mint_forward_backward_reverse_kl"] = "mint_forward_backward_reverse_kl"


class ReverseKLItemOutput(MintXBaseModel):
    loss: tinker.types.TensorData


class ForwardBackwardReverseKLResponse(MintXBaseModel):
    outputs: list[ReverseKLItemOutput]
    metrics: dict[str, float]
    type: Literal["mint_forward_backward_reverse_kl"] = "mint_forward_backward_reverse_kl"


def _coerce_model(model_cls, value):
    if isinstance(value, model_cls):
        return value
    if isinstance(value, dict):
        return model_cls.model_validate(value)
    return model_cls.model_validate(value)


@capture_exceptions(fatal=True)
def interpolate_checkpoints(
    service_client: tinker.ServiceClient,
    *,
    source_paths: list[str],
    coefficients: list[float],
    output_path: str | None = None,
    output_checkpoint_type: Literal["sampler"] = "sampler",
) -> InterpolateCheckpointsResponse:
    value = service_client.holder.run_coroutine_threadsafe(
        interpolate_checkpoints_async(
            service_client,
            source_paths=source_paths,
            coefficients=coefficients,
            output_path=output_path,
            output_checkpoint_type=output_checkpoint_type,
        )
    ).result().result()
    return _coerce_model(InterpolateCheckpointsResponse, value)


@capture_exceptions(fatal=True)
async def interpolate_checkpoints_async(
    service_client: tinker.ServiceClient,
    *,
    source_paths: list[str],
    coefficients: list[float],
    output_path: str | None = None,
    output_checkpoint_type: Literal["sampler"] = "sampler",
) -> APIFuture[InterpolateCheckpointsResponse]:
    request = InterpolateCheckpointsRequest(
        source_paths=source_paths,
        coefficients=coefficients,
        output_path=output_path,
        output_checkpoint_type=output_checkpoint_type,
    )

    async def _send_request() -> tinker.types.UntypedAPIFuture:
        with service_client.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
            return await client.post(
                "/api/v1/mint/checkpoints/interpolate",
                body=model_dump(request, mode="json"),
                cast_to=tinker.types.UntypedAPIFuture,
            )

    start_time = time.time()
    future = await service_client.holder.execute_with_retries(_send_request)
    return _APIFuture(
        InterpolateCheckpointsResponse,
        service_client.holder,
        future,
        request_start_time=start_time,
        request_type="MintInterpolateCheckpoints",
    )


@capture_exceptions(fatal=True)
def forward_backward_reverse_kl(
    training_client: tinker.TrainingClient,
    *,
    reference_model_path: str,
    data: list[ReverseKLDatum],
    temperature: float = 1.0,
    seq_id: int | None = None,
) -> ForwardBackwardReverseKLResponse:
    value = training_client.holder.run_coroutine_threadsafe(
        forward_backward_reverse_kl_async(
            training_client,
            reference_model_path=reference_model_path,
            data=data,
            temperature=temperature,
            seq_id=seq_id,
        )
    ).result().result()
    return _coerce_model(ForwardBackwardReverseKLResponse, value)


@capture_exceptions(fatal=True)
async def forward_backward_reverse_kl_async(
    training_client: tinker.TrainingClient,
    *,
    reference_model_path: str,
    data: list[ReverseKLDatum],
    temperature: float = 1.0,
    seq_id: int | None = None,
) -> APIFuture[ForwardBackwardReverseKLResponse]:
    request = ForwardBackwardReverseKLRequest(
        model_id=str(training_client.model_id),
        reference_model_path=reference_model_path,
        data=data,
        temperature=temperature,
        seq_id=seq_id,
    )

    async def _send_request() -> tinker.types.UntypedAPIFuture:
        with training_client.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
            return await client.post(
                "/api/v1/mint/forward_backward_reverse_kl",
                body=model_dump(request, mode="json"),
                cast_to=tinker.types.UntypedAPIFuture,
            )

    start_time = time.time()
    future = await training_client.holder.execute_with_retries(_send_request)
    return _APIFuture(
        ForwardBackwardReverseKLResponse,
        training_client.holder,
        future,
        request_start_time=start_time,
        request_type="MintForwardBackwardReverseKL",
        queue_state_observer=getattr(training_client, "_queue_state_logger", None),
    )


__all__ = [
    "ForwardBackwardReverseKLRequest",
    "ForwardBackwardReverseKLResponse",
    "InterpolateCheckpointsRequest",
    "InterpolateCheckpointsResponse",
    "ReverseKLDatum",
    "ReverseKLItemOutput",
    "forward_backward_reverse_kl",
    "forward_backward_reverse_kl_async",
    "interpolate_checkpoints",
    "interpolate_checkpoints_async",
]
