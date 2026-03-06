"""Namespace with explicit Tinker-compatible MinT exports."""

from __future__ import annotations

import tinker as _tinker

TINKER_COMPAT_EXPORTS = [
    "TrainingClient",
    "ServiceClient",
    "SamplingClient",
    "APIFuture",
    "AdamParams",
    "Checkpoint",
    "CheckpointType",
    "Datum",
    "EncodedTextChunk",
    "ForwardBackwardOutput",
    "LoraConfig",
    "ModelID",
    "ModelInput",
    "ModelInputChunk",
    "OptimStepRequest",
    "OptimStepResponse",
    "ParsedCheckpointTinkerPath",
    "SampledSequence",
    "SampleRequest",
    "SampleResponse",
    "SamplingParams",
    "StopReason",
    "TensorData",
    "TensorDtype",
    "TrainingRun",
    "Timeout",
    "RequestOptions",
    "TinkerError",
    "APIError",
    "APIStatusError",
    "APITimeoutError",
    "APIConnectionError",
    "APIResponseValidationError",
    "RequestFailedError",
    "BadRequestError",
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "ConflictError",
    "UnprocessableEntityError",
    "RateLimitError",
    "InternalServerError",
    "types",
    "__version__",
    "__title__",
]

for _symbol in TINKER_COMPAT_EXPORTS:
    if not hasattr(_tinker, _symbol):
        raise RuntimeError(
            f"tinker is missing expected symbol {_symbol!r}; "
            "mindlab-toolkit requires tinker symbols from 0.6.0"
        )
    globals()[_symbol] = getattr(_tinker, _symbol)

__all__ = [*TINKER_COMPAT_EXPORTS]
