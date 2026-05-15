"""MinT-specific compatibility layer built on top of tinker."""

from __future__ import annotations

import asyncio as _asyncio
import inspect as _inspect
import os as _os
import time as _time

from ._mintx import (
    ForwardBackwardReverseKLRequest,
    ForwardBackwardReverseKLResponse,
    InterpolateCheckpointsRequest,
    InterpolateCheckpointsResponse,
    ReverseKLDatum,
    ReverseKLItemOutput,
    forward_backward_reverse_kl,
    forward_backward_reverse_kl_async,
    interpolate_checkpoints,
    interpolate_checkpoints_async,
)
from .openpi import (
    CAMERA_LAYOUT,
    OPENPI_FAST_LORA_RANK,
    OPENPI_FAST_MODEL,
    OpenPITrainingClient,
    build_openpi_fast_datum,
    create_openpi_training_client,
    create_openpi_training_client_async,
)

MINT_VERSION = "0.1.0"
SUPPORTED_TINKER_VERSIONS = ("0.15.0",)
EXPECTED_TINKER_VERSION = SUPPORTED_TINKER_VERSIONS[0]
SUPPORTED_TINKER_SPEC = f"=={EXPECTED_TINKER_VERSION}"
_MINT_DEFAULT_BASE_URL = "https://mint.macaron.xin"
_TINKER_API_KEY_ENV = "TINKER_API_KEY"
_TINKER_API_KEY_PREFIX = "tml-"
_MINT_API_KEY_PREFIX = "sk-"
_TINKER_COMPAT_PLACEHOLDER_API_KEY = "tml-mint-compat-placeholder"
_ALLOW_UNSUPPORTED_TINKER_ENV = "MINT_ALLOW_UNSUPPORTED_TINKER"
_PATCH_STATE = {"applied": False}
_REQUIRED_TINKER_EXPORTS = (
    "TrainingClient",
    "ServiceClient",
    "SamplingClient",
    "APIFuture",
    "types",
    "__version__",
    "__title__",
)
_PATCH_POINTS = (
    ("tinker.lib.public_interfaces.service_client", "_get_default_headers", ()),
    ("tinker.lib.public_interfaces.service_client", "ServiceClient.__init__", ("self",)),
    ("tinker._client", "AsyncTinker.__init__", ("self",)),
    ("tinker.resources.futures", "AsyncFuturesResource.retrieve", ("self", "request")),
    ("tinker.lib.telemetry", "Telemetry.log", ("self", "event_name", "event_data", "severity")),
    (
        "tinker.lib.internal_client_holder",
        "InternalClientHolder._create_sampling_session",
        ("self", "model_path", "base_model"),
    ),
)


def sync_env() -> None:
    """Synchronize MinT env vars to the names expected by tinker."""
    if "TINKER_APIKEY" in _os.environ and "TINKER_API_KEY" not in _os.environ:
        _os.environ["TINKER_API_KEY"] = _os.environ["TINKER_APIKEY"]

    if "MINT_API_KEY" in _os.environ:
        _os.environ["TINKER_API_KEY"] = _os.environ["MINT_API_KEY"]

    if "MINT_BASE_URL" in _os.environ:
        _os.environ["TINKER_BASE_URL"] = _os.environ["MINT_BASE_URL"]
    elif "MINT_API_KEY" in _os.environ:
        _os.environ["TINKER_BASE_URL"] = _MINT_DEFAULT_BASE_URL
    elif "TINKER_BASE_URL" not in _os.environ:
        _os.environ["TINKER_BASE_URL"] = _MINT_DEFAULT_BASE_URL

    # Disable telemetry by default to suppress "Telemetry queue full" warnings.
    # Users can override by explicitly setting TINKER_TELEMETRY=1.
    _os.environ.setdefault("TINKER_TELEMETRY", "0")


def _resolve_attr(target: object, dotted_name: str) -> object:
    value = target
    for part in dotted_name.split("."):
        value = getattr(value, part)
    return value


def _signature_params(target: object) -> list[str]:
    original_target = getattr(target, "_mint_original", target)
    signature = _inspect.signature(original_target)
    return list(signature.parameters)


def _requested_tinker_api_key(kwargs: dict[str, object]) -> str | None:
    api_key = kwargs.get("api_key")
    if isinstance(api_key, str):
        return api_key

    env_api_key = _os.environ.get(_TINKER_API_KEY_ENV)
    if isinstance(env_api_key, str) and env_api_key:
        return env_api_key

    return None


def _is_mint_api_key(value: object) -> bool:
    return isinstance(value, str) and value.startswith(_MINT_API_KEY_PREFIX)


def _make_mint_api_key_auth_provider(api_key: str):
    try:
        from tinker.lib._auth_token_provider import AuthTokenProvider
    except ModuleNotFoundError:
        AuthTokenProvider = object

    class _MintApiKeyAuthProvider(AuthTokenProvider):
        def __init__(self, token: str) -> None:
            self._token = token

        async def get_token(self) -> str:
            return self._token

    return _MintApiKeyAuthProvider(api_key)


def _unsupported_tinker_message(actual: str, tinker_file: str) -> str:
    supported = ", ".join(SUPPORTED_TINKER_VERSIONS)
    return (
        "mindlab-toolkit requires a validated Tinker SDK version.\n"
        f"Supported tinker versions: {supported}\n"
        f"Installed tinker version: {actual or 'unknown'}\n"
        f"Loaded from: {tinker_file or 'unknown'}\n\n"
        "Fix this environment with:\n"
        f"  python -m pip install --force-reinstall 'tinker=={EXPECTED_TINKER_VERSION}'\n\n"
        "Then retry your MinT script."
    )


def _assert_supported_tinker_version(tinker_module: object) -> str:
    actual = str(getattr(tinker_module, "__version__", ""))
    if actual in SUPPORTED_TINKER_VERSIONS:
        return actual
    if _os.environ.get(_ALLOW_UNSUPPORTED_TINKER_ENV) in {"1", "true", "TRUE", "yes", "YES"}:
        return actual
    tinker_file = str(getattr(tinker_module, "__file__", ""))
    raise RuntimeError(_unsupported_tinker_message(actual, tinker_file))


def assert_tinker_compat() -> str:
    """Raise RuntimeError if installed tinker is not a validated MinT dependency."""
    import tinker

    actual = _assert_supported_tinker_version(tinker)
    exports = getattr(tinker, "__all__", None)
    if not isinstance(exports, (list, tuple)):
        raise RuntimeError(
            "mindlab-toolkit requires tinker to expose a list-like __all__ surface "
            f"compatible with {SUPPORTED_TINKER_SPEC}; found tinker=={actual or 'unknown'}"
        )

    missing_exports = [
        name for name in _REQUIRED_TINKER_EXPORTS if name not in exports or not hasattr(tinker, name)
    ]
    if missing_exports:
        raise RuntimeError(
            "mindlab-toolkit requires installed tinker to expose the MinT compatibility surface. "
            f"Missing exports for tinker=={actual or 'unknown'}: {', '.join(missing_exports)}"
        )

    for module_name, dotted_name, required_params in _PATCH_POINTS:
        module = __import__(module_name, fromlist=["__name__"])
        try:
            target = _resolve_attr(module, dotted_name)
            actual_params = _signature_params(target)
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "mindlab-toolkit requires installed tinker to keep the MinT patch points stable. "
                f"Missing or unreadable patch point {module_name}:{dotted_name} "
                f"for tinker=={actual or 'unknown'}"
            ) from exc

        missing_params = [name for name in required_params if name not in actual_params]
        if missing_params:
            raise RuntimeError(
                "mindlab-toolkit requires installed tinker to keep MinT patch point parameters stable. "
                f"Patch point {module_name}:{dotted_name} on tinker=={actual or 'unknown'} "
                f"is missing parameters: {', '.join(missing_params)}"
            )

    return actual


def assert_tinker_version() -> str:
    """Backward-compatible alias for capability-based validation."""
    return assert_tinker_compat()


def _env_ms(name: str, default_ms: int) -> float:
    value = _os.getenv(name)
    if value is None:
        return max(0.0, default_ms / 1000.0)
    try:
        return max(0.0, float(value) / 1000.0)
    except (TypeError, ValueError):
        return max(0.0, default_ms / 1000.0)


def _env_seconds(name: str, default_seconds: float) -> float:
    value = _os.getenv(name)
    if value is None:
        return max(0.0, default_seconds)
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return max(0.0, default_seconds)


def _extract_queue_state(error: object) -> str:
    response = getattr(error, "response", None)
    if response is None:
        return "unknown"

    try:
        payload = response.json()
    except Exception:
        return "unknown"

    if not isinstance(payload, dict):
        return "unknown"

    queue_state = payload.get("queue_state")
    if not isinstance(queue_state, str) or not queue_state:
        return "unknown"
    return queue_state


def _retrieve_poll_delay_seconds(queue_state: str) -> float:
    if queue_state == "active":
        return _env_ms("TINKER_RETRIEVE_POLL_ACTIVE_MS", 300)
    if queue_state == "paused_capacity":
        return _env_ms("TINKER_RETRIEVE_POLL_PAUSED_CAPACITY_MS", 1000)
    if queue_state == "paused_rate_limit":
        return _env_ms("TINKER_RETRIEVE_POLL_PAUSED_RATE_LIMIT_MS", 2000)
    return _env_ms("TINKER_RETRIEVE_POLL_UNKNOWN_MS", 1000)


def _patch_api_key_auth_provider() -> None:
    try:
        import tinker.lib._auth_token_provider as _auth_token_provider_module
    except ModuleNotFoundError:
        # Tinker 0.15 keeps API-key validation in AsyncTinker.__init__ and
        # request headers read self.api_key, so the AsyncTinker patch is enough.
        return

    import tinker.lib.internal_client_holder as _internal_client_holder_module

    auth_provider_cls = _auth_token_provider_module.ApiKeyAuthProvider
    if getattr(_auth_token_provider_module, "_mint_auth_patch_applied", False):
        return

    original_init = auth_provider_cls.__init__
    original_resolve_auth_provider = _auth_token_provider_module.resolve_auth_provider

    def _mint_api_key_auth_provider_init(self, api_key=None):
        resolved = api_key or _os.environ.get(_TINKER_API_KEY_ENV)
        if _is_mint_api_key(resolved):
            # This is request-time auth state, not Tinker client construction.
            # The constructor spoofing happens in the AsyncTinker patch below.
            self._token = resolved
            return None
        return original_init(self, api_key=api_key)

    def _mint_resolve_auth_provider(api_key, enforce_cmd):
        resolved = api_key or _os.environ.get(_TINKER_API_KEY_ENV, "")
        if _is_mint_api_key(resolved):
            return _make_mint_api_key_auth_provider(resolved)
        return original_resolve_auth_provider(api_key, enforce_cmd)

    _mint_api_key_auth_provider_init._mint_original = original_init
    _mint_resolve_auth_provider._mint_original = original_resolve_auth_provider
    auth_provider_cls.__init__ = _mint_api_key_auth_provider_init
    auth_provider_cls._mint_patch_applied = True
    _auth_token_provider_module.resolve_auth_provider = _mint_resolve_auth_provider
    _internal_client_holder_module.resolve_auth_provider = _mint_resolve_auth_provider
    _auth_token_provider_module._mint_auth_patch_applied = True


def _patch_service_client() -> None:
    import tinker.lib.public_interfaces.service_client as _service_client_module

    if getattr(_service_client_module, "_mint_patch_applied", False):
        return

    mint_headers = {"User-Agent": f"Mint/Python {MINT_VERSION}"}
    original_get_headers = _service_client_module._get_default_headers
    original_service_client_init = _service_client_module.ServiceClient.__init__

    def _mint_get_default_headers():
        return {**original_get_headers(), **mint_headers}

    def _mint_service_client_init(self, *args, **kwargs):
        # Re-sync env at client construction time so load_dotenv() works even
        # if called after importing mint.
        sync_env()
        return original_service_client_init(self, *args, **kwargs)

    _mint_get_default_headers._mint_original = original_get_headers
    _mint_service_client_init._mint_original = original_service_client_init
    _service_client_module._get_default_headers = _mint_get_default_headers
    _service_client_module.ServiceClient.__init__ = _mint_service_client_init
    _service_client_module._mint_patch_applied = True


def _patch_async_tinker_init() -> None:
    import tinker._client as _client_module

    if getattr(_client_module, "_mint_patch_applied", False):
        return

    original_async_tinker_init = _client_module.AsyncTinker.__init__

    def _mint_async_tinker_init(self, *args, **kwargs):
        sync_env()
        requested_api_key = _requested_tinker_api_key(kwargs)
        if not _is_mint_api_key(requested_api_key) or "_auth" in kwargs:
            result = original_async_tinker_init(self, *args, **kwargs)
            if requested_api_key:
                self.api_key = requested_api_key
            return result

        original_env_api_key = _os.environ.get(_TINKER_API_KEY_ENV)
        env_api_key_present = _TINKER_API_KEY_ENV in _os.environ
        patched_kwargs = dict(kwargs)
        if isinstance(patched_kwargs.get("api_key"), str):
            patched_kwargs["api_key"] = _TINKER_COMPAT_PLACEHOLDER_API_KEY
        else:
            _os.environ[_TINKER_API_KEY_ENV] = _TINKER_COMPAT_PLACEHOLDER_API_KEY

        try:
            result = original_async_tinker_init(self, *args, **patched_kwargs)
        finally:
            if env_api_key_present:
                _os.environ[_TINKER_API_KEY_ENV] = original_env_api_key or ""
            else:
                _os.environ.pop(_TINKER_API_KEY_ENV, None)

        self._auth = _make_mint_api_key_auth_provider(requested_api_key)
        self.api_key = requested_api_key
        return result

    _mint_async_tinker_init._mint_original = original_async_tinker_init
    _client_module.AsyncTinker.__init__ = _mint_async_tinker_init
    _client_module._mint_patch_applied = True


def _patch_sampling_session_model_path() -> None:
    from tinker import types as _types
    from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
    from tinker.lib.internal_client_holder import InternalClientHolder

    if getattr(InternalClientHolder, "_mint_patch_applied", False):
        return

    async def _mint_create_sampling_session(self, *args, model_path=None, base_model=None, **kwargs):
        """Patched version: no validation, pass path directly to server."""
        if args:
            if len(args) > 2:
                raise TypeError("_create_sampling_session accepts at most model_path and base_model")
            if len(args) >= 1:
                model_path = args[0]
            if len(args) >= 2:
                base_model = args[1]
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"_create_sampling_session received unexpected keyword arguments: {unexpected}")

        sampling_session_seq_id = self._sampling_client_counter
        self._sampling_client_counter += 1
        with self.aclient(ClientConnectionPoolType.SESSION) as client:
            request = _types.CreateSamplingSessionRequest(
                session_id=self._session_id,
                sampling_session_seq_id=sampling_session_seq_id,
                model_path=model_path,
                base_model=base_model,
            )
            result = await client.service.create_sampling_session(request=request)
            return result.sampling_session_id

    _mint_create_sampling_session._mint_original = InternalClientHolder._create_sampling_session
    InternalClientHolder._create_sampling_session = _mint_create_sampling_session
    InternalClientHolder._mint_patch_applied = True


def _patch_retrieve_future_polling() -> None:
    import tinker
    import tinker.resources.futures as _futures_resource_module

    current_retrieve = _futures_resource_module.AsyncFuturesResource.retrieve
    if getattr(current_retrieve, "_mint_busy_poll_patch", False):
        return

    original_retrieve = current_retrieve

    async def _mint_async_futures_retrieve(self, *args, **kwargs):
        try:
            return await original_retrieve(self, *args, **kwargs)
        except tinker.APIStatusError as error:
            if error.status_code == 408:
                delay_seconds = _retrieve_poll_delay_seconds(_extract_queue_state(error))
                if delay_seconds > 0.0:
                    await _asyncio.sleep(delay_seconds)
            raise

    _mint_async_futures_retrieve._mint_busy_poll_patch = True
    _mint_async_futures_retrieve._mint_original = original_retrieve
    _futures_resource_module.AsyncFuturesResource.retrieve = _mint_async_futures_retrieve


def _patch_telemetry_408_sampling() -> None:
    import tinker.lib.telemetry as _telemetry_module

    current_log = _telemetry_module.Telemetry.log
    if getattr(current_log, "_mint_408_sampling_patch", False):
        return

    original_log = current_log

    def _mint_telemetry_log(self, event_name, event_data=None, severity="INFO"):
        if (
            event_name == "APIFuture.result_async.api_status_error"
            and isinstance(event_data, dict)
            and event_data.get("status_code") == 408
        ):
            min_interval = _env_seconds("TINKER_408_TELEMETRY_MIN_INTERVAL_S", 60.0)
            if min_interval > 0.0:
                now = _time.monotonic()
                last = getattr(self, "_mint_last_408_telemetry_ts", None)
                if last is not None and now - last < min_interval:
                    return False
                setattr(self, "_mint_last_408_telemetry_ts", now)

        return original_log(self, event_name, event_data=event_data, severity=severity)

    _mint_telemetry_log._mint_408_sampling_patch = True
    _mint_telemetry_log._mint_original = original_log
    _telemetry_module.Telemetry.log = _mint_telemetry_log


def _ensure_datum_weights(datum):
    """Return datum with `weights` in loss_fn_inputs if no mask field exists."""
    lfi = datum.loss_fn_inputs
    if not isinstance(lfi, dict):
        return datum
    if "weights" in lfi or "loss_mask" in lfi or "mask" in lfi:
        return datum

    import tinker.types as _types

    target = lfi.get("target_tokens")
    advantages = lfi.get("advantages")

    if advantages is not None:
        raw = advantages.data if hasattr(advantages, "data") else advantages
        # `advantages` controls the GRPO update direction and magnitude.
        # `weights` only marks trainable agent tokens, so zero-advantage tokens
        # should still participate with a neutral gradient contribution.
        weights_list = [1.0] * len(raw)
    elif target is not None:
        raw = target.data if hasattr(target, "data") else target
        weights_list = [1.0] * len(raw)
    else:
        return datum

    weights_td = _types.TensorData(data=weights_list, dtype="float32")
    new_lfi = {**lfi, "weights": weights_td}
    return _types.Datum(model_input=datum.model_input, loss_fn_inputs=new_lfi)


def _patch_forward_backward_datum_weights() -> None:
    import tinker

    tc_cls = tinker.TrainingClient
    if getattr(tc_cls, "_mint_fb_weights_patch_applied", False):
        return

    original_fb = tc_cls.forward_backward
    original_fba = tc_cls.forward_backward_async
    original_fbc = tc_cls.forward_backward_custom
    original_fbca = tc_cls.forward_backward_custom_async

    def _ensure_data_weights(data):
        return [_ensure_datum_weights(d) for d in data]

    def _mint_forward_backward(self, data, loss_fn, loss_fn_config=None):
        data = _ensure_data_weights(data)
        return original_fb(self, data, loss_fn, loss_fn_config=loss_fn_config)

    def _mint_forward_backward_async(self, data, loss_fn, loss_fn_config=None):
        data = _ensure_data_weights(data)
        return original_fba(self, data, loss_fn, loss_fn_config=loss_fn_config)

    def _mint_forward_backward_custom(self, data, loss_fn):
        data = _ensure_data_weights(data)
        return original_fbc(self, data, loss_fn)

    async def _mint_forward_backward_custom_async(self, data, loss_fn):
        data = _ensure_data_weights(data)
        return await original_fbca(self, data, loss_fn)

    _mint_forward_backward._mint_original = original_fb
    _mint_forward_backward_async._mint_original = original_fba
    _mint_forward_backward_custom._mint_original = original_fbc
    _mint_forward_backward_custom_async._mint_original = original_fbca
    tc_cls.forward_backward = _mint_forward_backward
    tc_cls.forward_backward_async = _mint_forward_backward_async
    tc_cls.forward_backward_custom = _mint_forward_backward_custom
    tc_cls.forward_backward_custom_async = _mint_forward_backward_custom_async
    tc_cls._mint_fb_weights_patch_applied = True


def apply_mint_patches() -> None:
    """Apply MinT compatibility patches once per interpreter."""
    sync_env()
    assert_tinker_compat()
    if _PATCH_STATE["applied"]:
        return

    _patch_api_key_auth_provider()
    _patch_service_client()
    _patch_async_tinker_init()
    _patch_sampling_session_model_path()
    _patch_retrieve_future_polling()
    _patch_telemetry_408_sampling()
    _patch_forward_backward_datum_weights()
    _PATCH_STATE["applied"] = True


__all__ = [
    "ForwardBackwardReverseKLRequest",
    "ForwardBackwardReverseKLResponse",
    "InterpolateCheckpointsRequest",
    "InterpolateCheckpointsResponse",
    "MINT_VERSION",
    "ReverseKLDatum",
    "ReverseKLItemOutput",
    "SUPPORTED_TINKER_SPEC",
    "SUPPORTED_TINKER_VERSIONS",
    "EXPECTED_TINKER_VERSION",
    "CAMERA_LAYOUT",
    "OPENPI_FAST_MODEL",
    "OPENPI_FAST_LORA_RANK",
    "OpenPITrainingClient",
    "build_openpi_fast_datum",
    "create_openpi_training_client",
    "create_openpi_training_client_async",
    "apply_mint_patches",
    "assert_tinker_compat",
    "assert_tinker_version",
    "forward_backward_reverse_kl",
    "forward_backward_reverse_kl_async",
    "interpolate_checkpoints",
    "interpolate_checkpoints_async",
    "sync_env",
]
