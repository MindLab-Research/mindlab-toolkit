"""MinT-specific compatibility layer built on top of tinker."""

from __future__ import annotations

import asyncio as _asyncio
import os as _os
import time as _time

MINT_VERSION = "0.1.0"
EXPECTED_TINKER_VERSION = "0.6.0"
_MINT_DEFAULT_BASE_URL = "https://mint.macaron.im"
_PATCH_STATE = {"applied": False}


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


def assert_tinker_version() -> str:
    """Raise RuntimeError if installed tinker != EXPECTED_TINKER_VERSION."""
    import tinker

    actual = str(getattr(tinker, "__version__", ""))
    if actual != EXPECTED_TINKER_VERSION:
        raise RuntimeError(
            f"mindlab-toolkit requires tinker=={EXPECTED_TINKER_VERSION}, "
            f"but found tinker=={actual or 'unknown'}. "
            f"Install with: pip install 'tinker=={EXPECTED_TINKER_VERSION}'"
        )
    return actual


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
        return original_async_tinker_init(self, *args, **kwargs)

    _client_module.AsyncTinker.__init__ = _mint_async_tinker_init
    _client_module._mint_patch_applied = True


def _patch_sampling_session_model_path() -> None:
    from tinker import types as _types
    from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
    from tinker.lib.internal_client_holder import InternalClientHolder

    if getattr(InternalClientHolder, "_mint_patch_applied", False):
        return

    async def _mint_create_sampling_session(self, model_path=None, base_model=None):
        """Patched version: no validation, pass path directly to server."""
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

    InternalClientHolder._create_sampling_session = _mint_create_sampling_session
    InternalClientHolder._mint_patch_applied = True


def _patch_retrieve_future_polling() -> None:
    import tinker
    import tinker.resources.futures as _futures_resource_module

    current_retrieve = _futures_resource_module.AsyncFuturesResource.retrieve
    if getattr(current_retrieve, "_mint_busy_poll_patch", False):
        return

    original_retrieve = current_retrieve

    async def _mint_async_futures_retrieve(
        self,
        *,
        request,
        extra_headers=None,
        extra_query=None,
        extra_body=None,
        timeout=None,
        idempotency_key=None,
        max_retries=None,
    ):
        try:
            return await original_retrieve(
                self,
                request=request,
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                idempotency_key=idempotency_key,
                max_retries=max_retries,
            )
        except tinker.APIStatusError as error:
            if error.status_code == 408:
                delay_seconds = _retrieve_poll_delay_seconds(_extract_queue_state(error))
                if delay_seconds > 0.0:
                    await _asyncio.sleep(delay_seconds)
            raise

    _mint_async_futures_retrieve._mint_busy_poll_patch = True
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
    _telemetry_module.Telemetry.log = _mint_telemetry_log


def apply_mint_patches() -> None:
    """Apply MinT compatibility patches once per interpreter."""
    sync_env()
    assert_tinker_version()
    if _PATCH_STATE["applied"]:
        return

    _patch_service_client()
    _patch_async_tinker_init()
    _patch_sampling_session_model_path()
    _patch_retrieve_future_polling()
    _patch_telemetry_408_sampling()
    _PATCH_STATE["applied"] = True


__all__ = [
    "MINT_VERSION",
    "EXPECTED_TINKER_VERSION",
    "apply_mint_patches",
    "assert_tinker_version",
    "sync_env",
]
