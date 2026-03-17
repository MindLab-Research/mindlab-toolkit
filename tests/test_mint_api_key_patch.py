from __future__ import annotations

import asyncio
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import mint  # noqa: F401,E402
import tinker._client as client_module  # noqa: E402
import tinker.lib.internal_client_holder as holder_module  # noqa: E402


@pytest.mark.parametrize(
    ("explicit_api_key", "env_name", "env_value"),
    [
        ("sk-explicit-test", None, None),
        (None, "MINT_API_KEY", "sk-mint-env-test"),
        (None, "TINKER_API_KEY", "sk-tinker-env-test"),
    ],
)
def test_async_tinker_accepts_non_tml_api_keys(
    monkeypatch: pytest.MonkeyPatch,
    explicit_api_key: str | None,
    env_name: str | None,
    env_value: str | None,
) -> None:
    monkeypatch.delenv("MINT_API_KEY", raising=False)
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    monkeypatch.delenv("TINKER_APIKEY", raising=False)

    if env_name and env_value:
        monkeypatch.setenv(env_name, env_value)

    kwargs = {"base_url": "https://example.invalid"}
    if explicit_api_key is not None:
        kwargs["api_key"] = explicit_api_key

    client = client_module.AsyncTinker(**kwargs)
    try:
        expected_api_key = explicit_api_key or env_value
        assert client.api_key == expected_api_key
        assert client.auth_headers["X-API-Key"] == expected_api_key
    finally:
        asyncio.run(client.close())


def test_async_tinker_keeps_real_tml_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MINT_API_KEY", raising=False)
    monkeypatch.setenv("TINKER_API_KEY", "tml-live-test")

    client = client_module.AsyncTinker(base_url="https://example.invalid")
    try:
        assert client.api_key == "tml-live-test"
        assert client.auth_headers["X-API-Key"] == "tml-live-test"
    finally:
        asyncio.run(client.close())


@pytest.mark.parametrize("model_path", ["mint://model/checkpoint", "ckpt_uploaded"])
def test_sampling_session_patch_allows_non_tinker_model_paths(model_path: str) -> None:
    captured: dict[str, object] = {}

    class _FakeClient:
        def __init__(self) -> None:
            self.service = self

        async def create_sampling_session(self, *, request):
            captured["request"] = request
            return SimpleNamespace(sampling_session_id="sample-123")

    class _FakeHolder:
        _sampling_client_counter = 0
        _session_id = "session-123"

        @contextmanager
        def aclient(self, _pool_type):
            yield _FakeClient()

    sampling_session_id = asyncio.run(
        holder_module.InternalClientHolder._create_sampling_session(
            _FakeHolder(),
            model_path=model_path,
            base_model="Qwen/Qwen3-0.6B",
        )
    )

    assert sampling_session_id == "sample-123"
    assert captured["request"].model_path == model_path
