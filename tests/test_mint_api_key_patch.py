from __future__ import annotations

import asyncio
import os
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


def test_async_tinker_uses_placeholder_for_local_mint_key_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MINT_API_KEY", raising=False)
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    seen_api_keys: list[str | None] = []
    current_init = client_module.AsyncTinker.__init__
    original_init = current_init._mint_original

    def strict_tinker_init(self, *args, **kwargs) -> None:
        resolved = kwargs.get("api_key") or os.environ.get("TINKER_API_KEY")
        seen_api_keys.append(resolved)
        if isinstance(resolved, str) and resolved.startswith("sk-"):
            raise AssertionError("MinT key reached Tinker local constructor")
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(client_module.AsyncTinker, "__init__", strict_tinker_init)
    monkeypatch.setattr(client_module, "_mint_patch_applied", False, raising=False)
    mint.mint._patch_async_tinker_init()

    client = client_module.AsyncTinker(
        api_key="sk-explicit-test",
        base_url="https://example.invalid",
    )
    try:
        assert seen_api_keys == ["tml-mint-compat-placeholder"]
        assert client.auth_headers["X-API-Key"] == "sk-explicit-test"
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
