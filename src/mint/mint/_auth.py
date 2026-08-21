"""MinT auth compatibility: let ``sk-`` keys pass tinker's prefix validation.

tinker's ``ApiKeyAuthProvider`` (tinker.lib._auth_token_provider) rejects any
key that does not start with ``tml-`` or ``eyJ``. MinT issues ``sk-`` keys. The
rejection is reachable through three tinker 0.22.0 call sites:

1. ``InternalClientHolder`` calls ``resolve_auth_provider(api_key, ...)``
2. ``InternalClientHolder`` directly constructs ``ApiKeyAuthProvider`` when
   ``pjwt_auth_enabled=False``
3. ``AsyncTinker`` directly constructs ``ApiKeyAuthProvider`` when no custom
   ``_auth`` provider is supplied

The holder and client modules import auth symbols by value, so their module
bindings must be patched at each call site. The original ``ApiKeyAuthProvider``
class remains unchanged for callers outside MinT.

This module patches all three paths without weakening tinker's global key
validation.
"""

from __future__ import annotations

import os

from tinker.lib._auth_token_provider import ApiKeyAuthProvider

MINT_API_KEY_PREFIX = "sk-"


class MintApiKeyAuthProvider(ApiKeyAuthProvider):
    """Auth provider for MinT ``sk-`` keys, without tinker's prefix check."""

    def __init__(self, token: str) -> None:
        self._token = token


def apply_auth_patch() -> None:
    """Patch tinker 0.22.0 auth call sites used by MinT clients.

    ``resolve_auth_provider`` covers client-config and JWT-enabled holder flows.
    Holder- and client-local ``ApiKeyAuthProvider`` bindings cover direct
    construction without changing tinker's original provider class.
    """
    import tinker._client as _client
    import tinker.lib._auth_token_provider as _atp
    import tinker.lib.internal_client_holder as _ich

    if getattr(_atp, "_mint_auth_patch_applied", False):
        return

    original_resolve = _atp.resolve_auth_provider
    original_api_key_provider = _atp.ApiKeyAuthProvider

    def _mint_resolve_auth_provider(api_key, enforce_cmd):
        key = api_key or os.environ.get("TINKER_API_KEY", "")
        if isinstance(key, str) and key.startswith(MINT_API_KEY_PREFIX):
            return MintApiKeyAuthProvider(key)
        return original_resolve(api_key, enforce_cmd)

    def _mint_api_key_auth_provider(api_key=None):
        key = api_key or os.environ.get("TINKER_API_KEY", "")
        if isinstance(key, str) and key.startswith(MINT_API_KEY_PREFIX):
            return MintApiKeyAuthProvider(key)
        return original_api_key_provider(api_key=api_key)

    _mint_resolve_auth_provider._mint_original = original_resolve
    _mint_api_key_auth_provider._mint_original = original_api_key_provider
    _atp.resolve_auth_provider = _mint_resolve_auth_provider
    _ich.resolve_auth_provider = _mint_resolve_auth_provider
    _ich.ApiKeyAuthProvider = _mint_api_key_auth_provider
    _client.ApiKeyAuthProvider = _mint_api_key_auth_provider
    _atp._mint_auth_patch_applied = True


__all__ = ["MintApiKeyAuthProvider", "apply_auth_patch", "MINT_API_KEY_PREFIX"]
