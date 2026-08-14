"""MinT auth compatibility: let ``sk-`` keys pass tinker's prefix validation.

tinker's ``ApiKeyAuthProvider`` (tinker.lib._auth_token_provider) rejects any
key that does not start with ``tml-`` or ``eyJ``. MinT issues ``sk-`` keys. The
rejection fires inside ``InternalClientHolder.__init__`` via two code paths:

1. ``resolve_auth_provider(api_key, ...)`` at client-construction time, before any
   custom ``_auth`` argument takes effect
2. Direct instantiation of ``ApiKeyAuthProvider(api_key=...)`` when
   ``pjwt_auth_enabled=False`` (line 245 in internal_client_holder.py)

``internal_client_holder`` imports ``resolve_auth_provider`` by value
(``from ..._auth_token_provider import resolve_auth_provider``), so both module
namespaces must be overwritten; patching the origin module alone is a no-op for
the holder's call site.

This module patches both code paths to ensure ``sk-`` keys work in all scenarios.
"""

from __future__ import annotations

import os

from tinker.lib._auth_token_provider import AuthTokenProvider

MINT_API_KEY_PREFIX = "sk-"


class MintApiKeyAuthProvider(AuthTokenProvider):
    """Auth provider for MinT ``sk-`` keys, without tinker's prefix check."""

    def __init__(self, token: str) -> None:
        self._token = token

    async def get_token(self) -> str | None:
        return self._token


def apply_auth_patch() -> None:
    """Patch ``resolve_auth_provider`` and ``ApiKeyAuthProvider`` so ``sk-`` keys authenticate.

    Two patches are required:
    1. resolve_auth_provider: for JWT-enabled authentication flow
    2. ApiKeyAuthProvider.__init__: for direct instantiation when pjwt_auth_enabled=False
    """
    import tinker.lib._auth_token_provider as _atp
    import tinker.lib.internal_client_holder as _ich

    if getattr(_atp, "_mint_auth_patch_applied", False):
        return

    # Patch 1: resolve_auth_provider (existing)
    original_resolve = _atp.resolve_auth_provider

    def _mint_resolve_auth_provider(api_key, enforce_cmd):
        key = api_key or os.environ.get("TINKER_API_KEY", "")
        if isinstance(key, str) and key.startswith(MINT_API_KEY_PREFIX):
            return MintApiKeyAuthProvider(key)
        return original_resolve(api_key, enforce_cmd)

    _mint_resolve_auth_provider._mint_original = original_resolve
    _atp.resolve_auth_provider = _mint_resolve_auth_provider
    _ich.resolve_auth_provider = _mint_resolve_auth_provider

    # Patch 2: ApiKeyAuthProvider.__init__ (new)
    # Handles direct instantiation in internal_client_holder.py:245
    original_init = _atp.ApiKeyAuthProvider.__init__

    def _patched_api_key_init(self, *, api_key):
        if isinstance(api_key, str) and api_key.startswith(MINT_API_KEY_PREFIX):
            # Accept sk- keys by setting _token directly, bypassing prefix check
            self._token = api_key
        else:
            # Use original logic for tml- keys
            original_init(self, api_key=api_key)

    _atp.ApiKeyAuthProvider.__init__ = _patched_api_key_init

    _atp._mint_auth_patch_applied = True


__all__ = ["MintApiKeyAuthProvider", "apply_auth_patch", "MINT_API_KEY_PREFIX"]
