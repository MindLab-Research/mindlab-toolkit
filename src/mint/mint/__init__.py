"""MinT-specific compatibility layer built on top of tinker.

Scope (see openspec/changes/redesign-compat-on-tinker-0220): the only
unavoidable patch is letting MinT ``sk-`` API keys pass tinker's key-prefix
validation. base_url / headers / retries flow through tinker's public
``ServiceClient`` parameters or environment variables, so they need no patch.
"""

from __future__ import annotations

import os as _os

from ._auth import MintApiKeyAuthProvider, apply_auth_patch
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
MINT_VERSION = "0.2.0"
EXPECTED_TINKER_VERSION = "0.22.0"
_MINT_DEFAULT_BASE_URL = "https://mint.macaron.xin"
_PATCH_STATE = {"applied": False}


def sync_env() -> None:
    """Point tinker at MinT by default without overriding explicit config.

    Only fills defaults: an unset ``TINKER_BASE_URL`` is seeded so no-arg
    ``ServiceClient()`` reaches MinT. ``MINT_*`` values, when present, take
    precedence for this process. Explicit constructor arguments always win
    because tinker reads them before the environment.
    """
    if "MINT_API_KEY" in _os.environ:
        _os.environ["TINKER_API_KEY"] = _os.environ["MINT_API_KEY"]

    if "MINT_BASE_URL" in _os.environ:
        _os.environ["TINKER_BASE_URL"] = _os.environ["MINT_BASE_URL"]
    else:
        _os.environ.setdefault("TINKER_BASE_URL", _MINT_DEFAULT_BASE_URL)


def apply_mint_patches() -> None:
    """Apply MinT compatibility patches once per interpreter."""
    if _PATCH_STATE["applied"]:
        return
    sync_env()
    apply_auth_patch()
    _PATCH_STATE["applied"] = True


__all__ = [
    "MINT_VERSION",
    "EXPECTED_TINKER_VERSION",
    "MintApiKeyAuthProvider",
    "apply_auth_patch",
    "apply_mint_patches",
    "sync_env",
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
