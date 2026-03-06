"""Mind Lab Toolkit (MinT) compatibility package."""

from __future__ import annotations

from . import mint as mint
from .mint import MINT_VERSION, apply_mint_patches

# Apply compatibility patches before exposing SDK symbols.
apply_mint_patches()

from . import tinker as tinker

_TOP_LEVEL_REEXPORTS = [
    name for name in tinker.TINKER_COMPAT_EXPORTS if name != "__version__"
]

for _name in _TOP_LEVEL_REEXPORTS:
    globals()[_name] = getattr(tinker, _name)

__version__ = MINT_VERSION
__tinker_version__ = tinker.__version__

__all__ = [
    "mint",
    "tinker",
    *_TOP_LEVEL_REEXPORTS,
    "__version__",
    "__tinker_version__",
]
