"""Namespace with explicit Tinker-compatible MinT exports."""

from __future__ import annotations

import tinker as _tinker

_REQUIRED_TINKER_EXPORTS = (
    "TrainingClient",
    "ServiceClient",
    "SamplingClient",
    "APIFuture",
    "types",
    "__version__",
    "__title__",
)

_tinker_all = getattr(_tinker, "__all__", None)
if not isinstance(_tinker_all, (list, tuple)):
    raise RuntimeError("mindlab-toolkit requires tinker to expose a list-like __all__")

missing_exports = [
    name for name in _REQUIRED_TINKER_EXPORTS if name not in _tinker_all or not hasattr(_tinker, name)
]
if missing_exports:
    raise RuntimeError(
        "mindlab-toolkit requires installed tinker to expose the MinT compatibility surface; "
        f"missing exports: {', '.join(missing_exports)}"
    )

TINKER_COMPAT_EXPORTS = [str(name) for name in _tinker_all]

for _symbol in TINKER_COMPAT_EXPORTS:
    if not hasattr(_tinker, _symbol):
        raise RuntimeError(
            f"tinker is missing expected symbol {_symbol!r}; "
            "mindlab-toolkit requires tinker symbols compatible with the latest MinT surface"
        )
    globals()[_symbol] = getattr(_tinker, _symbol)

__all__ = [*TINKER_COMPAT_EXPORTS]
