"""mint.mint.completers — re-export of tinker_cookbook.completers."""

try:
    from tinker_cookbook.completers import *  # noqa: F401,F403
    from tinker_cookbook import completers as _completers_mod
except ImportError as _exc:
    raise ImportError(
        "mint.mint.completers requires the 'tinker-cookbook' package.\n"
        "Install it with:  pip install mindlab-toolkit\n"
        "Or directly:      pip install tinker-cookbook"
    ) from _exc

__all__ = getattr(_completers_mod, "__all__", [
    name for name in dir(_completers_mod)
    if not name.startswith("_")
])
