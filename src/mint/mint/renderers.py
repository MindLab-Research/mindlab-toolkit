"""mint.mint.renderers — re-export of tinker_cookbook.renderers plus supervised helpers.

Provides the full renderer surface (get_renderer, build_supervised_example, etc.)
and two commonly-used helpers that complete the renderer-to-Datum workflow:

- ``get_recommended_renderer_name``: model name -> renderer name
- ``datum_from_model_input_weights``: (ModelInput, weights) -> Datum
"""

try:
    from tinker_cookbook.renderers import *  # noqa: F401,F403
    from tinker_cookbook import renderers as _renderers_mod
except ImportError as _exc:
    raise ImportError(
        "mint.mint.renderers requires the 'tinker-cookbook' package.\n"
        "Install it with:  pip install mindlab-toolkit\n"
        "Or directly:      pip install tinker-cookbook"
    ) from _exc

from tinker_cookbook.model_info import get_recommended_renderer_name  # noqa: F401
from tinker_cookbook.supervised.common import datum_from_model_input_weights  # noqa: F401

_cookbook_all = getattr(_renderers_mod, "__all__", [
    name for name in dir(_renderers_mod)
    if not name.startswith("_")
])

__all__ = [
    *_cookbook_all,
    "get_recommended_renderer_name",
    "datum_from_model_input_weights",
]
