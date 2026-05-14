"""mint.recipe — re-export of tinker_cookbook."""
try:
    import tinker_cookbook as _tc
except ImportError as _exc:
    raise ImportError(
        "mint.recipe requires the 'tinker-cookbook' package.\n"
        "Install it with:  pip install mindlab-toolkit\n"
        "Or directly:      pip install tinker-cookbook"
    ) from _exc

# Top-level subpackages
from tinker_cookbook import (  # noqa: F401
    renderers,
    completers,
    rl,
    tool_use,
    supervised,
    distillation,
    eval,
    weights,
    sandbox,
    preference,
    stores,
    utils,
    xmux,
    model_info,
    recipes,
    third_party,
)

# Nested modules that parent __init__.py doesn't import
import tinker_cookbook.rl.data_processing  # noqa: F401
import tinker_cookbook.rl.metrics  # noqa: F401
import tinker_cookbook.rl.train  # noqa: F401
import tinker_cookbook.rl.rollouts  # noqa: F401
import tinker_cookbook.supervised.train  # noqa: F401

# Recipe implementations (namespace packages, not auto-exposed by recipes/__init__)
import tinker_cookbook.recipes.chat_sl  # noqa: F401
import tinker_cookbook.recipes.code_rl  # noqa: F401
import tinker_cookbook.recipes.distillation  # noqa: F401
import tinker_cookbook.recipes.harbor_rl  # noqa: F401
import tinker_cookbook.recipes.math_rl  # noqa: F401
import tinker_cookbook.recipes.multiplayer_rl  # noqa: F401
import tinker_cookbook.recipes.preference  # noqa: F401
import tinker_cookbook.recipes.prompt_distillation  # noqa: F401
import tinker_cookbook.recipes.rl_basic  # noqa: F401
import tinker_cookbook.recipes.rl_loop  # noqa: F401
import tinker_cookbook.recipes.rubric  # noqa: F401
import tinker_cookbook.recipes.sdft  # noqa: F401
import tinker_cookbook.recipes.search_tool  # noqa: F401
import tinker_cookbook.recipes.sl_basic  # noqa: F401
import tinker_cookbook.recipes.sl_loop  # noqa: F401
import tinker_cookbook.recipes.true_thinking_score  # noqa: F401
import tinker_cookbook.recipes.verifiers_rl  # noqa: F401
import tinker_cookbook.recipes.vlm_classifier  # noqa: F401

# Convenience shortcuts
from tinker_cookbook.supervised.common import datum_from_model_input_weights  # noqa: F401
from tinker_cookbook.model_info import get_recommended_renderer_name  # noqa: F401

__all__ = [
    # Subpackages
    "renderers",
    "completers",
    "rl",
    "tool_use",
    "supervised",
    "distillation",
    "eval",
    "weights",
    "sandbox",
    "preference",
    "stores",
    "utils",
    "xmux",
    "model_info",
    "recipes",
    "third_party",
    # Convenience shortcuts
    "datum_from_model_input_weights",
    "get_recommended_renderer_name",
]
