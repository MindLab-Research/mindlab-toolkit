"""Reasoning-effort levels a model family actually writes into the prompt.

Kept free of renderer imports so validation can use it without loading tinker.

GLM-5.2 ``chat_template.jinja`` keeps an explicit ``high`` and otherwise writes
``max``. GLM-5.3 keeps ``low`` or ``high`` and otherwise writes ``max``.
"""

from __future__ import annotations

GLM52_REASONING_EFFORTS = ("high", "max")
GLM53_REASONING_EFFORTS = ("low", "high", "max")


def reasoning_efforts_for_model(model_name: str) -> tuple[str, ...]:
    """Return the efforts this base model can put on the system line."""
    head = model_name.split(":", 1)[0]
    if "GLM-5.3" in head:
        return GLM53_REASONING_EFFORTS
    return GLM52_REASONING_EFFORTS
