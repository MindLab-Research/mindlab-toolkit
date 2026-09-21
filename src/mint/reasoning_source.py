"""Where assistant reasoning leftovers come from.

Kept outside ``mint.renderers`` so the GLM SFT validator can import it
without executing ``mint.renderers.__init__`` (torch / tinker-cookbook).
See ``mint/renderers/README.md``.
"""

from __future__ import annotations

from typing import Any

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _is_thinking_envelope(part: dict[str, Any]) -> bool:
    return part.get("type") == "thinking" or "thinking" in part


def _text_part_payload(part: dict[str, Any]) -> str | None:
    """Text from a leftover-visible part, or ``None`` if this part aborts leftover.

    Only a real text part: ``type=="text"``, or a typeless ``{"text": "..."}``
    envelope. A media part that also has a ``text`` key (caption / reminder)
    is not a serving dump — leftover must abort so the image is not dropped.
    """
    if _is_thinking_envelope(part):
        return None
    part_type = part.get("type")
    if part_type == "text" or (part_type is None and "text" in part):
        text = part.get("text")
        return text if isinstance(text, str) else None
    return None


def leftover_visible_source(content: Any) -> str | None:
    """Visible text that leftover promotion may split, or ``None``.

    Strings and lists of string / text parts are the same serving dump.
    A ThinkingPart, media part, or any other envelope means leftover
    must not run (structured CoT already won, or we cannot recover a
    single dumped completion).
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    texts: list[str] = []
    for part in content:
        if isinstance(part, str):
            texts.append(part)
            continue
        if not isinstance(part, dict):
            return None
        text = _text_part_payload(part)
        if text is None:
            return None
        texts.append(text)
    if not texts:
        return None
    return "".join(texts)


def visible_has_think_markup(content: Any) -> bool:
    """Whether visible answer text already contains a think tag.

    Used by the Qwen False-path empty scaffold. Leftover promotion runs
    first; this only matters when leftover returned ``None``.

    ``</think>`` does **not** contain the substring ``<think>`` (the slash
    sits between ``<`` and ``think>``). Both tags must be checked, or a
    leftover-abort row whose text is ``</think>ANSWER`` plus an image
    would stack a second official empty block.

    String / text-part lists use the same concatenation as leftover.
    When leftover aborts (media / other envelopes), walk remaining text.
    """
    source = leftover_visible_source(content)
    if source is not None:
        return _THINK_OPEN in source or _THINK_CLOSE in source
    if not isinstance(content, list):
        return False
    for part in content:
        if isinstance(part, str) and (
            _THINK_OPEN in part or _THINK_CLOSE in part
        ):
            return True
        if not isinstance(part, dict) or _is_thinking_envelope(part):
            continue
        text = _text_part_payload(part)
        if text is not None and (_THINK_OPEN in text or _THINK_CLOSE in text):
            return True
    return False


def split_serving_fragment(content: Any) -> tuple[str, str] | None:
    """Split a serving leftover ``REASON</think>ANSWER`` into field + visible.

    Training leftover only: the generation prompt already prefills
    ``<think>``, so a dumped completion has a close tag and no open tag.
    Promote when the unstructured visible text (string ``content``, or a
    list of string / text parts concatenated) contains ``</think>`` and
    the first close is **not** preceded by a complete ``<think>``.
    An empty left side is still a leftover: the model chose not to think
    and emitted ``</think>ANSWER`` after the prefilled ``<think>``.

    ``REASON</think>`` (empty right side) is the same rule: the model
    closed and wrote no answer. That is leftover, not a parse error.
    It is poor data; the GLM SFT validator warns. Do not refuse it here.

    ``see <think>x</think> please`` is not a leftover. The prefix before
    the first close is ``see <think>x``, which already contains the
    contiguous open tag ``<think>`` — return ``None`` and keep the whole
    string as the answer. The same check keeps a copied complete block
    (``<think>\\nREASON\\n</think>\\n\\nANSWER``) as visible text. Do not
    "fix" this by requiring the open tag to sit at index 0.

    Concrete splits::

        "</think>ANSWER" → ("", "ANSWER")
        "</think>REASON" → ("", "REASON")
        "REASON</think>" → ("REASON", "")
        "REASON</think>see <think>x</think>" → ("REASON", "see <think>x</think>")
        "see <think>x</think> please" → None
        "<think></think>ANSWER" → None

    The leftover+mention row is still leftover: the first close has no
    open in its prefix. The mention sits in the **answer**. Do not treat
    it as the mention-only row above.

    This is not ``parse_response``. Serving parse always splits on the
    first ``</think>`` after stripping a leading ``<think>`` from sampler
    output. Leftover promotion is a training-data heuristic and must
    refuse copied complete blocks.

    ``"</think>"`` does not contain the substring ``"<think>"``.
    """
    source = leftover_visible_source(content)
    if source is None:
        return None
    close_at = source.find(_THINK_CLOSE)
    if close_at < 0:
        return None
    # Prefix before the first close. A complete "<think>" there means the
    # model (or a copied block) already opened a think span in the answer
    # text — not a serving leftover. A leftover+mention such as
    # "REASON</think>see <think>x</think>" has prefix "REASON" and must
    # still promote; the mention is the answer, not the reason.
    if _THINK_OPEN in source[:close_at]:
        return None
    return source[:close_at], source[close_at + len(_THINK_CLOSE) :]
