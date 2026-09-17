"""MinT Qwen3.5 renderer: one think-block contract, optional CoT loss.

An assistant turn may carry reasoning in exactly one of:

* ``reasoning_content``
* a ``ThinkingPart`` in ``content``
* a leading inline ``<think>...</think>`` in text ``content`` (normalized to a
  ThinkingPart)

``supervise_reasoning=True`` (default) puts loss on the generated think body,
``</think>``, and the visible answer. The opening ``<think>\\n`` is prefilled by
the generation prompt and stays in the unsupervised header.
``supervise_reasoning=False`` moves the whole ``<think>...</think>`` span into
the header. Structured turns keep the same token sequence as ``True``.

Intentional HF asymmetry (not a sequence-parity bug). The official Qwen3.5
template prefills generation with ``<think>\\n`` (token 198) but renders a
completed empty CoT as ``<think>\\n\\n</think>\\n\\n`` (token 271). ``True``
follows the serving prefix via an empty ThinkingPart; ``False`` with no
structure keeps the completed-empty header from ``_assistant_header_suffix``.
Do not inject on the ``False`` path just to make ``to_ints()`` match.

The upstream Qwen3.5 renderer only reads ThinkingPart and always supervises the
think block. This subclass normalizes the three writings above, then relocates
already-encoded tokens between ``header`` and ``output``.
"""

from __future__ import annotations

import re
from typing import Any

import tinker
from tinker_cookbook.exceptions import RendererError
from tinker_cookbook.renderers import register_renderer
from tinker_cookbook.renderers.base import RenderContext, RenderedMessage
from tinker_cookbook.renderers.qwen3_5 import Qwen3_5Renderer

QWEN35_RENDERER = "MindLab/qwen35"

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
# Leading think block only. Remainder must not contain another pair — that is a
# second structure, not "plain text that happens to mention the tags".
_INLINE_THINK_RE = re.compile(
    r"\A\s*<think>(.*?)</think>\s*(.*)\Z",
    re.DOTALL,
)


def _reasoning_field(message: dict[str, Any]) -> str | None:
    """Return the ``reasoning_content`` string, or None if the key is absent."""
    if "reasoning_content" not in message:
        return None
    reasoning = message["reasoning_content"]
    if not isinstance(reasoning, str):
        raise RendererError("Qwen3.5 reasoning_content must be a string")
    return reasoning


def _thinking_part_texts(content: Any) -> list[str]:
    if not isinstance(content, list):
        return []
    texts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "thinking" or "thinking" in part:
            thinking = part.get("thinking")
            if not isinstance(thinking, str):
                raise RendererError("Qwen3.5 ThinkingPart requires a string 'thinking' field")
            texts.append(thinking)
    return texts


def _split_inline_think(text: str) -> tuple[str, str] | None:
    """Split a leading ``<think>...</think>`` block from string content.

    Returns ``None`` when there are no think tags. Raises if tags are present
    but not a single well-formed leading block (unclosed, close-before-open,
    or a second block in the remainder).
    """
    if _THINK_OPEN not in text and _THINK_CLOSE not in text:
        return None
    match = _INLINE_THINK_RE.match(text)
    if match is None:
        raise RendererError(
            "inline <think> markers in Qwen3.5 content must be a single leading "
            "<think>...</think> block; use a ThinkingPart or reasoning_content, "
            "or put the tags only as a prefix of content"
        )
    reasoning, visible = match.group(1), match.group(2)
    if _THINK_OPEN in visible or _THINK_CLOSE in visible:
        raise RendererError(
            "inline <think> markers in Qwen3.5 content contain a second think block; "
            "normalize to one ThinkingPart plus visible text"
        )
    return reasoning, visible


def _text_from_content_without_thinking(content: Any) -> str | None:
    """If ``content`` is only text (string or text parts), return it joined.

    Returns None when there are non-text parts (images, etc.) so the caller
    can leave those untouched.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content) if content else ""
    texts: list[str] = []
    for part in content:
        if isinstance(part, str):
            texts.append(part)
            continue
        if isinstance(part, dict) and part.get("type") in (None, "text") and "thinking" not in part:
            texts.append(str(part.get("text", "")))
            continue
        if isinstance(part, dict) and (part.get("type") == "thinking" or "thinking" in part):
            continue
        return None
    return "".join(texts)


def _parts_with_thinking(reasoning: str, visible: Any) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = [{"type": "thinking", "thinking": reasoning}]
    if isinstance(visible, list):
        parts.extend(
            part
            for part in visible
            if not (isinstance(part, dict) and (part.get("type") == "thinking" or "thinking" in part))
        )
    elif isinstance(visible, str):
        if visible:
            parts.append({"type": "text", "text": visible})
    elif visible:
        parts.append({"type": "text", "text": str(visible)})
    return parts


def _normalize_assistant_message(message: dict[str, Any]) -> tuple[str, bool, dict[str, Any]]:
    """Normalize an assistant message to a single ThinkingPart.

    Exactly one of ``reasoning_content``, a ThinkingPart, or a leading inline
    ``<think>`` block may be present; two at once is an error. Returns
    ``(reasoning, has_structure, promoted)``. ``promoted`` never carries
    ``reasoning_content``. When ``has_structure`` is true, ``content`` is a list
    starting with a ThinkingPart, so the upstream renderer emits one think block.
    """
    promoted = {key: value for key, value in message.items() if key != "reasoning_content"}
    content = promoted.get("content", "")
    field = _reasoning_field(message)
    part_texts = _thinking_part_texts(content)
    inline_source = _text_from_content_without_thinking(content)
    inline = _split_inline_think(inline_source) if isinstance(inline_source, str) else None

    sources = [
        name
        for name, present in (
            ("reasoning_content", field is not None),
            ("ThinkingPart", bool(part_texts)),
            ("inline <think>", inline is not None),
        )
        if present
    ]
    if len(sources) > 1:
        raise RendererError(
            "use either reasoning_content, ThinkingPart, or a single leading inline "
            f"<think> block for Qwen3.5 reasoning, not {sources}"
        )

    if field is not None:
        return field, True, {**promoted, "content": _parts_with_thinking(field, content)}
    if part_texts:
        return "".join(part_texts), True, promoted
    if inline is not None:
        reasoning, visible = inline
        return reasoning, True, {**promoted, "content": _parts_with_thinking(reasoning, visible)}
    return "", False, promoted


def _content_has_thinking(content: Any) -> bool:
    return bool(_thinking_part_texts(content))


def _promote_reasoning_content(message: dict[str, Any]) -> dict[str, Any]:
    """Fold every accepted CoT writing into a ThinkingPart. See ``_normalize_assistant_message``."""
    _, _, promoted = _normalize_assistant_message(message)
    return promoted


def _inject_empty_thinking(message: dict[str, Any]) -> dict[str, Any]:
    """Give a no-reasoning answer turn an empty ThinkingPart.

    Without it, the upstream renderer parks the empty ``</think>`` in the
    unsupervised header. An empty ThinkingPart sends those tokens through
    ``_format_thinking_text`` as ``output``, so ``</think>`` can carry loss.
    """
    promoted = dict(message)
    content = promoted.get("content", "")
    if _content_has_thinking(content):
        return promoted
    parts: list[dict[str, Any]] = [{"type": "thinking", "thinking": ""}]
    if isinstance(content, list):
        parts.extend(content)
    elif isinstance(content, str):
        if content:
            parts.append({"type": "text", "text": content})
    else:
        parts.append({"type": "text", "text": str(content)})
    promoted["content"] = parts
    return promoted


def _chunk_tokens(chunk: Any) -> list[int] | None:
    tokens = getattr(chunk, "tokens", None)
    if tokens is None:
        return None
    return list(tokens)


def _mask_think_open(rendered: RenderedMessage, tokenizer: Any) -> RenderedMessage:
    """Move the leading ``<think>\\n`` from ``output`` into ``header``.

    The generation prompt already prefills ``<think>\\n``, so those tokens must
    not carry loss. The remaining output (think body, ``</think>``, answer) is
    unchanged; only the header/output split moves.
    """
    header_tokens = _chunk_tokens(rendered.header)
    if header_tokens is None:
        return rendered

    output_tokens: list[int] = []
    for chunk in rendered.output:
        chunk_tokens = _chunk_tokens(chunk)
        if chunk_tokens is None:  # non-text chunk (image/audio): leave untouched
            return rendered
        output_tokens.extend(chunk_tokens)

    open_ids = tokenizer.encode("<think>\n", add_special_tokens=False)
    if not open_ids or output_tokens[: len(open_ids)] != open_ids:
        return rendered

    new_header = tinker.types.EncodedTextChunk(tokens=header_tokens + open_ids)
    remaining = output_tokens[len(open_ids) :]
    new_output = [tinker.types.EncodedTextChunk(tokens=remaining)] if remaining else []
    return RenderedMessage(header=new_header, output=new_output)


def _mask_think_span(
    rendered: RenderedMessage,
    tokenizer: Any,
    think_block: str | None = None,
) -> RenderedMessage:
    """Move the leading ``<think>...</think>`` span from ``output`` into ``header``.

    When ``think_block`` is the exact formatted think text just rendered, the
    split is that encoded prefix — so a ``</think>`` inside the reasoning body
    is not treated as the end of the block. Without ``think_block``, falls back
    to the first ``</think>`` token (helper tests). Image chunks or a missing
    boundary leave ``rendered`` unchanged.
    """
    header_tokens = _chunk_tokens(rendered.header)
    if header_tokens is None:
        return rendered

    output_tokens: list[int] = []
    for chunk in rendered.output:
        chunk_tokens = _chunk_tokens(chunk)
        if chunk_tokens is None:  # non-text chunk (image/audio): leave untouched
            return rendered
        output_tokens.extend(chunk_tokens)

    if think_block is not None:
        block_ids = tokenizer.encode(think_block, add_special_tokens=False)
        if not block_ids or output_tokens[: len(block_ids)] != block_ids:
            raise RendererError(
                "could not locate the rendered think block to mask; the encoded "
                "<think>...</think> prefix does not match the assistant output"
            )
        split = len(block_ids)
    else:
        close_ids = tokenizer.encode(_THINK_CLOSE, add_special_tokens=False)
        if len(close_ids) != 1:
            return rendered
        close_id = close_ids[0]
        try:
            close_idx = output_tokens.index(close_id)
        except ValueError:
            return rendered
        split = close_idx + 1
        # Whitespace after </think> (Qwen emits "\n\n") stays with the think span.
        while split < len(output_tokens) and not tokenizer.decode([output_tokens[split]]).strip():
            split += 1

    new_header = tinker.types.EncodedTextChunk(tokens=header_tokens + output_tokens[:split])
    remaining = output_tokens[split:]
    new_output = [tinker.types.EncodedTextChunk(tokens=remaining)] if remaining else []
    return RenderedMessage(header=new_header, output=new_output)


class Qwen35ReasoningRenderer(Qwen3_5Renderer):
    """Qwen3.5 renderer with a ``supervise_reasoning`` loss switch.

    ``True`` (default): supervise think body, ``</think>``, and the answer.
    ``False``: think span in the header (loss only if the TrainOnWhat mode
    trains headers). Same tokens as ``True`` when a think structure is present.
    No-structure ``True``/``False`` ids differ on purpose: see the module
    docstring (HF generation prefix vs completed-empty block).
    """

    def __init__(
        self,
        *args: Any,
        supervise_reasoning: bool = True,
        strip_thinking_from_history: bool = False,
        **kwargs: Any,
    ):
        # Keep prior-turn think blocks in the sequence (constructor and factory
        # share this default). The upstream class defaults to stripping them.
        kwargs.setdefault("strip_thinking_from_history", strip_thinking_from_history)
        super().__init__(*args, **kwargs)
        self.supervise_reasoning = bool(supervise_reasoning)

    def _format_thinking_text(self, thinking: str) -> str:
        # Serving-aligned empty: "<think>\n</think>\n\n" so the opener matches
        # build_generation_prompt ([248068, 198]). Official completed-empty is
        # "<think>\n\n</think>\n\n" ([248068, 271]) — that form stays on the
        # False / no-structure path via _assistant_header_suffix, not here.
        if not thinking:
            return "<think>\n</think>\n\n"
        return super()._format_thinking_text(thinking)

    def _is_trained_answer_turn(self, message: dict[str, Any], ctx: RenderContext) -> bool:
        """True for the assistant turn(s) after the last user message, which is the
        span the Qwen3.5 template opens a think block for (and we train)."""
        if message.get("role") != "assistant":
            return False
        last_user_index = getattr(ctx, "last_user_index", -1)
        return ctx.idx > last_user_index

    def _rendered_think_block(self, promoted: dict[str, Any]) -> str:
        """The exact ``<think>...</think>`` text upstream will emit for ``promoted``.

        One formatted block per ThinkingPart, concatenated. The reasoning string
        is not stripped: render does not strip, so the mask prefix must not either.
        """
        texts = _thinking_part_texts(promoted.get("content"))
        if not texts:
            return self._format_thinking_text("")
        return "".join(self._format_thinking_text(text) for text in texts)

    def render_message(self, message: dict[str, Any], ctx: RenderContext) -> RenderedMessage:
        if message.get("role") != "assistant":
            return super().render_message(message, ctx)

        _, has_structure, promoted = _normalize_assistant_message(message)

        if self.supervise_reasoning:
            # Empty ThinkingPart so </think> is in output; opener stays in header.
            if not has_structure and self._is_trained_answer_turn(message, ctx):
                promoted = _inject_empty_thinking(promoted)
            rendered = super().render_message(promoted, ctx)
            return _mask_think_open(rendered, self.tokenizer)

        # Structured turns: same tokens as True, then move the think span off loss.
        # No-structure: leave the official empty ``<think>\n\n</think>\n\n`` in
        # the header. Do not _inject_empty_thinking here (that is True-only).
        rendered = super().render_message(promoted, ctx)
        if not has_structure:
            return rendered
        return _mask_think_span(
            rendered, self.tokenizer, self._rendered_think_block(promoted)
        )


def register_qwen35_renderers() -> None:
    """Register the MinT Qwen3.5 renderer.

    The factory passes ``strip_thinking_from_history=False`` so prior-turn think
    blocks stay in the sequence for training, validation, and serving.
    """
    register_renderer(
        QWEN35_RENDERER,
        lambda tokenizer, image_processor=None: Qwen35ReasoningRenderer(
            tokenizer,
            image_processor=image_processor,
            strip_thinking_from_history=False,
        ),
    )


register_qwen35_renderers()


__all__ = [
    "QWEN35_RENDERER",
    "Qwen35ReasoningRenderer",
    "register_qwen35_renderers",
]
