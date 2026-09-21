"""MinT Qwen3.5 renderer: one think-block contract, optional CoT loss.

An assistant turn may carry reasoning in exactly one of:

* ``reasoning_content``
* a ``ThinkingPart`` in ``content``

A serving leftover ``REASON</think>ANSWER`` (no open tag before the first
close; empty ``REASON`` means the model closed without thinking) is
folded into a ThinkingPart. String and text-part ``content`` share this
rule. Other literal tags are visible text. Leftover-None list encodings
(bare strings, typeless ``{"text": ...}``) become TextParts so the
upstream VL preprocessor can read ``type``; copied complete blocks stay
answers. See ``README.md``.

``supervise_reasoning=True`` (default) puts loss on the generated think body,
``</think>``, and the visible answer. The opening ``<think>\\n`` is prefilled by
the generation prompt and stays in the unsupervised header.
``supervise_reasoning=False`` moves the whole ``<think>...</think>`` span into
the header. Structured turns keep the same token sequence as ``True``.

Intentional HF asymmetry (not a sequence-parity bug). The official Qwen3.5
template prefills generation with ``<think>\\n`` (token 198) but renders a
completed empty CoT as ``<think>\\n\\n</think>\\n\\n`` (token 271). ``True``
follows the serving prefix via an empty ThinkingPart. ``False`` with no
structure uses that completed-empty header only when visible answer text
has no ``<think>`` / ``</think>`` (string ``content`` and text parts;
this class overrides cookbook ``_assistant_header_suffix`` so literal
tags do not get a second block). Do not inject on the ``False`` path
just to make ``to_ints()`` match.

The upstream Qwen3.5 renderer only reads ThinkingPart and always supervises the
think block. This subclass folds ``reasoning_content`` into a ThinkingPart, then
relocates already-encoded tokens between ``header`` and ``output``. False-path
mask locates the think span on that merged encoding; do not assume
``encode(think_block)`` is a token prefix of the assistant output.
"""

from __future__ import annotations

from typing import Any

import tinker
from tinker_cookbook.exceptions import RendererError
from tinker_cookbook.renderers import register_renderer
from tinker_cookbook.renderers.base import RenderContext, RenderedMessage
from tinker_cookbook.renderers.qwen3_5 import Qwen3_5Renderer

from mint.reasoning_source import split_serving_fragment, visible_has_think_markup

QWEN35_RENDERER = "MindLab/qwen35"

_THINK_CLOSE = "</think>"


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


def _standardize_visible_parts(content: Any) -> Any:
    """Turn leftover-None list encodings into cookbook TextParts.

    The upstream VL preprocessor indexes ``part["type"]``. Bare strings and
    typeless ``{"text": ...}`` are still visible answers after leftover
    returns None (copied complete blocks, mentions). Do not reread tags
    as reasoning here.
    """
    if not isinstance(content, list):
        return content
    parts: list[Any] = []
    for part in content:
        if isinstance(part, str):
            parts.append({"type": "text", "text": part})
            continue
        if (
            isinstance(part, dict)
            and part.get("type") is None
            and isinstance(part.get("text"), str)
        ):
            parts.append({**part, "type": "text"})
            continue
        parts.append(part)
    return parts


def _normalize_assistant_message(message: dict[str, Any]) -> tuple[str, bool, dict[str, Any]]:
    """Normalize an assistant message to at most one ThinkingPart.

    Exactly one of ``reasoning_content`` or a ThinkingPart may be present; both
    at once is an error. A serving leftover in string or text-part ``content``
    is promoted to a field first. Other literal tags stay as visible text.
    Leftover-None list encodings are coerced to TextParts. Returns
    ``(reasoning, has_structure, promoted)``. ``promoted`` never carries
    ``reasoning_content``. When ``has_structure`` is true, ``content`` is a
    list starting with a ThinkingPart.
    """
    promoted = {key: value for key, value in message.items() if key != "reasoning_content"}
    content = _standardize_visible_parts(promoted.get("content", ""))
    promoted["content"] = content
    field = _reasoning_field(message)
    part_texts = _thinking_part_texts(content)

    if field is not None and part_texts:
        raise RendererError(
            "use either reasoning_content or ThinkingPart for Qwen3.5 reasoning, not both"
        )

    if field is not None:
        return field, True, {**promoted, "content": _parts_with_thinking(field, content)}
    if part_texts:
        return "".join(part_texts), True, promoted
    fragment = split_serving_fragment(content)
    if fragment is not None:
        reason, visible = fragment
        return reason, True, {**promoted, "content": _parts_with_thinking(reason, visible)}
    return "", False, promoted


def _has_thinking_part(content: Any) -> bool:
    """Whether ``content`` lists a thinking part. Payload is not validated.

    The False-path suffix only asks "is a structure block already present?"
    A malformed part still suppresses the official empty scaffold.
    ``render_message`` validates via ``_normalize_assistant_message`` before
    any mask runs.
    """
    if not isinstance(content, list):
        return False
    return any(
        isinstance(part, dict) and (part.get("type") == "thinking" or "thinking" in part)
        for part in content
    )


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
    # Production only calls this when normalize reported no structure.
    # Still skip if a thinking part is already listed (helper used in tests).
    promoted = dict(message)
    content = promoted.get("content", "")
    if _has_thinking_part(content):
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


def _decode_token_ids(tokenizer: Any, token_ids: list[int]) -> str:
    decoded = tokenizer.decode(token_ids)
    return decoded if isinstance(decoded, str) else str(decoded)


def _token_count_covering_prefix(
    output_tokens: list[int], text: str, tokenizer: Any
) -> int:
    """Shortest leading token count whose decode covers ``text``.

    ``encode(text)`` is usually that prefix. BPE may merge the last piece
    of ``text`` with the following answer (think-block trailing ``\\n\\n``
    plus leftover-visible leading ``\\n\\n`` becomes one 4-newline token on
    Qwen3.5). Then take the shortest prefix of the already-merged output
    whose decode starts with ``text``. The merged token goes with the think
    span so reasoning stays off loss.
    """
    locate_error = (
        "could not locate the rendered think block to mask; the encoded "
        "<think>...</think> prefix does not match the assistant output"
    )
    if not text:
        raise RendererError(locate_error)
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if encoded and output_tokens[: len(encoded)] == encoded:
        return len(encoded)

    if not output_tokens or not _decode_token_ids(tokenizer, output_tokens).startswith(
        text
    ):
        raise RendererError(locate_error)

    start = max(1, len(encoded) - 2) if encoded else 1
    for k in range(start, len(output_tokens) + 1):
        if _decode_token_ids(tokenizer, output_tokens[:k]).startswith(text):
            return k
    for k in range(1, start):
        if _decode_token_ids(tokenizer, output_tokens[:k]).startswith(text):
            return k
    raise RendererError(locate_error)


def _mask_think_span(
    rendered: RenderedMessage,
    tokenizer: Any,
    think_block: str | None = None,
) -> RenderedMessage:
    """Move the leading ``<think>...</think>`` span from ``output`` into ``header``.

    When ``think_block`` is the exact formatted think text just rendered, the
    split is the shortest already-encoded prefix that covers that text — so a
    ``</think>`` inside the reasoning body is not treated as the end of the
    block, and BPE merges at the think/answer boundary do not have to match
    ``encode(think_block)``. Without ``think_block``, falls back to the first
    ``</think>`` token (helper tests). Image chunks or a missing boundary
    leave ``rendered`` unchanged.
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
        split = _token_count_covering_prefix(output_tokens, think_block, tokenizer)
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

    def _assistant_header_suffix(self, message: dict[str, Any], ctx: RenderContext) -> str:
        """Park the official empty think block when none is already on the wire.

        MinT override of the cookbook suffix. Cookbook would emit the
        official empty block whenever there is no ThinkingPart. We also
        skip when leftover already created structure, and when leftover
        aborted but visible text still has ``<think>`` or ``</think>``.
        Role is checked here too (cookbook only calls this for assistant).
        """
        if message.get("role") != "assistant":
            return ""
        if ctx.idx <= getattr(ctx, "last_user_index", -1):
            return ""
        content = message.get("content", "")
        if _has_thinking_part(content):
            return ""
        # Leftover already ran. Skip the official empty scaffold when
        # leftover returned None but visible text still has a think tag
        # (mention, copied block, or leftover-abort such as text+image).
        # Check both tags: </think> is not a substring of <think>, so an
        # open-only guard re-stacks on leftover-abort ``</think>ANSWER``.
        if visible_has_think_markup(content):
            return ""
        return "<think>\n\n</think>\n\n"

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

        # Normalize first so a non-string ThinkingPart raises here. True-path
        # ``_mask_think_open`` never sees a malformed payload; the suffix
        # predicate (``_has_thinking_part``) is the only unvalidated check.
        _, has_structure, promoted = _normalize_assistant_message(message)

        if self.supervise_reasoning:
            # Empty ThinkingPart so </think> is in output; opener stays in header.
            if not has_structure and self._is_trained_answer_turn(message, ctx):
                promoted = _inject_empty_thinking(promoted)
            rendered = super().render_message(promoted, ctx)
            return _mask_think_open(rendered, self.tokenizer)

        # Structured turns: same tokens as True, then move the think span off
        # loss. That mask is the structured CoT, not a second pass over
        # literal tags in the visible answer. No-structure: official empty
        # header, unless visible text already has a think tag. Do not inject.
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
