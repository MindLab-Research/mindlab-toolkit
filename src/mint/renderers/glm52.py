"""Tinker Cookbook renderer for ``zai-org/GLM-5.2``.

The wire format is intentionally kept byte-compatible with the pinned
GLM-5.2 SFT template in ``MindLab-Research/agent-model-training-mono``.  In
particular, role tokens terminate the preceding assistant action, the
``<think>`` prompt scaffold is not trained, and tool calls use GLM's XML
format rather than OpenAI JSON blocks.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Literal, cast

import tinker
import torch
from tinker_cookbook.exceptions import RendererError
from tinker_cookbook.renderers import (
    Message,
    ParseTermination,
    RenderContext,
    Renderer,
    ToolCall,
    ToolSpec,
    TrainOnWhat,
    register_renderer,
)
from tinker_cookbook.renderers.base import RenderedMessage, UnparsedToolCall
from tinker_cookbook.tokenizer_utils import Tokenizer

from mint.reasoning_effort import GLM52_REASONING_EFFORTS, GLM53_REASONING_EFFORTS
from mint.reasoning_source import split_serving_fragment

GLM52_RENDERER = "MindLab/glm52"
GLM53_RENDERER = "MindLab/glm53"
GLM52_HIGH_REASONING_RENDERER = "MindLab/glm52_high_reasoning"
GLM52_DISABLE_THINKING_RENDERER = "MindLab/glm52_disable_thinking"
# Union kept for callers that only need "is this spelling one of the families".
# A renderer accepts only its own ``reasoning_efforts``.
REASONING_EFFORTS = tuple(dict.fromkeys((*GLM52_REASONING_EFFORTS, *GLM53_REASONING_EFFORTS)))

_GMASK_SOP = "[gMASK]<sop>"
_SYSTEM = "<|system|>"
_USER = "<|user|>"
_ASSISTANT = "<|assistant|>"
_OBSERVATION = "<|observation|>"
_END_OF_TEXT = "<|endoftext|>"
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"

_STOP_TOKENS = (_END_OF_TEXT, _USER, _OBSERVATION)
_KNOWN_ROLES = ("system", "user", "assistant", "tool")
_MEDIA_TYPES = {
    "image": "image",
    "image_url": "image",
    "video": "video",
    "video_url": "video",
    "audio": "audio",
    "audio_url": "audio",
    "input_audio": "audio",
}

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_ARG_RE = re.compile(
    r"<arg_key>(?P<key>.*?)</arg_key>\s*"
    r"<arg_value>(?P<value>.*?)</arg_value>",
    re.DOTALL,
)

# Tinker Cookbook 0.4.0/0.4.1 (the releases compatible with this repository's
# tinker==0.22.0 pin) did not yet put ``next_message`` on RenderContext.  A
# context variable supplies the same look-ahead without mutable renderer state,
# so concurrent rollouts remain isolated while newer Cookbook releases use the
# native field directly.
_ACTIVE_MESSAGES: ContextVar[tuple[Message, ...] | None] = ContextVar(
    "mint_glm52_active_messages", default=None
)
# Set only for one render. Unset means the effort chosen when the renderer was built.
_RENDER_EFFORT: ContextVar[str | None] = ContextVar("mint_glm_render_effort", default=None)
_TOOL_CONTEXT_KEY = "_mint_glm52_tools"
_ACTIVE_TOOL_SPECS: ContextVar[tuple[dict[str, Any], ...] | None] = ContextVar(
    "mint_glm52_active_tool_specs", default=None
)


def _encode_ids(tokenizer: Tokenizer, text: str) -> list[int]:
    if not text:
        return []
    return list(tokenizer.encode(text, add_special_tokens=False))


def _bpe_ids_without_added_tokens(tokenizer: Tokenizer, text: str) -> list[int] | None:
    """Byte-level BPE of ``text``, skipping the added-token matcher.

    Hugging Face ``split_special_tokens`` only splits tokens flagged
    ``special: true``. GLM also registers control strings such as ``<think>``,
    ``<tool_call>``, and ``<|user|>`` as added tokens with ``special: false``,
    so that flag still emits them as one id. There is no tokenizer switch for
    this case. Running the normalizer, pre-tokenizer, and BPE model directly
    is the same path transformers uses to encode text that must not match an
    added token.

    Pretokenizer pieces may contain the byte-level space marker (``Ġ``).
    ``model.tokenize`` accepts those pieces. Text with no added-token spelling
    matches ``tokenizer.encode``. A spelling such as ``<|user|>`` is split, and
    the space before it merges into the following characters (``Ġ<|`` rather
    than a separate space plus one added-token id). That split is intentional.
    """
    backend = getattr(tokenizer, "backend_tokenizer", None)
    model = getattr(backend, "model", None) if backend is not None else None
    tokenize = getattr(model, "tokenize", None)
    if backend is None or tokenize is None:
        return None
    normalizer = getattr(backend, "normalizer", None)
    if normalizer is not None:
        text = normalizer.normalize_str(text)
    pre_tokenizer = getattr(backend, "pre_tokenizer", None)
    if pre_tokenizer is not None:
        pieces = [piece for piece, _ in pre_tokenizer.pre_tokenize_str(text)]
    else:
        pieces = [text]
    return [int(token.id) for piece in pieces for token in tokenize(piece)]


def _encode_body_ids(tokenizer: Tokenizer, text: str) -> list[int]:
    """Encode data text so every added token is an ordinary BPE piece.

    Role markers, the structural think tags, and tool wrappers are not passed
    here. The renderer encodes those strings on their own with
    :func:`_encode_ids`. A tokenizer without ``backend_tokenizer`` falls back
    to :func:`_encode_ids`; the GLM fast tokenizer has that backend.
    """
    if not text:
        return []
    piece = _bpe_ids_without_added_tokens(tokenizer, text)
    if piece is None:
        return _encode_ids(tokenizer, text)
    return piece


def _encoded_chunk(tokenizer: Tokenizer, text: str) -> tinker.types.EncodedTextChunk:
    return tinker.types.EncodedTextChunk(tokens=_encode_ids(tokenizer, text))


def _chunk_from_ids(token_ids: list[int]) -> tinker.types.EncodedTextChunk | None:
    if not token_ids:
        return None
    return tinker.types.EncodedTextChunk(tokens=token_ids)


def _single_token_id(tokenizer: Tokenizer, token: str) -> int:
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    if len(token_ids) != 1:
        raise RendererError(
            f"GLM-5.2 special token {token!r} must encode to one token; "
            f"got {token_ids!r}. Use the zai-org/GLM-5.2 tokenizer."
        )
    return int(token_ids[0])


def _media_reminder(media_type: str) -> str:
    return (
        f"<reminder>You are unable to process this {media_type} because you "
        "don't have multi-modal input ability. Try different methods.</reminder>"
    )


def _iter_visible_pieces(
    content: Any, *, allow_thinking: bool = False
) -> list[str | Mapping[str, Any]]:
    """Split visible content into text strings and untouched media parts."""
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        raise RendererError(
            "GLM-5.2 message content must be a string or a list of content parts"
        )

    pieces: list[str | Mapping[str, Any]] = []
    for part in content:
        if isinstance(part, str):
            pieces.append(part)
            continue
        if not isinstance(part, Mapping):
            raise RendererError(
                f"GLM-5.2 content parts must be mappings or strings, got {type(part).__name__}"
            )
        part_type = part.get("type")
        if part_type == "text":
            text = part.get("text")
            if not isinstance(text, str):
                raise RendererError(
                    "GLM-5.2 text content parts require a string 'text' field"
                )
            pieces.append(text)
        elif part_type == "thinking":
            if allow_thinking:
                continue
            raise RendererError("thinking content is only valid on assistant messages")
        elif isinstance(part_type, str) and part_type in _MEDIA_TYPES:
            pieces.append(part)
        else:
            raise RendererError(f"unsupported GLM-5.2 content part type: {part_type!r}")
    return pieces


def _visible_segments(
    content: Any,
    *,
    allow_thinking: bool = False,
    render_media: Any | None = None,
) -> list[tuple[str, bool]]:
    """Return ``(text, atomic)`` pieces. Media placeholders stay one encode."""
    write_media = render_media or (lambda kind, _part: _media_reminder(kind))
    segments: list[tuple[str, bool]] = []
    for piece in _iter_visible_pieces(content, allow_thinking=allow_thinking):
        if isinstance(piece, str):
            segments.append((piece, False))
        else:
            placeholder = write_media(_MEDIA_TYPES[str(piece.get("type"))], piece)
            segments.append((placeholder, True))
    return segments


def _strip_visible_edges(segments: list[tuple[str, bool]]) -> list[tuple[str, bool]]:
    """Match ``"".join(texts).strip()``, including whitespace on a media piece.

    A media placeholder stays its own atomic segment. Only whitespace at the
    ends of the concatenated string is removed, so padding around ``<|image|>``
    does not survive the official visible-text strip.
    """
    if not segments:
        return []
    joined = "".join(text for text, _atomic in segments)
    stripped = joined.strip()
    if not stripped:
        return []
    start = len(joined) - len(joined.lstrip())
    end = start + len(stripped)
    kept: list[tuple[str, bool]] = []
    position = 0
    for text, atomic in segments:
        segment_end = position + len(text)
        left = max(position, start)
        right = min(segment_end, end)
        if right > left:
            kept.append((text[left - position : right - position], atomic))
        position = segment_end
    return kept


def _encode_visible_ids(
    tokenizer: Tokenizer,
    content: Any,
    *,
    allow_thinking: bool = False,
    render_media: Any | None = None,
    strip: bool = False,
) -> list[int]:
    """Body-BPE ordinary text. Renderer-authored media placeholders stay atomic.

    Adjacent text pieces are joined before BPE so a split content list matches
    one encode of the concatenation. A placeholder such as ``<|image|>`` is
    encoded on its own.
    """
    segments = _visible_segments(
        content, allow_thinking=allow_thinking, render_media=render_media
    )
    if strip:
        segments = _strip_visible_edges(segments)
    ids: list[int] = []
    body: list[str] = []

    def flush() -> None:
        text = "".join(body)
        body.clear()
        if text:
            ids.extend(_encode_body_ids(tokenizer, text))

    for text, atomic in segments:
        if atomic:
            flush()
            if text:
                ids.extend(_encode_ids(tokenizer, text))
        else:
            body.append(text)
    flush()
    return ids


def _visible_text(
    content: Any,
    *,
    allow_thinking: bool = False,
    render_media: Any | None = None,
) -> str:
    """Mirror the GLM-5.2 template's ``visible_text`` macro."""
    return "".join(
        text
        for text, _atomic in _visible_segments(
            content, allow_thinking=allow_thinking, render_media=render_media
        )
    )


def _assistant_content_raw(message: Message) -> tuple[bool, str, Any]:
    """Return ``(has_reasoning_signal, reasoning, visible)`` without flattening media.

    Visible is the leftover remainder or the original ``content`` (string or
    parts). Callers that need 5.2 reminder text run it through ``_visible_text``.
    """
    raw_message = cast(dict[str, Any], message)
    content = message["content"]

    if "reasoning_content" in raw_message:
        reasoning = raw_message["reasoning_content"]
        if not isinstance(reasoning, str):
            raise RendererError("GLM-5.2 reasoning_content must be a string")
        if isinstance(content, list) and any(
            isinstance(part, Mapping) and part.get("type") == "thinking"
            for part in content
        ):
            raise RendererError(
                "use either reasoning_content or ThinkingPart for GLM-5.2 reasoning, not both"
            )
        return True, reasoning, content

    if isinstance(content, str):
        fragment = split_serving_fragment(content)
        if fragment is not None:
            # Leftover, including leftover+mention
            # ("REASON</think>see <think>x</think>"). Visible is not
            # stripped here; render_message strips once at write.
            return True, fragment[0], fragment[1]
        return False, "", content

    reasoning_parts: list[str] = []
    for part in content:
        if isinstance(part, Mapping) and part.get("type") == "thinking":
            reasoning = part.get("thinking")
            if not isinstance(reasoning, str):
                raise RendererError(
                    "GLM-5.2 ThinkingPart requires a string 'thinking' field"
                )
            reasoning_parts.append(reasoning)
    if reasoning_parts:
        return True, "".join(reasoning_parts), content
    fragment = split_serving_fragment(content)
    if fragment is not None:
        return True, fragment[0], fragment[1]
    return False, "", content


def _assistant_content(message: Message) -> tuple[bool, str, str]:
    """Return ``(has_reasoning_signal, reasoning, visible_content)``.

    Tinker represents reasoning with ``ThinkingPart``.  The legacy
    ``reasoning_content`` key is also accepted so existing GLM-5.2 SFT records
    can be handed to the renderer without a preprocessing pass.

    Literal tags in visible ``content`` stay as answer text once a field or
    ThinkingPart is present. A serving leftover ``REASON</think>ANSWER`` with
    neither is promoted to ``reasoning_content`` (see README). Empty
    ``REASON`` (``</think>ANSWER``) is the no-think serving leftover.
    """
    has_reasoning, reasoning, visible = _assistant_content_raw(message)
    if isinstance(visible, str):
        return has_reasoning, reasoning, visible
    return has_reasoning, reasoning, _visible_text(visible, allow_thinking=True)


def _tool_arguments(tool_call: ToolCall) -> dict[str, Any]:
    raw_arguments = tool_call.function.arguments
    if raw_arguments == "":
        return {}
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise RendererError(
            f"GLM-5.2 tool {tool_call.function.name!r} arguments must be a JSON object: {exc}"
        ) from exc
    if not isinstance(arguments, dict):
        raise RendererError(
            f"GLM-5.2 tool {tool_call.function.name!r} arguments must decode to an object"
        )
    return arguments


def _encode_tool_calls(tokenizer: Tokenizer, tool_calls: list[ToolCall]) -> list[int]:
    """Keep tool markup atomic and encode names, keys, and values as data.

    Non-string values are ``json.dumps``'d first and still take the body path,
    so a ``<tool_call>`` inside a nested value is not a second tool-call token.
    """
    ids: list[int] = []
    for tool_call in tool_calls:
        arguments = _tool_arguments(tool_call)
        ids.extend(_encode_ids(tokenizer, "<tool_call>"))
        ids.extend(_encode_body_ids(tokenizer, tool_call.function.name))
        for key, value in arguments.items():
            rendered_value = (
                value
                if isinstance(value, str)
                else json.dumps(value, ensure_ascii=False)
            )
            ids.extend(_encode_ids(tokenizer, "<arg_key>"))
            ids.extend(_encode_body_ids(tokenizer, str(key)))
            ids.extend(_encode_ids(tokenizer, "</arg_key><arg_value>"))
            ids.extend(_encode_body_ids(tokenizer, rendered_value))
            ids.extend(_encode_ids(tokenizer, "</arg_value>"))
        ids.extend(_encode_ids(tokenizer, "</tool_call>"))
    return ids


def _normalize_tool_spec(raw_tool: ToolSpec | Mapping[str, Any]) -> dict[str, Any]:
    tool = cast(dict[str, Any], raw_tool)
    if "function" in tool and isinstance(tool["function"], Mapping):
        tool = cast(dict[str, Any], tool["function"])
    return dict(tool)


def _serializable_tool_spec(tool: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in tool.items()
        if key not in ("defer_loading", "strict")
    }


def _tool_specs_from_messages(
    messages: list[Message],
) -> tuple[dict[str, Any], ...] | None:
    for message in messages:
        raw_specs = cast(dict[str, Any], message).get(_TOOL_CONTEXT_KEY)
        if raw_specs is None:
            continue
        if not isinstance(raw_specs, (list, tuple)) or not all(
            isinstance(tool, Mapping) for tool in raw_specs
        ):
            raise RendererError("invalid GLM-5.2 tool context on prefix message")
        return tuple(dict(tool) for tool in raw_specs)
    return None


def _tool_reference_body(
    content: list[Any], tool_specs: tuple[dict[str, Any], ...] | None
) -> str:
    if tool_specs is None:
        raise RendererError(
            "GLM-5.2 tool_reference results require tool definitions from "
            "create_conversation_prefix_with_tools()"
        )

    tool_lines: list[str] = []
    for reference in content:
        if not isinstance(reference, Mapping):
            continue
        name = reference.get("name")
        for tool in tool_specs:
            if tool.get("name") == name:
                tool_lines.append(
                    json.dumps(_serializable_tool_spec(tool), ensure_ascii=False)
                )
    return "".join(f"{line}\n" for line in tool_lines)


def _encode_tool_response(
    tokenizer: Tokenizer,
    content: Any,
    tool_specs: tuple[dict[str, Any], ...] | None,
    render_media: Any | None = None,
) -> list[int]:
    """Keep ``<tool_response>`` wrappers atomic and encode result text as data.

    Media placeholders inside a non-string result stay one token each. Result
    JSON is not the format template, so schema text stays on the body path.
    """
    if isinstance(content, str):
        return (
            _encode_ids(tokenizer, "<tool_response>")
            + _encode_body_ids(tokenizer, content)
            + _encode_ids(tokenizer, "</tool_response>")
        )
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, Mapping) and first.get("type") == "tool_reference":
            return (
                _encode_ids(tokenizer, "<tool_response><tools>\n")
                + _encode_body_ids(tokenizer, _tool_reference_body(content, tool_specs))
                + _encode_ids(tokenizer, "</tools></tool_response>")
            )
        if isinstance(first, Mapping) and "output" in first:
            ids: list[int] = []
            for item in content:
                if not isinstance(item, Mapping) or not isinstance(
                    item.get("output"), str
                ):
                    raise RendererError(
                        "structured GLM-5.2 tool results require string 'output' fields"
                    )
                ids.extend(_encode_ids(tokenizer, "<tool_response>"))
                ids.extend(_encode_body_ids(tokenizer, item["output"]))
                ids.extend(_encode_ids(tokenizer, "</tool_response>"))
            return ids
    return (
        _encode_ids(tokenizer, "<tool_response>")
        + _encode_visible_ids(tokenizer, content, render_media=render_media)
        + _encode_ids(tokenizer, "</tool_response>")
    )


def _parse_tool_call(raw_block: str, body: str) -> ToolCall | UnparsedToolCall:
    first_arg = body.find("<arg_key>")
    name = (body if first_arg < 0 else body[:first_arg]).strip()
    raw_args = "" if first_arg < 0 else body[first_arg:]
    if not name:
        return UnparsedToolCall(raw_text=raw_block, error="missing GLM-5.2 tool name")

    arguments: dict[str, str] = {}
    consumed: list[tuple[int, int]] = []
    for match in _ARG_RE.finditer(raw_args):
        key = match.group("key").strip()
        if not key:
            return UnparsedToolCall(
                raw_text=raw_block, error="empty GLM-5.2 argument key"
            )
        arguments[key] = match.group("value")
        consumed.append(match.span())

    remainder = raw_args
    for start, end in reversed(consumed):
        remainder = remainder[:start] + remainder[end:]
    if remainder.strip():
        return UnparsedToolCall(
            raw_text=raw_block,
            error="malformed GLM-5.2 <arg_key>/<arg_value> sequence",
        )

    return ToolCall(
        function=ToolCall.FunctionBody(
            name=name,
            arguments=json.dumps(arguments, ensure_ascii=False),
        )
    )


def _parse_tool_calls(
    content: str,
) -> tuple[str, list[ToolCall], list[UnparsedToolCall]]:
    tool_calls: list[ToolCall] = []
    unparsed: list[UnparsedToolCall] = []
    visible_parts: list[str] = []
    position = 0

    for match in _TOOL_CALL_RE.finditer(content):
        visible_parts.append(content[position : match.start()])
        parsed = _parse_tool_call(match.group(0), match.group(1))
        if isinstance(parsed, ToolCall):
            tool_calls.append(parsed)
        else:
            unparsed.append(parsed)
        position = match.end()
    visible_parts.append(content[position:])

    dangling = content.find("<tool_call>", position)
    if dangling >= 0:
        raw_text = content[dangling:]
        visible_parts[-1] = content[position:dangling]
        unparsed.append(
            UnparsedToolCall(
                raw_text=raw_text,
                error="unterminated GLM-5.2 <tool_call> block",
            )
        )
    return "".join(visible_parts), tool_calls, unparsed


class GLM52Renderer(Renderer):
    """Renderer for GLM-5.2 with thinking enabled.

    ``reasoning_effort`` only changes the system line ``Reasoning Effort:
    {Max|High}``. Default ``max``. GLM-5.3 adds ``Low`` on
    ``GLM53Renderer``. Historical reasoning is preserved, which gives this
    renderer Tinker's sequence extension property.

    ``supervise_reasoning=False`` renders an identical token sequence but moves
    the assistant ``<think>{reasoning}</think>`` block into the header. Modes
    that leave headers unsupervised (the usual assistant-only settings) then
    train only the visible answer/tool calls. ``ALL_TOKENS`` still applies loss
    to headers, including that think span. Orthogonal to ``reasoning_effort``
    and ``enable_thinking``.
    """

    supports_streaming = False
    # Official GLM-5.2 template: explicit high, otherwise max. GLM-5.3 overrides this.
    reasoning_efforts: tuple[str, ...] = GLM52_REASONING_EFFORTS

    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        reasoning_effort: str = "max",
        enable_thinking: bool = True,
        supervise_reasoning: bool = True,
    ):
        super().__init__(tokenizer)
        if reasoning_effort not in self.reasoning_efforts:
            allowed = ", ".join(self.reasoning_efforts)
            raise ValueError(
                f"{type(self).__name__} reasoning_effort must be one of {allowed}"
            )
        self.reasoning_effort = reasoning_effort
        self.enable_thinking = bool(enable_thinking)
        self.disables_thinking = not self.enable_thinking
        self.supervise_reasoning = bool(supervise_reasoning)

        self._prefix_tokens_by_effort = {
            effort: self._encode_effort_prefix(effort) for effort in self.reasoning_efforts
        }
        self._prefix_tokens = list(self._prefix_tokens_by_effort[self.reasoning_effort])
        self._stop_token_ids = {
            token: _single_token_id(tokenizer, token) for token in _STOP_TOKENS
        }

    def _encode_effort_prefix(self, effort: str) -> list[int]:
        prefix = _GMASK_SOP
        if self.enable_thinking:
            prefix += f"{_SYSTEM}Reasoning Effort: {effort.capitalize()}"
        return self.tokenizer.encode(prefix, add_special_tokens=False)

    @contextmanager
    def reasoning_effort_scope(self, effort: str) -> Iterator[None]:
        """Select this effort's prefix for one render, then restore the default.

        The allowed prefixes are encoded once. This does not replace the renderer.
        """
        if effort not in self._prefix_tokens_by_effort:
            allowed = ", ".join(self.reasoning_efforts)
            raise ValueError(
                f"{type(self).__name__} reasoning_effort must be one of {allowed}"
            )
        token = _RENDER_EFFORT.set(effort)
        try:
            yield
        finally:
            _RENDER_EFFORT.reset(token)

    @property
    def _bos_tokens(self) -> list[int]:
        effort = _RENDER_EFFORT.get()
        if effort not in self._prefix_tokens_by_effort:
            effort = self.reasoning_effort
        return list(self._prefix_tokens_by_effort[effort])

    @property
    def has_extension_property(self) -> bool:
        return True

    def get_stop_sequences(self) -> list[int]:
        return [self._stop_token_ids[token] for token in _STOP_TOKENS]

    def _assistant_header(self) -> str:
        if self.enable_thinking:
            return f"{_ASSISTANT}{_THINK_OPEN}"
        return f"{_ASSISTANT}{_THINK_OPEN}{_THINK_CLOSE}"

    def _media_placeholder(self, kind: str, part: Mapping[str, Any]) -> str:
        del part
        return _media_reminder(kind)

    def _flatten_visible(self, content: Any, *, allow_thinking: bool = False) -> str:
        return _visible_text(
            content,
            allow_thinking=allow_thinking,
            render_media=self._media_placeholder,
        )

    def _boundary_after_assistant(self, ctx: RenderContext) -> str:
        next_message = getattr(ctx, "next_message", None)
        if next_message is None:
            active_messages = _ACTIVE_MESSAGES.get()
            if active_messages is not None and ctx.idx + 1 < len(active_messages):
                next_message = active_messages[ctx.idx + 1]
        if next_message is None:
            return ""
        next_role = next_message["role"]
        if next_role == "user":
            return _USER
        if next_role == "tool":
            return _OBSERVATION
        return ""

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        role = message["role"]
        if role not in _KNOWN_ROLES:
            raise RendererError(
                f"unsupported GLM-5.2 role {role!r}; expected one of {_KNOWN_ROLES!r}"
            )

        if role == "assistant":
            has_reasoning, reasoning, visible = _assistant_content_raw(message)
            if not self.enable_thinking and has_reasoning:
                raise RendererError(
                    "GLM-5.2 thinking is disabled, but the assistant message contains "
                    "ThinkingPart/reasoning_content"
                )
            # Structural markers stay one token each. Text the row actually
            # contains — reasoning, the visible answer, tool names, and
            # argument values — is ordinary BPE, including any added-token
            # spelling such as <think> or <|user|>.
            header_ids = _encode_ids(self.tokenizer, self._assistant_header())
            output_ids: list[int] = []
            if self.enable_thinking:
                reasoning_ids = _encode_body_ids(self.tokenizer, reasoning)
                close_ids = _encode_ids(self.tokenizer, _THINK_CLOSE)
                if self.supervise_reasoning:
                    output_ids.extend(reasoning_ids)
                    output_ids.extend(close_ids)
                else:
                    # Keep the chain-of-thought in the sequence; put it in the
                    # header so modes that skip headers do not train it.
                    header_ids.extend(reasoning_ids)
                    header_ids.extend(close_ids)
            # Official template strips visible_text. Same strip for field,
            # ThinkingPart, leftover, and plain answers — do it here, not
            # in _assistant_content, so tests that decode the wire see the
            # stripped form for every writing. Media placeholders stay atomic.
            output_ids.extend(
                _encode_visible_ids(
                    self.tokenizer,
                    visible,
                    allow_thinking=True,
                    render_media=self._media_placeholder,
                    strip=True,
                )
            )
            tool_calls = list(message.get("tool_calls", []))
            if tool_calls:
                output_ids.extend(_encode_tool_calls(self.tokenizer, tool_calls))
            output_ids.extend(
                _encode_ids(self.tokenizer, self._boundary_after_assistant(ctx))
            )
            header = _chunk_from_ids(header_ids)
            output = [chunk] if (chunk := _chunk_from_ids(output_ids)) is not None else []
            return RenderedMessage(header=header, output=output)

        prev_role = ctx.prev_message["role"] if ctx.prev_message is not None else None
        # Dataset text is body BPE. The tool declaration built by
        # create_conversation_prefix_with_tools is the exception: one
        # _encode_ids call, so the official format tags (<tool_call>,
        # <arg_key>, <arg_value>) stay one token each. Do not switch that
        # message to _encode_body_ids. Schema descriptions are ordinary prose
        # and match either encoder.
        if role == "user":
            header_text = "" if prev_role == "assistant" else _USER
            output_ids = _encode_visible_ids(
                self.tokenizer,
                message["content"],
                render_media=self._media_placeholder,
            )
        elif role == "system":
            header_text = _SYSTEM
            # The tool declaration is one _encode_ids call so format tags stay
            # atomic. Ordinary system text, including media placeholders, uses
            # the visible encoder.
            if cast(dict[str, Any], message).get(_TOOL_CONTEXT_KEY) is not None:
                # One encode of the whole declaration keeps format tags atomic.
                # render_media still runs, so a media part is not the 5.2 reminder
                # when a subclass replaces the placeholder.
                output_ids = _encode_ids(
                    self.tokenizer,
                    _visible_text(
                        message["content"], render_media=self._media_placeholder
                    ),
                )
            else:
                output_ids = _encode_visible_ids(
                    self.tokenizer,
                    message["content"],
                    render_media=self._media_placeholder,
                )
        else:
            header_text = "" if prev_role in ("assistant", "tool") else _OBSERVATION
            output_ids = _encode_tool_response(
                self.tokenizer,
                message["content"],
                _ACTIVE_TOOL_SPECS.get(),
                render_media=self._media_placeholder,
            )
        header = _encoded_chunk(self.tokenizer, header_text) if header_text else None
        output = [chunk] if (chunk := _chunk_from_ids(output_ids)) is not None else []
        return RenderedMessage(header=header, output=output)

    def _get_generation_suffix(self, role: str, ctx: RenderContext) -> list[int]:
        del ctx
        if role == "assistant":
            suffix = self._assistant_header()
        elif role == "user":
            suffix = _USER
        elif role == "system":
            suffix = _SYSTEM
        elif role == "tool":
            suffix = _OBSERVATION
        else:
            raise RendererError(f"unsupported GLM-5.2 generation role: {role!r}")
        return self.tokenizer.encode(suffix, add_special_tokens=False)

    def build_generation_prompt(
        self,
        messages: list[Message],
        role: str = "assistant",
        prefill: str | None = None,
    ) -> tinker.ModelInput:
        tool_specs = _tool_specs_from_messages(messages)
        active_messages = _ACTIVE_MESSAGES.set(tuple(messages))
        active_tools = _ACTIVE_TOOL_SPECS.set(tool_specs)
        try:
            return super().build_generation_prompt(messages, role=role, prefill=prefill)
        finally:
            _ACTIVE_TOOL_SPECS.reset(active_tools)
            _ACTIVE_MESSAGES.reset(active_messages)

    def build_supervised_example(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        tool_specs = _tool_specs_from_messages(messages)
        active_messages = _ACTIVE_MESSAGES.set(tuple(messages))
        active_tools = _ACTIVE_TOOL_SPECS.set(tool_specs)
        try:
            model_input, weights = super().build_supervised_example(
                messages, train_on_what
            )
        finally:
            _ACTIVE_TOOL_SPECS.reset(active_tools)
            _ACTIVE_MESSAGES.reset(active_messages)
        if not messages or messages[-1]["role"] != "assistant":
            return model_input, weights

        boundary = tinker.types.EncodedTextChunk(tokens=[self._stop_token_ids[_USER]])
        final_message = messages[-1]
        if train_on_what == TrainOnWhat.ALL_USER_AND_SYSTEM_MESSAGES:
            boundary_weight = 0.0
        elif train_on_what == TrainOnWhat.CUSTOMIZED:
            boundary_weight = float(bool(final_message.get("trainable", False)))
        else:
            boundary_weight = 1.0

        return (
            tinker.ModelInput(chunks=[*model_input.chunks, boundary]),
            torch.cat((weights, weights.new_tensor([boundary_weight]))),
        )

    def parse_response(self, response: list[int]) -> tuple[Message, ParseTermination]:
        stop_ids = set(self._stop_token_ids.values())
        stop_positions = [
            index for index, token in enumerate(response) if token in stop_ids
        ]
        if len(stop_positions) > 1:
            raise RendererError(
                "GLM-5.2 response contains more than one stop token; check sampler stop sequences"
            )

        if stop_positions:
            stop_position = stop_positions[0]
            stop_id = response[stop_position]
            body_tokens = response[:stop_position]
            termination = (
                ParseTermination.EOS
                if stop_id == self._stop_token_ids[_END_OF_TEXT]
                else ParseTermination.STOP_SEQUENCE
            )
        else:
            body_tokens = response
            termination = ParseTermination.MALFORMED

        body = str(self.tokenizer.decode(body_tokens))
        reasoning: str | None = None
        if self.enable_thinking:
            # Sampler output after the prefilled ``<think>``. First close
            # ends the think span. This is serving parse, not leftover
            # promotion: a completion that *is* ``see <think>x</think> please``
            # is cut at the first close. Training records with that text go
            # through render, not this method.
            # Sampled ``</think>ANSWER`` after the prefilled ``<think>`` is
            # empty CoT + answer — the reverse of leftover promotion, not a
            # second contract.
            body = body.removeprefix(_THINK_OPEN)
            if _THINK_CLOSE in body:
                reasoning, body = body.split(_THINK_CLOSE, 1)
            else:
                reasoning = body
                body = ""
                termination = ParseTermination.MALFORMED
        elif body.startswith(f"{_THINK_OPEN}{_THINK_CLOSE}"):
            # Disable-thinking prompt already wrote ``<think></think>``.
            # Strip that exact scaffold if it leaked into the sampled
            # string. A bare ``</think>ANSWER`` is visible answer text
            # (literal tags), not an empty-reasoning split — do not treat
            # it like the thinking-enabled path.
            body = body[len(_THINK_OPEN) + len(_THINK_CLOSE) :]

        visible, tool_calls, unparsed = _parse_tool_calls(body)
        if reasoning is None:
            content: Any = visible
        else:
            content = [{"type": "thinking", "thinking": reasoning}]
            if visible:
                content.append({"type": "text", "text": visible})

        message = Message(role="assistant", content=content)
        if tool_calls:
            message["tool_calls"] = tool_calls
        if unparsed:
            message["unparsed_tool_calls"] = unparsed
        return message, termination

    def to_openai_message(self, message: Message) -> dict[str, Any]:
        result: dict[str, Any] = {"role": message["role"]}
        if message["role"] == "assistant":
            has_reasoning, reasoning, visible = _assistant_content_raw(message)
            result["content"] = (
                visible
                if isinstance(visible, str)
                else self._flatten_visible(visible, allow_thinking=True)
            )
            if has_reasoning:
                result["reasoning_content"] = reasoning
        else:
            result["content"] = self._flatten_visible(message["content"])

        if message.get("tool_calls"):
            result["tool_calls"] = [
                {
                    "type": "function",
                    "id": tool_call.id,
                    "function": {
                        "name": tool_call.function.name,
                        # GLM-5.2's HF template calls ``arguments.items()``.
                        "arguments": _tool_arguments(tool_call),
                    },
                }
                for tool_call in message["tool_calls"]
            ]
        if message["role"] == "tool":
            if "tool_call_id" in message:
                result["tool_call_id"] = message["tool_call_id"]
            if "name" in message:
                result["name"] = message["name"]
        return result

    def create_conversation_prefix_with_tools(
        self, tools: list[ToolSpec], system_prompt: str = ""
    ) -> list[Message]:
        messages: list[Message] = []
        if tools:
            normalized_tools = [_normalize_tool_spec(raw_tool) for raw_tool in tools]
            tool_lines: list[str] = []
            for tool in normalized_tools:
                if tool.get("defer_loading"):
                    continue
                tool_lines.append(
                    json.dumps(_serializable_tool_spec(tool), ensure_ascii=False)
                )

            tools_text = "\n".join(tool_lines)
            declaration = (
                "\n# Tools\n\n"
                "You may call one or more functions to assist with the user query.\n\n"
                "You are provided with function signatures within <tools></tools> XML tags:\n"
                f"<tools>\n{tools_text}\n</tools>\n\n"
                "For each function call, output the function name and arguments within the "
                "following XML format:\n"
                "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key>"
                "<arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key>"
                "<arg_value>{arg-value-2}</arg_value>...</tool_call>"
            )
            tool_message = Message(role="system", content=declaration)
            cast(dict[str, Any], tool_message)[_TOOL_CONTEXT_KEY] = normalized_tools
            messages.append(tool_message)
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        return messages


class GLM53Renderer(GLM52Renderer):
    """GLM-5.3 text renderer. Official template accepts low, high, and max."""

    reasoning_efforts = GLM53_REASONING_EFFORTS


class GLM52DisableThinkingRenderer(GLM52Renderer):
    """GLM-5.2 renderer whose prompt pre-fills ``<think></think>``.

    Thinking is off, so the template writes no reasoning-effort line.
    """

    disables_thinking = True

    def __init__(self, tokenizer: Tokenizer):
        super().__init__(tokenizer, enable_thinking=False)


def register_glm52_renderers() -> None:
    """Register MinT's namespaced GLM renderer factories."""
    register_renderer(
        GLM52_RENDERER,
        lambda tokenizer, image_processor=None: GLM52Renderer(tokenizer),
    )
    register_renderer(
        GLM53_RENDERER,
        lambda tokenizer, image_processor=None: GLM53Renderer(tokenizer),
    )
    register_renderer(
        GLM52_HIGH_REASONING_RENDERER,
        lambda tokenizer, image_processor=None: GLM52Renderer(
            tokenizer, reasoning_effort="high"
        ),
    )
    register_renderer(
        GLM52_DISABLE_THINKING_RENDERER,
        lambda tokenizer, image_processor=None: GLM52DisableThinkingRenderer(tokenizer),
    )


register_glm52_renderers()


__all__ = [
    "GLM52_DISABLE_THINKING_RENDERER",
    "GLM52_HIGH_REASONING_RENDERER",
    "GLM52_RENDERER",
    "GLM53_RENDERER",
    "GLM52DisableThinkingRenderer",
    "GLM52Renderer",
    "GLM53Renderer",
    "REASONING_EFFORTS",
    "register_glm52_renderers",
]
