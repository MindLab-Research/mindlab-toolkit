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
from collections.abc import Mapping
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

GLM52_RENDERER = "MindLab/glm52"
GLM52_HIGH_REASONING_RENDERER = "MindLab/glm52_high_reasoning"
GLM52_DISABLE_THINKING_RENDERER = "MindLab/glm52_disable_thinking"

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
_TOOL_CONTEXT_KEY = "_mint_glm52_tools"
_ACTIVE_TOOL_SPECS: ContextVar[tuple[dict[str, Any], ...] | None] = ContextVar(
    "mint_glm52_active_tool_specs", default=None
)


def _encoded_chunk(tokenizer: Tokenizer, text: str) -> tinker.types.EncodedTextChunk:
    return tinker.types.EncodedTextChunk(
        tokens=tokenizer.encode(text, add_special_tokens=False)
    )


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


def _visible_text(content: Any, *, allow_thinking: bool = False) -> str:
    """Mirror the GLM-5.2 template's ``visible_text`` macro."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise RendererError(
            "GLM-5.2 message content must be a string or a list of content parts"
        )

    rendered: list[str] = []
    for part in content:
        if isinstance(part, str):
            rendered.append(part)
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
            rendered.append(text)
        elif part_type == "thinking":
            if allow_thinking:
                continue
            raise RendererError("thinking content is only valid on assistant messages")
        elif isinstance(part_type, str) and part_type in _MEDIA_TYPES:
            rendered.append(_media_reminder(_MEDIA_TYPES[part_type]))
        else:
            raise RendererError(f"unsupported GLM-5.2 content part type: {part_type!r}")
    return "".join(rendered)


def _assistant_content(message: Message) -> tuple[bool, str, str]:
    """Return ``(has_reasoning_signal, reasoning, visible_content)``.

    Tinker represents reasoning with ``ThinkingPart``.  The legacy
    ``reasoning_content`` key is also accepted so existing GLM-5.2 SFT records
    can be handed to the renderer without a preprocessing pass.
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
        return True, reasoning, _visible_text(content, allow_thinking=True)

    if isinstance(content, str):
        if _THINK_OPEN in content or _THINK_CLOSE in content:
            raise RendererError(
                "inline <think> markers are not valid GLM-5.2 message content; "
                "use a ThinkingPart or reasoning_content"
            )
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
    return (
        bool(reasoning_parts),
        "".join(reasoning_parts),
        _visible_text(content, allow_thinking=True),
    )


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


def _render_tool_calls(tool_calls: list[ToolCall]) -> str:
    rendered: list[str] = []
    for tool_call in tool_calls:
        arguments = _tool_arguments(tool_call)
        parts = ["<tool_call>", tool_call.function.name]
        for key, value in arguments.items():
            rendered_value = (
                value
                if isinstance(value, str)
                else json.dumps(value, ensure_ascii=False)
            )
            parts.extend(
                (
                    "<arg_key>",
                    str(key),
                    "</arg_key><arg_value>",
                    rendered_value,
                    "</arg_value>",
                )
            )
        parts.append("</tool_call>")
        rendered.append("".join(parts))
    return "".join(rendered)


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


def _render_tool_references(
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
    rendered_tools = "".join(f"{line}\n" for line in tool_lines)
    return f"<tool_response><tools>\n{rendered_tools}</tools></tool_response>"


def _render_tool_response(
    content: Any, tool_specs: tuple[dict[str, Any], ...] | None
) -> str:
    if isinstance(content, str):
        return f"<tool_response>{content}</tool_response>"
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, Mapping) and first.get("type") == "tool_reference":
            return _render_tool_references(content, tool_specs)
        if isinstance(first, Mapping) and "output" in first:
            outputs: list[str] = []
            for item in content:
                if not isinstance(item, Mapping) or not isinstance(
                    item.get("output"), str
                ):
                    raise RendererError(
                        "structured GLM-5.2 tool results require string 'output' fields"
                    )
                outputs.append(f"<tool_response>{item['output']}</tool_response>")
            return "".join(outputs)
    return f"<tool_response>{_visible_text(content)}</tool_response>"


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

    ``reasoning_effort='max'`` matches the tokenizer template default.  Use
    ``'high'`` for the lower-effort supported mode.  Historical reasoning is
    preserved, which gives this renderer Tinker's sequence extension property.
    """

    supports_streaming = False

    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        reasoning_effort: Literal["high", "max"] = "max",
        enable_thinking: bool = True,
    ):
        super().__init__(tokenizer)
        if reasoning_effort not in ("high", "max"):
            raise ValueError("GLM-5.2 reasoning_effort must be 'high' or 'max'")
        self.reasoning_effort = reasoning_effort
        self.enable_thinking = bool(enable_thinking)
        self.disables_thinking = not self.enable_thinking

        prefix = _GMASK_SOP
        if self.enable_thinking:
            prefix += f"{_SYSTEM}Reasoning Effort: {reasoning_effort.capitalize()}"
        self._prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        self._stop_token_ids = {
            token: _single_token_id(tokenizer, token) for token in _STOP_TOKENS
        }

    @property
    def _bos_tokens(self) -> list[int]:
        return list(self._prefix_tokens)

    @property
    def has_extension_property(self) -> bool:
        return True

    def get_stop_sequences(self) -> list[int]:
        return [self._stop_token_ids[token] for token in _STOP_TOKENS]

    def _assistant_header(self) -> str:
        if self.enable_thinking:
            return f"{_ASSISTANT}{_THINK_OPEN}"
        return f"{_ASSISTANT}{_THINK_OPEN}{_THINK_CLOSE}"

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
            has_reasoning, reasoning, content = _assistant_content(message)
            if not self.enable_thinking and has_reasoning:
                raise RendererError(
                    "GLM-5.2 thinking is disabled, but the assistant message contains "
                    "ThinkingPart/reasoning_content"
                )
            header = _encoded_chunk(self.tokenizer, self._assistant_header())
            output_text = ""
            if self.enable_thinking:
                output_text += reasoning + _THINK_CLOSE
            if content.strip():
                output_text += content.strip()
            tool_calls = list(message.get("tool_calls", []))
            if tool_calls:
                output_text += _render_tool_calls(tool_calls)
            output_text += self._boundary_after_assistant(ctx)
            output = (
                [_encoded_chunk(self.tokenizer, output_text)] if output_text else []
            )
            return RenderedMessage(header=header, output=output)

        prev_role = ctx.prev_message["role"] if ctx.prev_message is not None else None
        if role == "user":
            header_text = "" if prev_role == "assistant" else _USER
            output_text = _visible_text(message["content"])
        elif role == "system":
            header_text = _SYSTEM
            output_text = _visible_text(message["content"])
        else:
            header_text = "" if prev_role in ("assistant", "tool") else _OBSERVATION
            output_text = _render_tool_response(
                message["content"], _ACTIVE_TOOL_SPECS.get()
            )

        header = _encoded_chunk(self.tokenizer, header_text) if header_text else None
        output = [_encoded_chunk(self.tokenizer, output_text)] if output_text else []
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
            body = body.removeprefix(_THINK_OPEN)
            if _THINK_CLOSE in body:
                reasoning, body = body.split(_THINK_CLOSE, 1)
            else:
                reasoning = body
                body = ""
                termination = ParseTermination.MALFORMED
        elif body.startswith(f"{_THINK_OPEN}{_THINK_CLOSE}"):
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
            has_reasoning, reasoning, visible = _assistant_content(message)
            result["content"] = visible
            if has_reasoning:
                result["reasoning_content"] = reasoning
        else:
            result["content"] = _visible_text(message["content"])

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


class GLM52DisableThinkingRenderer(GLM52Renderer):
    """GLM-5.2 renderer whose prompt pre-fills ``<think></think>``."""

    disables_thinking = True

    def __init__(self, tokenizer: Tokenizer):
        super().__init__(tokenizer, enable_thinking=False)


def register_glm52_renderers() -> None:
    """Register MinT's namespaced GLM-5.2 renderer factories."""
    register_renderer(
        GLM52_RENDERER,
        lambda tokenizer, image_processor=None: GLM52Renderer(tokenizer),
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
    "GLM52DisableThinkingRenderer",
    "GLM52Renderer",
    "register_glm52_renderers",
]
