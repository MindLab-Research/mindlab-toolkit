"""Default-suite contract tests for the GLM-5.2 Tinker renderer."""

from __future__ import annotations

import json
from typing import Any, ClassVar

import pytest

pytest.importorskip(
    "tinker_cookbook.renderers",
    reason="install mindlab-toolkit[test] to run GLM-5.2 renderer tests",
)

from tinker_cookbook.exceptions import RendererError
from tinker_cookbook.renderers import (
    Message,
    ParseTermination,
    ToolCall,
    TrainOnWhat,
    get_renderer,
)

from mint.renderers import (
    GLM52_DISABLE_THINKING_RENDERER,
    GLM52_HIGH_REASONING_RENDERER,
    GLM52_RENDERER,
    GLM52DisableThinkingRenderer,
    GLM52Renderer,
)
from mint.renderers import glm52 as glm52_module


class CharacterTokenizer:
    """Small reversible tokenizer with GLM special tokens kept atomic."""

    name_or_path = "zai-org/GLM-5.2-test-tokenizer"
    _specials: ClassVar[tuple[str, ...]] = (
        "<|endoftext|>",
        "<|observation|>",
        "<|assistant|>",
        "<|system|>",
        "<|user|>",
        "[gMASK]",
        "<sop>",
    )
    _token_by_text: ClassVar[dict[str, int]] = {
        text: 0x110000 + index for index, text in enumerate(_specials)
    }
    _text_by_token: ClassVar[dict[int, str]] = {
        token: text for text, token in _token_by_text.items()
    }

    def encode(
        self, text: str, add_special_tokens: bool = False, **_: Any
    ) -> list[int]:
        del add_special_tokens
        result: list[int] = []
        position = 0
        while position < len(text):
            special = next(
                (item for item in self._specials if text.startswith(item, position)),
                None,
            )
            if special is None:
                result.append(ord(text[position]))
                position += 1
            else:
                result.append(self._token_by_text[special])
                position += len(special)
        return result

    def decode(self, tokens: list[int]) -> str:
        return "".join(
            self._text_by_token[token] if token in self._text_by_token else chr(token)
            for token in tokens
        )


@pytest.fixture
def tokenizer() -> CharacterTokenizer:
    return CharacterTokenizer()


def _decode(tokenizer: CharacterTokenizer, model_input: Any) -> str:
    return tokenizer.decode(model_input.to_ints())


def _tool_call(name: str, arguments: dict[str, Any]) -> ToolCall:
    return ToolCall(
        function=ToolCall.FunctionBody(
            name=name,
            arguments=json.dumps(arguments, ensure_ascii=False),
        ),
        id="call-1",
    )


def test_generation_prompt_matches_glm52_thinking_template(
    tokenizer: CharacterTokenizer,
):
    renderer = GLM52Renderer(tokenizer)
    prompt = renderer.build_generation_prompt([Message(role="user", content=" hi ")])

    assert _decode(tokenizer, prompt) == (
        "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|> hi <|assistant|><think>"
    )
    assert renderer.has_extension_property is True


def test_generation_prompt_supports_high_and_disable_modes(
    tokenizer: CharacterTokenizer,
):
    high = GLM52Renderer(tokenizer, reasoning_effort="high")
    disabled = GLM52DisableThinkingRenderer(tokenizer)
    messages = [Message(role="user", content="hello")]

    assert _decode(tokenizer, high.build_generation_prompt(messages)).startswith(
        "[gMASK]<sop><|system|>Reasoning Effort: High"
    )
    assert _decode(tokenizer, disabled.build_generation_prompt(messages)) == (
        "[gMASK]<sop><|user|>hello<|assistant|><think></think>"
    )
    assert disabled.disables_thinking is True


def test_thinking_sft_weights_begin_after_prefilled_opener(
    tokenizer: CharacterTokenizer,
):
    renderer = GLM52Renderer(tokenizer)
    messages = [
        Message(role="user", content="question"),
        Message(
            role="assistant",
            content=[
                {"type": "thinking", "thinking": "reason"},
                {"type": "text", "text": " answer "},
            ],
        ),
    ]

    model_input, weights = renderer.build_supervised_example(messages)
    prompt_length = renderer.build_generation_prompt(messages[:-1]).length

    assert _decode(tokenizer, model_input) == (
        "[gMASK]<sop><|system|>Reasoning Effort: Max"
        "<|user|>question<|assistant|><think>reason</think>answer<|user|>"
    )
    assert weights[:prompt_length].tolist() == [0.0] * prompt_length
    assert weights[prompt_length:].tolist() == [1.0] * (len(weights) - prompt_length)


def test_legacy_reasoning_content_is_rendered_without_preprocessing(
    tokenizer: CharacterTokenizer,
):
    renderer = GLM52Renderer(tokenizer)
    messages = [
        Message(role="user", content="question"),
        {"role": "assistant", "reasoning_content": " legacy ", "content": "answer"},
    ]

    model_input, _ = renderer.build_supervised_example(messages)  # type: ignore[arg-type]
    assert "<think> legacy </think>answer" in _decode(tokenizer, model_input)


def test_disable_thinking_sft_does_not_train_prefilled_pair(
    tokenizer: CharacterTokenizer,
):
    renderer = GLM52DisableThinkingRenderer(tokenizer)
    messages = [
        Message(role="user", content="question"),
        Message(role="assistant", content="answer"),
    ]

    model_input, weights = renderer.build_supervised_example(messages)
    prompt_length = renderer.build_generation_prompt(messages[:-1]).length

    assert _decode(tokenizer, model_input).endswith(
        "<|assistant|><think></think>answer<|user|>"
    )
    assert weights[:prompt_length].tolist() == [0.0] * prompt_length
    assert weights[prompt_length:].tolist() == [1.0] * (len(weights) - prompt_length)


def test_disable_thinking_rejects_reasoning(tokenizer: CharacterTokenizer):
    renderer = GLM52DisableThinkingRenderer(tokenizer)
    message = Message(
        role="assistant",
        content=[{"type": "thinking", "thinking": "reason"}],
    )
    with pytest.raises(RendererError, match="thinking is disabled"):
        renderer.build_supervised_example([message])


def test_multi_turn_assistant_boundaries_are_trained(tokenizer: CharacterTokenizer):
    renderer = GLM52DisableThinkingRenderer(tokenizer)
    messages = [
        Message(role="user", content="q1"),
        Message(role="assistant", content="a1"),
        Message(role="user", content="q2"),
        Message(role="assistant", content="a2"),
    ]

    model_input, weights = renderer.build_supervised_example(
        messages, TrainOnWhat.ALL_ASSISTANT_MESSAGES
    )
    user_id = tokenizer._token_by_text["<|user|>"]
    user_weights = [
        float(weights[index])
        for index, token in enumerate(model_input.to_ints())
        if token == user_id
    ]

    assert _decode(tokenizer, model_input) == (
        "[gMASK]<sop><|user|>q1<|assistant|><think></think>a1"
        "<|user|>q2<|assistant|><think></think>a2<|user|>"
    )
    assert user_weights == [0.0, 1.0, 1.0]


def test_tool_round_trip_and_observation_boundary(tokenizer: CharacterTokenizer):
    renderer = GLM52Renderer(tokenizer)
    response = tokenizer.encode(
        "reason</think>checking"
        "<tool_call>weather<arg_key>city</arg_key><arg_value>Paris</arg_value>"
        "</tool_call><|observation|>"
    )

    assistant, termination = renderer.parse_response(response)

    assert termination == ParseTermination.STOP_SEQUENCE
    assert assistant["content"] == [
        {"type": "thinking", "thinking": "reason"},
        {"type": "text", "text": "checking"},
    ]
    assert assistant["tool_calls"][0].function.name == "weather"
    assert json.loads(assistant["tool_calls"][0].function.arguments) == {
        "city": "Paris"
    }

    messages = [
        Message(role="user", content="weather?"),
        assistant,
        Message(role="tool", content="sunny", tool_call_id="call-1", name="weather"),
    ]
    next_prompt = _decode(tokenizer, renderer.build_generation_prompt(messages))
    assert next_prompt.endswith(
        "<|assistant|><think>reason</think>checking"
        "<tool_call>weather<arg_key>city</arg_key><arg_value>Paris</arg_value>"
        "</tool_call><|observation|><tool_response>sunny</tool_response>"
        "<|assistant|><think>"
    )


def test_tool_declaration_matches_pinned_template_shape(tokenizer: CharacterTokenizer):
    renderer = GLM52Renderer(tokenizer)
    prefix = renderer.create_conversation_prefix_with_tools(
        [
            {
                "name": "weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            }
        ],
        system_prompt="Be concise.",
    )
    prompt = _decode(
        tokenizer,
        renderer.build_generation_prompt(
            [*prefix, Message(role="user", content="weather?")]
        ),
    )

    assert prompt.startswith(
        "[gMASK]<sop><|system|>Reasoning Effort: Max<|system|>\n# Tools\n\n"
    )
    assert '<tools>\n{"name": "weather", "description": "Get weather"' in prompt
    assert "</tools>\n\nFor each function call" in prompt
    assert "<|system|>Be concise.<|user|>weather?<|assistant|><think>" in prompt


def test_parse_eos_and_malformed_thinking(tokenizer: CharacterTokenizer):
    renderer = GLM52Renderer(tokenizer)
    complete, complete_termination = renderer.parse_response(
        tokenizer.encode("r</think>a<|endoftext|>")
    )
    malformed, malformed_termination = renderer.parse_response(
        tokenizer.encode("unfinished reasoning<|user|>")
    )

    assert complete_termination == ParseTermination.EOS
    assert complete["content"] == [
        {"type": "thinking", "thinking": "r"},
        {"type": "text", "text": "a"},
    ]
    assert malformed_termination == ParseTermination.MALFORMED
    assert malformed["content"] == [
        {"type": "thinking", "thinking": "unfinished reasoning"}
    ]


def test_parse_preserves_malformed_tool_call(tokenizer: CharacterTokenizer):
    renderer = GLM52DisableThinkingRenderer(tokenizer)
    message, termination = renderer.parse_response(
        tokenizer.encode("<tool_call>broken<arg_key>x</arg_key><|user|>")
    )

    assert termination == ParseTermination.STOP_SEQUENCE
    assert message["content"] == ""
    assert "tool_calls" not in message
    assert message["unparsed_tool_calls"][0].raw_text.startswith("<tool_call>broken")


def test_openai_conversion_uses_glm_argument_object(tokenizer: CharacterTokenizer):
    renderer = GLM52Renderer(tokenizer)
    message = Message(
        role="assistant",
        content=[
            {"type": "thinking", "thinking": "reason"},
            {"type": "text", "text": "answer"},
        ],
        tool_calls=[_tool_call("f", {"count": 2, "label": "two"})],
    )

    converted = renderer.to_openai_message(message)

    assert converted["reasoning_content"] == "reason"
    assert converted["content"] == "answer"
    assert converted["tool_calls"][0]["function"]["arguments"] == {
        "count": 2,
        "label": "two",
    }


def test_media_parts_render_as_text_only_reminders(tokenizer: CharacterTokenizer):
    renderer = GLM52Renderer(tokenizer)
    prompt = renderer.build_generation_prompt(
        [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "inspect "},
                    {"type": "image", "image": "https://example.test/image.png"},
                ],
            )
        ]
    )

    assert "inspect <reminder>You are unable to process this image" in _decode(
        tokenizer, prompt
    )


@pytest.mark.parametrize(
    ("name", "expected_type"),
    [
        (GLM52_RENDERER, GLM52Renderer),
        (GLM52_HIGH_REASONING_RENDERER, GLM52Renderer),
        (GLM52_DISABLE_THINKING_RENDERER, GLM52DisableThinkingRenderer),
    ],
)
def test_registered_renderer_names(
    tokenizer: CharacterTokenizer, name: str, expected_type: type[GLM52Renderer]
):
    renderer = get_renderer(name, tokenizer, model_name=tokenizer.name_or_path)
    assert isinstance(renderer, expected_type)


def test_renderer_rejects_non_atomic_stop_tokens():
    class BadTokenizer(CharacterTokenizer):
        def encode(
            self, text: str, add_special_tokens: bool = False, **kwargs: Any
        ) -> list[int]:
            if text == "<|user|>":
                return [1, 2]
            return super().encode(text, add_special_tokens=add_special_tokens, **kwargs)

    with pytest.raises(RendererError, match="must encode to one token"):
        GLM52Renderer(BadTokenizer())


@pytest.mark.parametrize(
    ("train_on_what", "trainable", "expected_boundary_weight"),
    [
        (TrainOnWhat.LAST_ASSISTANT_MESSAGE, False, 1.0),
        (TrainOnWhat.LAST_ASSISTANT_TURN, False, 1.0),
        (TrainOnWhat.ALL_ASSISTANT_MESSAGES, False, 1.0),
        (TrainOnWhat.ALL_MESSAGES, False, 1.0),
        (TrainOnWhat.ALL_TOKENS, False, 1.0),
        (TrainOnWhat.ALL_USER_AND_SYSTEM_MESSAGES, True, 0.0),
        (TrainOnWhat.CUSTOMIZED, False, 0.0),
        (TrainOnWhat.CUSTOMIZED, True, 1.0),
    ],
)
def test_final_user_boundary_obeys_every_tinker_training_mode(
    tokenizer: CharacterTokenizer,
    train_on_what: TrainOnWhat,
    trainable: bool,
    expected_boundary_weight: float,
):
    renderer = GLM52DisableThinkingRenderer(tokenizer)
    if train_on_what == TrainOnWhat.CUSTOMIZED:
        messages = [
            Message(role="user", content="q", trainable=False),
            Message(role="assistant", content="a", trainable=trainable),
        ]
    else:
        messages = [
            Message(role="user", content="q"),
            Message(role="assistant", content="a"),
        ]

    model_input, weights = renderer.build_supervised_example(messages, train_on_what)

    assert model_input.to_ints()[-1] == tokenizer._token_by_text["<|user|>"]
    assert float(weights[-1]) == expected_boundary_weight
    assert len(model_input.to_ints()) == len(weights)


@pytest.mark.parametrize("enable_thinking", [True, False])
def test_supervised_action_starts_at_generation_prompt_and_parses_back(
    tokenizer: CharacterTokenizer, enable_thinking: bool
):
    renderer = (
        GLM52Renderer(tokenizer)
        if enable_thinking
        else GLM52DisableThinkingRenderer(tokenizer)
    )
    assistant = (
        Message(
            role="assistant",
            content=[
                {"type": "thinking", "thinking": "reason"},
                {"type": "text", "text": "answer"},
            ],
        )
        if enable_thinking
        else Message(role="assistant", content="answer")
    )
    messages = [Message(role="user", content="question"), assistant]

    model_input, weights = renderer.build_supervised_example(messages)
    tokens = model_input.to_ints()
    first_trained = weights.tolist().index(1)
    observation, action = tokens[:first_trained], tokens[first_trained:]

    assert observation == renderer.build_generation_prompt(messages[:-1]).to_ints()
    parsed, termination = renderer.parse_response(action)
    assert termination == ParseTermination.STOP_SEQUENCE
    assert parsed["content"] == assistant["content"]


def test_generation_prefill_appends_after_renderer_thinking_scaffold(
    tokenizer: CharacterTokenizer,
):
    messages = [Message(role="user", content="q")]

    for renderer in (
        GLM52Renderer(tokenizer),
        GLM52DisableThinkingRenderer(tokenizer),
    ):
        plain = _decode(tokenizer, renderer.build_generation_prompt(messages))
        prefilled = _decode(
            tokenizer,
            renderer.build_generation_prompt(messages, prefill="Sure"),
        )
        assert prefilled == plain + "Sure"


def test_terminal_non_assistant_turn_does_not_gain_synthetic_boundary(
    tokenizer: CharacterTokenizer,
):
    renderer = GLM52Renderer(tokenizer)
    messages = [
        Message(role="user", content="q"),
        Message(
            role="assistant",
            content=[{"type": "thinking", "thinking": "r"}],
            tool_calls=[_tool_call("f", {})],
        ),
        Message(role="tool", content="result"),
    ]

    model_input, weights = renderer.build_supervised_example(
        messages, TrainOnWhat.ALL_ASSISTANT_MESSAGES
    )
    rendered = _decode(tokenizer, model_input)

    assert rendered.endswith("<tool_response>result</tool_response>")
    assert not rendered.endswith("<|user|>")
    assert len(model_input.to_ints()) == len(weights)


@pytest.mark.parametrize(
    "arguments",
    ["not-json", "[]", "null", "1", '"scalar"'],
)
def test_tool_call_rejects_arguments_that_are_not_json_objects(
    tokenizer: CharacterTokenizer, arguments: str
):
    renderer = GLM52Renderer(tokenizer)
    tool_call = ToolCall(
        function=ToolCall.FunctionBody(name="bad", arguments=arguments)
    )
    message = Message(role="assistant", content="", tool_calls=[tool_call])

    with pytest.raises(RendererError, match="arguments"):
        renderer.build_supervised_example([message])


def test_parser_rejects_multiple_protocol_stop_tokens(
    tokenizer: CharacterTokenizer,
):
    renderer = GLM52Renderer(tokenizer)
    response = tokenizer.encode("r</think>a<|user|>junk<|endoftext|>")

    with pytest.raises(RendererError, match="more than one stop token"):
        renderer.parse_response(response)


def test_tool_prefix_accepts_wrapped_specs_and_omits_deferred_tools(
    tokenizer: CharacterTokenizer,
):
    renderer = GLM52Renderer(tokenizer)
    prefix = renderer.create_conversation_prefix_with_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "live",
                    "description": "available",
                    "parameters": {"type": "object", "properties": {}},
                    "strict": True,
                },
            },
            {
                "name": "later",
                "description": "deferred",
                "parameters": {"type": "object", "properties": {}},
                "defer_loading": True,
            },
        ]
    )
    prompt = _decode(
        tokenizer,
        renderer.build_generation_prompt([*prefix, Message(role="user", content="q")]),
    )

    assert '"name": "live"' in prompt
    assert '"strict"' not in prompt
    assert '"name": "later"' not in prompt

    reference_prompt = _decode(
        tokenizer,
        renderer.build_generation_prompt(
            [
                *prefix,
                Message(
                    role="tool",
                    content=[
                        {"type": "tool_reference", "name": "later"},
                        {"type": "tool_reference", "name": "live"},
                    ],
                ),
            ]
        ),
    )
    assert (
        '<tool_response><tools>\n{"name": "later", "description": "deferred", '
        '"parameters": {"type": "object", "properties": {}}}\n'
        '{"name": "live", "description": "available", "parameters": '
        '{"type": "object", "properties": {}}}\n</tools></tool_response>'
        in reference_prompt
    )
    assert '"defer_loading"' not in reference_prompt
    assert '"strict"' not in reference_prompt


def test_tool_reference_context_is_isolated_between_conversations(
    tokenizer: CharacterTokenizer,
):
    renderer = GLM52Renderer(tokenizer)
    prefix_a = renderer.create_conversation_prefix_with_tools(
        [
            {
                "name": "tool_a",
                "description": "A",
                "parameters": {"type": "object", "properties": {}},
                "defer_loading": True,
            }
        ]
    )
    prefix_b = renderer.create_conversation_prefix_with_tools(
        [
            {
                "name": "tool_b",
                "description": "B",
                "parameters": {"type": "object", "properties": {}},
                "defer_loading": True,
            }
        ]
    )

    def render(prefix: list[Message], name: str) -> str:
        return _decode(
            tokenizer,
            renderer.build_generation_prompt(
                [
                    *prefix,
                    Message(
                        role="tool",
                        content=[{"type": "tool_reference", "name": name}],
                    ),
                ]
            ),
        )

    rendered_a = render(prefix_a, "tool_a")
    rendered_b = render(prefix_b, "tool_b")
    rendered_a_again = render(prefix_a, "tool_a")

    assert rendered_a == rendered_a_again
    assert '"name": "tool_a"' in rendered_a
    assert '"name": "tool_b"' not in rendered_a
    assert '"name": "tool_b"' in rendered_b
    assert '"name": "tool_a"' not in rendered_b


@pytest.mark.parametrize("supervised", [False, True])
def test_invalid_tool_context_does_not_leak_active_messages(
    tokenizer: CharacterTokenizer, supervised: bool
):
    renderer = GLM52Renderer(tokenizer)
    invalid_prefix = Message(role="system", content="invalid tool context")
    invalid_prefix["_mint_glm52_tools"] = "not-a-tool-list"  # type: ignore[typeddict-unknown-key]

    assert glm52_module._ACTIVE_MESSAGES.get() is None
    with pytest.raises(RendererError, match="invalid GLM-5.2 tool context"):
        if supervised:
            renderer.build_supervised_example([invalid_prefix])
        else:
            renderer.build_generation_prompt([invalid_prefix])
    assert glm52_module._ACTIVE_MESSAGES.get() is None


def test_stop_sequences_are_all_atomic_glm_protocol_boundaries(
    tokenizer: CharacterTokenizer,
):
    renderer = GLM52Renderer(tokenizer)

    assert renderer.get_stop_sequences() == [
        tokenizer._token_by_text["<|endoftext|>"],
        tokenizer._token_by_text["<|user|>"],
        tokenizer._token_by_text["<|observation|>"],
    ]


@pytest.mark.parametrize(
    ("role", "suffix"),
    [
        ("user", "<|user|>"),
        ("system", "<|system|>"),
        ("tool", "<|observation|>"),
    ],
)
def test_generation_supports_every_protocol_role_suffix(
    tokenizer: CharacterTokenizer, role: str, suffix: str
):
    renderer = GLM52Renderer(tokenizer)

    prompt = _decode(tokenizer, renderer.build_generation_prompt([], role=role))

    assert prompt.endswith(suffix)


def test_generation_rejects_unknown_target_and_message_roles(
    tokenizer: CharacterTokenizer,
):
    renderer = GLM52Renderer(tokenizer)

    with pytest.raises(RendererError, match="generation role"):
        renderer.build_generation_prompt([], role="developer")
    with pytest.raises(RendererError, match="unsupported GLM-5.2 role"):
        renderer.build_generation_prompt(
            [{"role": "developer", "content": "bad"}]  # type: ignore[list-item]
        )


@pytest.mark.parametrize(
    ("message", "error"),
    [
        ({"role": "user", "content": 3}, "string or a list"),
        ({"role": "user", "content": [3]}, "mappings or strings"),
        (
            {"role": "user", "content": [{"type": "text", "text": 3}]},
            "string 'text' field",
        ),
        (
            {
                "role": "user",
                "content": [{"type": "thinking", "thinking": "private"}],
            },
            "only valid on assistant",
        ),
        (
            {"role": "user", "content": [{"type": "unknown", "value": "x"}]},
            "unsupported GLM-5.2 content part",
        ),
    ],
)
def test_invalid_content_parts_fail_loudly(
    tokenizer: CharacterTokenizer, message: dict[str, Any], error: str
):
    renderer = GLM52Renderer(tokenizer)

    with pytest.raises(RendererError, match=error):
        renderer.build_generation_prompt([message])  # type: ignore[list-item]


@pytest.mark.parametrize(
    ("message", "error"),
    [
        (
            {"role": "assistant", "content": "a", "reasoning_content": 3},
            "reasoning_content must be a string",
        ),
        (
            {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": "r"}],
                "reasoning_content": "also-r",
            },
            "either reasoning_content or ThinkingPart",
        ),
        (
            {"role": "assistant", "content": "<think>inline</think>a"},
            "inline <think> markers",
        ),
        (
            {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": 3}],
            },
            "ThinkingPart requires a string",
        ),
    ],
)
def test_invalid_reasoning_shapes_fail_loudly(
    tokenizer: CharacterTokenizer, message: dict[str, Any], error: str
):
    renderer = GLM52Renderer(tokenizer)

    with pytest.raises(RendererError, match=error):
        renderer.build_supervised_example([message])  # type: ignore[list-item]


def test_invalid_reasoning_effort_is_rejected(tokenizer: CharacterTokenizer):
    with pytest.raises(ValueError, match="must be 'high' or 'max'"):
        GLM52Renderer(tokenizer, reasoning_effort="low")  # type: ignore[arg-type]


def test_empty_tool_arguments_render_as_an_empty_call(
    tokenizer: CharacterTokenizer,
):
    renderer = GLM52Renderer(tokenizer)
    call = ToolCall(function=ToolCall.FunctionBody(name="ping", arguments=""))
    message = Message(role="assistant", content="", tool_calls=[call])

    model_input, _ = renderer.build_supervised_example([message])

    assert "<tool_call>ping</tool_call>" in _decode(tokenizer, model_input)


@pytest.mark.parametrize(
    ("content", "error"),
    [
        ([{"output": "ok"}, {"output": 3}], "require string 'output' fields"),
    ],
)
def test_unsupported_structured_tool_results_fail_loudly(
    tokenizer: CharacterTokenizer, content: list[dict[str, Any]], error: str
):
    renderer = GLM52Renderer(tokenizer)

    with pytest.raises(RendererError, match=error):
        renderer.build_generation_prompt(
            [Message(role="tool", content=content)]  # type: ignore[arg-type]
        )


def test_tool_reference_without_tool_prefix_fails_loudly(
    tokenizer: CharacterTokenizer,
):
    renderer = GLM52Renderer(tokenizer)

    with pytest.raises(RendererError, match="require tool definitions"):
        renderer.build_generation_prompt(
            [
                Message(
                    role="tool",
                    content=[{"type": "tool_reference", "name": "deferred"}],
                )
            ]
        )


def test_text_part_tool_result_uses_visible_text_fallback(
    tokenizer: CharacterTokenizer,
):
    renderer = GLM52Renderer(tokenizer)
    prompt = renderer.build_generation_prompt(
        [Message(role="tool", content=[{"type": "text", "text": "result"}])]
    )

    assert "<|observation|><tool_response>result</tool_response>" in _decode(
        tokenizer, prompt
    )


@pytest.mark.parametrize(
    "block",
    [
        "<tool_call><arg_key>x</arg_key><arg_value>1</arg_value></tool_call>",
        "<tool_call>f<arg_key> </arg_key><arg_value>1</arg_value></tool_call>",
        "<tool_call>f<arg_key>x</arg_key>missing-value</tool_call>",
    ],
)
def test_closed_but_malformed_tool_calls_are_preserved(
    tokenizer: CharacterTokenizer, block: str
):
    renderer = GLM52DisableThinkingRenderer(tokenizer)

    message, termination = renderer.parse_response(tokenizer.encode(f"{block}<|user|>"))

    assert termination == ParseTermination.STOP_SEQUENCE
    assert message["content"] == ""
    assert "tool_calls" not in message
    assert message["unparsed_tool_calls"][0].raw_text == block


def test_parse_without_stop_and_disable_scaffold_normalization(
    tokenizer: CharacterTokenizer,
):
    thinking = GLM52Renderer(tokenizer)
    disabled = GLM52DisableThinkingRenderer(tokenizer)

    incomplete, incomplete_termination = thinking.parse_response(
        tokenizer.encode("reason</think>answer")
    )
    normalized, normalized_termination = disabled.parse_response(
        tokenizer.encode("<think></think>answer<|user|>")
    )

    assert incomplete_termination == ParseTermination.MALFORMED
    assert incomplete["content"][-1] == {"type": "text", "text": "answer"}
    assert normalized_termination == ParseTermination.STOP_SEQUENCE
    assert normalized["content"] == "answer"


def test_to_openai_preserves_tool_response_identifiers(
    tokenizer: CharacterTokenizer,
):
    renderer = GLM52Renderer(tokenizer)
    message = Message(
        role="tool",
        content="result",
        tool_call_id="call-7",
        name="lookup",
    )

    assert renderer.to_openai_message(message) == {
        "role": "tool",
        "content": "result",
        "tool_call_id": "call-7",
        "name": "lookup",
    }
