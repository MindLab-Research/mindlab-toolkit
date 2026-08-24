"""Opt-in parity tests against the original GLM-5.2 SFT renderer.

These tests intentionally use the real ``zai-org/GLM-5.2`` tokenizer and the
SHA-pinned Jinja template/loss-mask implementation from
``MindLab-Research/agent-model-training-mono/glm52_sft``.  They run whenever
that reference checkout is available next to this repository, or when
``GLM52_SFT_REFERENCE_DIR`` points at it.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

pytest.importorskip(
    "tinker_cookbook.renderers",
    reason="install mindlab-toolkit[test] to run GLM-5.2 renderer tests",
)

from tinker_cookbook.renderers import Message, ToolCall, ToolSpec, TrainOnWhat

from mint.renderers import GLM52DisableThinkingRenderer, GLM52Renderer


def _reference_dir() -> Path:
    configured = os.environ.get("GLM52_SFT_REFERENCE_DIR")
    if configured:
        return Path(configured)
    test_file = Path(__file__).resolve()
    repository_root = next(
        parent for parent in test_file.parents if (parent / "pyproject.toml").is_file()
    )
    return repository_root.parent / "agent-model-training-mono" / "glm52_sft"


REFERENCE_DIR = _reference_dir()
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (REFERENCE_DIR / "template_mask.py").is_file()
        or not (REFERENCE_DIR / "chat_template.jinja").is_file(),
        reason=(
            "clone MindLab-Research/agent-model-training-mono next to this repository "
            "or set GLM52_SFT_REFERENCE_DIR to run GLM-5.2 SFT parity tests"
        ),
    ),
]


def _load_reference() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "mint_glm52_sft_reference", REFERENCE_DIR / "template_mask.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def real_tokenizer() -> Any:
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained("zai-org/GLM-5.2")
    return _load_reference().attach_pinned_chat_template(tokenizer)


@pytest.fixture(scope="module")
def reference() -> ModuleType:
    return _load_reference()


def _tool_call(raw: dict[str, Any], index: int) -> ToolCall:
    function = raw.get("function") or raw
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        arguments = {} if arguments == "" else json.loads(arguments)
    return ToolCall(
        id=str(raw.get("id", f"call-{index}")),
        function=ToolCall.FunctionBody(
            name=str(function["name"]),
            arguments=json.dumps(arguments, ensure_ascii=False),
        ),
    )


def _to_tinker_messages(messages: list[dict[str, Any]]) -> list[Message]:
    converted: list[Message] = []
    for raw in messages:
        message = dict(raw)
        if message.get("tool_calls"):
            message["tool_calls"] = [
                _tool_call(tool_call, index)
                for index, tool_call in enumerate(message["tool_calls"])
            ]
        converted.append(cast(Message, message))
    return converted


def _renderer_and_messages(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    *,
    enable_thinking: bool,
    reasoning_effort: str = "max",
) -> tuple[GLM52Renderer, list[Message]]:
    renderer: GLM52Renderer
    if enable_thinking:
        renderer = GLM52Renderer(tokenizer, reasoning_effort=reasoning_effort)
    else:
        renderer = GLM52DisableThinkingRenderer(tokenizer)
    prefix = renderer.create_conversation_prefix_with_tools(
        cast(list[ToolSpec], tools or [])
    )
    return renderer, [*prefix, *_to_tinker_messages(messages)]


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up structured data",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
            },
            "strict": True,
        },
    },
    {
        "name": "deferred_tool",
        "description": "Must not be rendered",
        "parameters": {"type": "object", "properties": {}},
        "defer_loading": True,
    },
]


GENERATION_CASES = [
    pytest.param(
        [{"role": "user", "content": " hi "}],
        None,
        {},
        id="thinking-max",
    ),
    pytest.param(
        [
            {"role": "system", "content": "Be exact."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "inspect "},
                    {"type": "image_url", "image_url": {"url": "unused"}},
                    " now",
                ],
            },
        ],
        None,
        {"reasoning_effort": "high"},
        id="high-system-media-unicode",
    ),
    pytest.param(
        [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": " a1 "},
            {"role": "user", "content": "q2"},
        ],
        None,
        {"enable_thinking": False},
        id="disabled-multiturn",
    ),
    pytest.param(
        [
            {"role": "user", "content": "q1"},
            {
                "role": "assistant",
                "reasoning_content": "推理 1",
                "content": " a1 ",
            },
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "q3"},
        ],
        None,
        {},
        id="mixed-thinking-history",
    ),
    pytest.param(
        [
            {"role": "system", "content": "Use tools."},
            {"role": "user", "content": "查天气"},
            {
                "role": "assistant",
                "reasoning_content": "先查数据",
                "content": "calling",
                "tool_calls": [
                    {
                        "id": "call-lookup",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": json.dumps(
                                {
                                    "query": "上海 & <杭州>",
                                    "limit": 2,
                                    "flags": [True, None],
                                    "nested": {"x": 1},
                                },
                                ensure_ascii=False,
                            ),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "name": "lookup",
                "tool_call_id": "call-lookup",
                "content": [{"output": "晴"}, {"output": "23°C"}],
            },
            {"role": "tool", "content": "humidity=50%"},
            {
                "role": "assistant",
                "reasoning_content": "组织回答",
                "content": "结果已返回。",
            },
            {"role": "user", "content": "继续"},
        ],
        TOOLS,
        {},
        id="tools-consecutive-results",
    ),
]


@pytest.mark.parametrize(("messages", "tools", "overrides"), GENERATION_CASES)
def test_generation_tokens_match_pinned_sft_template(
    real_tokenizer: Any,
    reference: ModuleType,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    overrides: dict[str, Any],
) -> None:
    prepared, kwargs = reference.prepare_messages_for_render(
        messages, explicit_kwargs=overrides
    )
    reasoning_effort = overrides.get("reasoning_effort", "max")
    expected_text, _ = reference.render_chat_with_assistant_spans(
        real_tokenizer,
        prepared,
        tools=tools,
        add_generation_prompt=True,
        reasoning_effort=reasoning_effort,
        **kwargs,
    )
    renderer, converted = _renderer_and_messages(
        real_tokenizer,
        messages,
        tools,
        enable_thinking=kwargs["enable_thinking"],
        reasoning_effort=reasoning_effort,
    )

    actual_tokens = renderer.build_generation_prompt(converted).to_ints()
    expected_tokens = real_tokenizer.encode(expected_text, add_special_tokens=False)
    assert actual_tokens == expected_tokens
    assert real_tokenizer.decode(actual_tokens) == expected_text


SFT_CASES = [
    pytest.param(
        [
            {"role": "user", "content": "q1"},
            {
                "role": "assistant",
                "reasoning_content": "r1",
                "content": " a1 ",
            },
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ],
        None,
        {},
        id="thinking-mixed-final-assistant",
    ),
    pytest.param(
        [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "assistant", "content": "a2"},
            {"role": "system", "content": "audit"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a3"},
        ],
        None,
        {"enable_thinking": False},
        id="disabled-adjacent-assistant-system",
    ),
    pytest.param(
        [
            {"role": "user", "content": "查"},
            {
                "role": "assistant",
                "reasoning_content": "r",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "lookup",
                            "arguments": {
                                "query": "A/B",
                                "limit": 3,
                                "nested": {"ok": True},
                            },
                        }
                    }
                ],
            },
            {"role": "tool", "content": "result-1"},
            {"role": "tool", "content": [{"output": "result-2"}]},
            {
                "role": "assistant",
                "reasoning_content": "done",
                "content": "final",
            },
        ],
        TOOLS,
        {},
        id="tool-boundaries-and-final-suffix",
    ),
    pytest.param(
        [
            {"role": "user", "content": "load the deferred tool"},
            {
                "role": "assistant",
                "reasoning_content": "request its definition",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "lookup",
                            "arguments": {"query": "deferred_tool"},
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "content": [{"type": "tool_reference", "name": "deferred_tool"}],
            },
            {
                "role": "assistant",
                "reasoning_content": "definition loaded",
                "content": "ready",
            },
        ],
        TOOLS,
        {},
        id="deferred-tool-reference",
    ),
    pytest.param(
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "reasoning_content": "", "content": "a"},
            {"role": "user", "content": "terminal user"},
        ],
        None,
        {},
        id="terminal-user-no-final-suffix",
    ),
]


@pytest.mark.parametrize(("messages", "tools", "overrides"), SFT_CASES)
def test_sft_tokens_and_loss_weights_match_original_training_code(
    real_tokenizer: Any,
    reference: ModuleType,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    overrides: dict[str, Any],
) -> None:
    prepared, kwargs = reference.prepare_messages_for_render(
        messages, explicit_kwargs=overrides
    )
    reasoning_effort = overrides.get("reasoning_effort", "max")
    expected_text, spans = reference.render_chat_with_assistant_spans(
        real_tokenizer,
        prepared,
        tools=tools,
        add_generation_prompt=False,
        reasoning_effort=reasoning_effort,
        **kwargs,
    )
    encoding = real_tokenizer(
        expected_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    expected_tokens = list(encoding["input_ids"])
    expected_weights = reference.token_weights_for_mask_spec(
        expected_text,
        list(encoding["offset_mapping"]),
        spans,
    )
    expected_tokens, expected_weights = reference.append_final_boundary_token_for_sft(
        expected_tokens,
        expected_weights,
        cleaned_text=expected_text,
        last_role=messages[-1]["role"] if messages else None,
        tokenizer=real_tokenizer,
    )

    renderer, converted = _renderer_and_messages(
        real_tokenizer,
        messages,
        tools,
        enable_thinking=kwargs["enable_thinking"],
        reasoning_effort=reasoning_effort,
    )
    actual_input, actual_weights = renderer.build_supervised_example(
        converted, TrainOnWhat.ALL_ASSISTANT_MESSAGES
    )

    assert actual_input.to_ints() == expected_tokens
    assert [float(weight) for weight in actual_weights.tolist()] == expected_weights
    assert real_tokenizer.decode(actual_input.to_ints()) == (
        expected_text
        + (
            "<|user|>"
            if expected_tokens[-1] == real_tokenizer.convert_tokens_to_ids("<|user|>")
            and not expected_text.endswith("<|user|>")
            else ""
        )
    )


def test_parse_render_extension_property_with_real_tokenizer(
    real_tokenizer: Any,
) -> None:
    renderer = GLM52Renderer(real_tokenizer)
    initial = [Message(role="user", content="weather?")]
    prompt = renderer.build_generation_prompt(initial).to_ints()
    sampled = real_tokenizer.encode(
        "check</think>calling"
        "<tool_call>lookup<arg_key>query</arg_key>"
        "<arg_value>上海</arg_value></tool_call><|observation|>",
        add_special_tokens=False,
    )
    assistant, termination = renderer.parse_response(sampled)
    assert termination.is_clean

    continued = renderer.build_generation_prompt(
        [
            *initial,
            assistant,
            Message(role="tool", content="sunny", tool_call_id="call-0"),
        ]
    ).to_ints()
    expected_prefix = [
        *prompt,
        *sampled,
        *real_tokenizer.encode(
            "<tool_response>sunny</tool_response><|assistant|><think>",
            add_special_tokens=False,
        ),
    ]
    assert continued == expected_prefix
