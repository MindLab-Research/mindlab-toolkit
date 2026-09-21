"""README shared-contract cases: reasoning source + True/False mask.

Covers the table in ``src/mint/renderers/README.md`` for both families.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip(
    "tinker_cookbook.renderers",
    reason="install mindlab-toolkit[test] to run renderer contract tests",
)

_RENDERER_TESTS = Path(__file__).resolve().parent
if str(_RENDERER_TESTS) not in sys.path:
    sys.path.insert(0, str(_RENDERER_TESTS))

from tinker_cookbook.exceptions import RendererError
from tinker_cookbook.renderers import Message, TrainOnWhat, get_renderer

from mint.renderers.glm52 import GLM52DisableThinkingRenderer, GLM52Renderer
from mint.renderers.qwen35 import QWEN35_RENDERER, Qwen35ReasoningRenderer
from mint.reasoning_source import split_serving_fragment, visible_has_think_markup

from test_glm52 import CharacterTokenizer
from test_qwen35 import _QwenCharTokenizer, _supervised_ids_weights

FRAGMENT = "REASON</think>COMMONCONTENT"
FRAGMENT_THEN_MENTION = "REASON</think>see <think>x</think>"
MENTION = "see <think>x</think> please"
COPIED_BLOCK = "<think>\nREASON\n</think>\n\nANSWER"
PLAIN = "ANSWER"


def _trained(tokenizer, model_input, weights) -> str:
    ints = model_input.to_ints()
    return tokenizer.decode([ints[i] for i, w in enumerate(weights.tolist()) if w == 1.0])


def _glm_pair(tokenizer, messages):
    supervised = GLM52Renderer(tokenizer)
    masked = GLM52Renderer(tokenizer, supervise_reasoning=False)
    return (
        supervised.build_supervised_example(list(messages)),
        masked.build_supervised_example(list(messages)),
        supervised,
        masked,
    )


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (FRAGMENT, ("REASON", "COMMONCONTENT")),
        (FRAGMENT_THEN_MENTION, ("REASON", "see <think>x</think>")),
        ("REASON</think>see <think>x</think>", ("REASON", "see <think>x</think>")),
        ("REASON\n</think>\n\nANSWER", ("REASON\n", "\n\nANSWER")),
        ("</think>ANSWER", ("", "ANSWER")),
        ("</think>", ("", "")),
        ("REASON</think>", ("REASON", "")),
        (MENTION, None),
        (COPIED_BLOCK, None),
        (PLAIN, None),
        ("<think>REASON</think>ANSWER", None),
        ("<think></think>ANSWER", None),
        (["REASON</think>COMMONCONTENT"], ("REASON", "COMMONCONTENT")),
        ([{"type": "text", "text": FRAGMENT}], ("REASON", "COMMONCONTENT")),
        ([{"text": FRAGMENT}], ("REASON", "COMMONCONTENT")),
        ([{"type": "text", "text": MENTION}], None),
        ([{"type": "text", "text": FRAGMENT}, {"type": "image", "image": "x"}], None),
        (
            [{"type": "image", "image": "x", "text": FRAGMENT}],
            None,
        ),
    ],
)
def test_split_serving_fragment_matches_readme_table(content, expected):
    assert split_serving_fragment(content) == expected


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (PLAIN, False),
        ("<think>", True),
        ("</think>ANSWER", True),
        (MENTION, True),
        ([{"type": "text", "text": "</think>ANSWER"}], True),
        (
            [
                {"type": "text", "text": "</think>ANSWER"},
                {"type": "image", "image": "x"},
            ],
            True,
        ),
        (
            [
                {"type": "text", "text": "ANSWER"},
                {"type": "image", "image": "x"},
            ],
            False,
        ),
        (
            [{"type": "image", "image": "x", "text": "</think>ANSWER"}],
            False,
        ),
    ],
)
def test_visible_has_think_markup_covers_close_and_leftover_abort(content, expected):
    assert visible_has_think_markup(content) is expected


def test_glm_plain_true_supervises_empty_close(tokenizer: CharacterTokenizer):
    messages = [Message(role="user", content="q"), {"role": "assistant", "content": PLAIN}]
    (sup_in, sup_w), (msk_in, msk_w), *_ = _glm_pair(tokenizer, messages)
    assert msk_in.to_ints() == sup_in.to_ints()
    assert "<think></think>ANSWER" in tokenizer.decode(sup_in.to_ints())
    assert _trained(tokenizer, sup_in, sup_w) == "</think>ANSWER<|user|>"
    assert _trained(tokenizer, msk_in, msk_w) == "ANSWER<|user|>"


def test_glm_fragment_promotes_like_reasoning_content(tokenizer: CharacterTokenizer):
    fragment = [
        Message(role="user", content="q"),
        {"role": "assistant", "content": FRAGMENT},
    ]
    field = [
        Message(role="user", content="q"),
        {"role": "assistant", "reasoning_content": "REASON", "content": "COMMONCONTENT"},
    ]
    (frag_sup, frag_sup_w), (frag_msk, frag_msk_w), *_ = _glm_pair(tokenizer, fragment)
    (field_sup, field_sup_w), (field_msk, field_msk_w), *_ = _glm_pair(tokenizer, field)

    assert frag_sup.to_ints() == field_sup.to_ints() == frag_msk.to_ints()
    assert "<think>REASON</think>COMMONCONTENT" in tokenizer.decode(frag_sup.to_ints())
    assert _trained(tokenizer, frag_msk, frag_msk_w) == "COMMONCONTENT<|user|>"
    assert _trained(tokenizer, frag_sup, frag_sup_w) == "REASON</think>COMMONCONTENT<|user|>"
    assert _trained(tokenizer, field_msk, field_msk_w) == _trained(
        tokenizer, frag_msk, frag_msk_w
    )


def test_glm_fragment_then_mention_keeps_tags_in_answer(tokenizer: CharacterTokenizer):
    messages = [
        Message(role="user", content="q"),
        {"role": "assistant", "content": FRAGMENT_THEN_MENTION},
    ]
    (sup_in, sup_w), (msk_in, msk_w), *_ = _glm_pair(tokenizer, messages)
    assert msk_in.to_ints() == sup_in.to_ints()
    assert "<think>REASON</think>see <think>x</think>" in tokenizer.decode(sup_in.to_ints())
    assert _trained(tokenizer, msk_in, msk_w) == "see <think>x</think><|user|>"
    assert "REASON" in _trained(tokenizer, sup_in, sup_w)
    assert "see <think>x</think>" in _trained(tokenizer, sup_in, sup_w)


def test_glm_mention_and_copied_block_are_plain_answers(tokenizer: CharacterTokenizer):
    for content in (
        MENTION,
        COPIED_BLOCK,
        [{"type": "text", "text": MENTION}],
        [{"type": "text", "text": COPIED_BLOCK}],
    ):
        messages = [
            Message(role="user", content="q"),
            {"role": "assistant", "content": content},
        ]
        (sup_in, sup_w), (msk_in, msk_w), *_ = _glm_pair(tokenizer, messages)
        full = tokenizer.decode(sup_in.to_ints())
        assert msk_in.to_ints() == sup_in.to_ints()
        visible = content if isinstance(content, str) else content[0]["text"]
        assert visible in full
        assert f"<think></think>{visible}" in full
        assert _trained(tokenizer, msk_in, msk_w) == f"{visible}<|user|>"
        assert _trained(tokenizer, sup_in, sup_w) == f"</think>{visible}<|user|>"


def test_glm_field_or_thinking_part_wins_over_fragment_text(tokenizer: CharacterTokenizer):
    field = [
        Message(role="user", content="q"),
        {
            "role": "assistant",
            "reasoning_content": "why",
            "content": FRAGMENT,
        },
    ]
    part = [
        Message(role="user", content="q"),
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "why"},
                {"type": "text", "text": FRAGMENT},
            ],
        },
    ]
    for messages in (field, part):
        (sup_in, _), (msk_in, msk_w), *_ = _glm_pair(tokenizer, messages)
        assert msk_in.to_ints() == sup_in.to_ints()
        assert "<think>why</think>REASON</think>COMMONCONTENT" in tokenizer.decode(
            sup_in.to_ints()
        )
        assert _trained(tokenizer, msk_in, msk_w) == "REASON</think>COMMONCONTENT<|user|>"


def test_glm_list_string_and_text_part_are_fragments(tokenizer: CharacterTokenizer):
    field = [
        Message(role="user", content="q"),
        {"role": "assistant", "reasoning_content": "REASON", "content": "COMMONCONTENT"},
    ]
    (field_sup, _), *_ = _glm_pair(tokenizer, field)
    for content in ([FRAGMENT], [{"type": "text", "text": FRAGMENT}]):
        messages = [
            Message(role="user", content="q"),
            {"role": "assistant", "content": content},
        ]
        (sup_in, sup_w), (msk_in, msk_w), *_ = _glm_pair(tokenizer, messages)
        assert sup_in.to_ints() == field_sup.to_ints()
        assert _trained(tokenizer, msk_in, msk_w) == "COMMONCONTENT<|user|>"
        assert _trained(tokenizer, sup_in, sup_w) == "REASON</think>COMMONCONTENT<|user|>"


@pytest.mark.parametrize("content", [FRAGMENT, "</think>ANSWER"])
def test_glm_disable_thinking_rejects_fragment(
    tokenizer: CharacterTokenizer, content: str
):
    renderer = GLM52DisableThinkingRenderer(tokenizer)
    with pytest.raises(RendererError, match="thinking is disabled"):
        renderer.build_supervised_example(
            [{"role": "assistant", "content": content}]
        )


def test_glm_empty_leftover_matches_empty_reasoning_field(tokenizer: CharacterTokenizer):
    leftover = [
        Message(role="user", content="q"),
        {"role": "assistant", "content": "</think>ANSWER"},
    ]
    field = [
        Message(role="user", content="q"),
        {"role": "assistant", "reasoning_content": "", "content": "ANSWER"},
    ]
    (leftover_sup, leftover_sup_w), (leftover_msk, leftover_msk_w), *_ = _glm_pair(
        tokenizer, leftover
    )
    (field_sup, field_sup_w), (field_msk, field_msk_w), *_ = _glm_pair(tokenizer, field)
    assert leftover_sup.to_ints() == field_sup.to_ints() == leftover_msk.to_ints()
    assert leftover_sup.to_ints() == field_msk.to_ints()
    assert "<think></think>ANSWER" in tokenizer.decode(leftover_sup.to_ints())
    assert _trained(tokenizer, leftover_msk, leftover_msk_w) == "ANSWER<|user|>"
    assert _trained(tokenizer, leftover_sup, leftover_sup_w) == "</think>ANSWER<|user|>"
    assert _trained(tokenizer, field_msk, field_msk_w) == _trained(
        tokenizer, leftover_msk, leftover_msk_w
    )


def test_glm_leftover_without_answer_matches_empty_content_field(
    tokenizer: CharacterTokenizer,
):
    leftover = [
        Message(role="user", content="q"),
        {"role": "assistant", "content": "REASON</think>"},
    ]
    field = [
        Message(role="user", content="q"),
        {"role": "assistant", "reasoning_content": "REASON", "content": ""},
    ]
    (leftover_sup, leftover_sup_w), (leftover_msk, leftover_msk_w), *_ = _glm_pair(
        tokenizer, leftover
    )
    (field_sup, field_sup_w), (field_msk, field_msk_w), *_ = _glm_pair(tokenizer, field)
    assert leftover_sup.to_ints() == field_sup.to_ints() == leftover_msk.to_ints()
    assert "<think>REASON</think>" in tokenizer.decode(leftover_sup.to_ints())
    assert _trained(tokenizer, leftover_msk, leftover_msk_w) == "<|user|>"
    assert _trained(tokenizer, leftover_sup, leftover_sup_w) == "REASON</think><|user|>"
    assert leftover_sup_w.tolist() == field_sup_w.tolist()
    assert leftover_msk_w.tolist() == field_msk_w.tolist()


def test_glm_generation_prompt_ignores_supervise_reasoning(tokenizer: CharacterTokenizer):
    messages = [Message(role="user", content="q")]
    a = GLM52Renderer(tokenizer).build_generation_prompt(messages)
    b = GLM52Renderer(tokenizer, supervise_reasoning=False).build_generation_prompt(
        messages
    )
    assert a.to_ints() == b.to_ints()


def test_qwen_plain_true_false_empty_think_asymmetry():
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": PLAIN},
    ]
    ids_true, w_true = _supervised_ids_weights(True, messages)
    ids_false, w_false = _supervised_ids_weights(False, messages)
    tok = _QwenCharTokenizer()
    assert tok.decode(ids_true).count("<think>\n</think>") == 1
    assert tok.decode(ids_false).count("<think>\n\n</think>") == 1
    assert ids_true != ids_false
    assert tok.decode([t for t, w in zip(ids_true, w_true) if w == 1.0]) == (
        "</think>\n\nANSWER<|im_end|>"
    )
    assert tok.decode([t for t, w in zip(ids_false, w_false) if w == 1.0]) == (
        "ANSWER<|im_end|>"
    )


@pytest.mark.parametrize(
    "content",
    [FRAGMENT, [{"type": "text", "text": FRAGMENT}], [FRAGMENT]],
)
def test_qwen_fragment_promotes_like_reasoning_content(content):
    fragment = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": content},
    ]
    field = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "reasoning_content": "REASON", "content": "COMMONCONTENT"},
    ]
    frag_true, frag_true_w = _supervised_ids_weights(True, fragment)
    frag_false, frag_false_w = _supervised_ids_weights(False, fragment)
    field_true, _ = _supervised_ids_weights(True, field)
    tok = _QwenCharTokenizer()
    assert frag_true == frag_false == field_true
    assert "<think>\nREASON\n</think>\n\nCOMMONCONTENT" in tok.decode(frag_true)
    assert tok.decode([t for t, w in zip(frag_false, frag_false_w) if w == 1.0]) == (
        "COMMONCONTENT<|im_end|>"
    )
    assert tok.decode([t for t, w in zip(frag_true, frag_true_w) if w == 1.0]) == (
        "REASON\n</think>\n\nCOMMONCONTENT<|im_end|>"
    )


def test_qwen_fragment_then_mention_keeps_tags_in_answer():
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": FRAGMENT_THEN_MENTION},
    ]
    ids_true, w_true = _supervised_ids_weights(True, messages)
    ids_false, w_false = _supervised_ids_weights(False, messages)
    tok = _QwenCharTokenizer()
    assert ids_true == ids_false
    assert "<think>\nREASON\n</think>\n\nsee <think>x</think>" in tok.decode(ids_true)
    assert tok.decode([t for t, w in zip(ids_false, w_false) if w == 1.0]) == (
        "see <think>x</think><|im_end|>"
    )
    assert "REASON" in tok.decode([t for t, w in zip(ids_true, w_true) if w == 1.0])


@pytest.mark.parametrize(
    "content",
    [MENTION, [{"type": "text", "text": MENTION}], [MENTION], [{"text": MENTION}]],
)
def test_qwen_mention_is_plain_and_false_does_not_double_wrap(content):
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": content},
    ]
    ids_true, w_true = _supervised_ids_weights(True, messages)
    ids_false, w_false = _supervised_ids_weights(False, messages)
    tok = _QwenCharTokenizer()
    assert MENTION in tok.decode(ids_true)
    assert "<think>\n\n</think>" not in tok.decode(ids_false)
    assert tok.decode(ids_false).count("<think>") == 1
    assert tok.decode(ids_true).count("<think>\n</think>") == 1
    assert tok.decode([t for t, w in zip(ids_false, w_false) if w == 1.0]) == (
        f"{MENTION}<|im_end|>"
    )
    assert tok.decode([t for t, w in zip(ids_true, w_true) if w == 1.0]) == (
        f"</think>\n\n{MENTION}<|im_end|>"
    )


def test_qwen_empty_leftover_matches_empty_thinking_part():
    leftover = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "</think>ANSWER"},
    ]
    field = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": ""},
                {"type": "text", "text": "ANSWER"},
            ],
        },
    ]
    leftover_true, leftover_true_w = _supervised_ids_weights(True, leftover)
    leftover_false, leftover_false_w = _supervised_ids_weights(False, leftover)
    field_true, field_true_w = _supervised_ids_weights(True, field)
    field_false, field_false_w = _supervised_ids_weights(False, field)
    tok = _QwenCharTokenizer()
    assert leftover_true == leftover_false == field_true == field_false
    assert tok.decode(leftover_true).count("<think>\n</think>") == 1
    assert tok.decode([t for t, w in zip(leftover_false, leftover_false_w) if w == 1.0]) == (
        "ANSWER<|im_end|>"
    )
    assert tok.decode([t for t, w in zip(leftover_true, leftover_true_w) if w == 1.0]) == (
        "</think>\n\nANSWER<|im_end|>"
    )
    assert leftover_true_w == field_true_w
    assert leftover_false_w == field_false_w


def test_qwen_leftover_without_answer_matches_empty_thinking_part():
    leftover = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "REASON</think>"},
    ]
    field = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "REASON"}],
        },
    ]
    leftover_true, leftover_true_w = _supervised_ids_weights(True, leftover)
    leftover_false, leftover_false_w = _supervised_ids_weights(False, leftover)
    field_true, field_true_w = _supervised_ids_weights(True, field)
    field_false, field_false_w = _supervised_ids_weights(False, field)
    tok = _QwenCharTokenizer()
    assert leftover_true == leftover_false == field_true == field_false
    assert "<think>\nREASON\n</think>" in tok.decode(leftover_true)
    assert tok.decode([t for t, w in zip(leftover_false, leftover_false_w) if w == 1.0]) == (
        "<|im_end|>"
    )
    assert leftover_true_w == field_true_w
    assert leftover_false_w == field_false_w


@pytest.mark.parametrize(
    "content",
    [
        COPIED_BLOCK,
        [{"type": "text", "text": COPIED_BLOCK}],
        [COPIED_BLOCK],
        [{"text": COPIED_BLOCK}],
    ],
)
def test_qwen_copied_block_is_plain_and_false_does_not_double_wrap(content):
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": content},
    ]
    ids_true, w_true = _supervised_ids_weights(True, messages)
    ids_false, w_false = _supervised_ids_weights(False, messages)
    tok = _QwenCharTokenizer()
    full_false = tok.decode(ids_false)
    assert "<think>\n\n</think>\n\n<think>" not in full_false
    assert full_false.count("<think>") == 1
    assert tok.decode([t for t, w in zip(ids_false, w_false) if w == 1.0]) == (
        f"{COPIED_BLOCK}<|im_end|>"
    )
    assert tok.decode(ids_true).count("<think>\n</think>") == 1
    assert tok.decode([t for t, w in zip(ids_true, w_true) if w == 1.0]) == (
        f"</think>\n\n{COPIED_BLOCK}<|im_end|>"
    )


def test_qwen_field_wins_over_fragment_text():
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "reasoning_content": "why", "content": FRAGMENT},
    ]
    ids_true, _ = _supervised_ids_weights(True, messages)
    ids_false, w_false = _supervised_ids_weights(False, messages)
    tok = _QwenCharTokenizer()
    assert ids_true == ids_false
    assert "<think>\nwhy\n</think>\n\nREASON</think>COMMONCONTENT" in tok.decode(ids_true)
    assert tok.decode([t for t, w in zip(ids_false, w_false) if w == 1.0]) == (
        "REASON</think>COMMONCONTENT<|im_end|>"
    )


def test_qwen_thinking_part_wins_over_fragment_text():
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "why"},
                {"type": "text", "text": FRAGMENT},
            ],
        },
    ]
    ids_true, _ = _supervised_ids_weights(True, messages)
    ids_false, w_false = _supervised_ids_weights(False, messages)
    tok = _QwenCharTokenizer()
    assert ids_true == ids_false
    assert "<think>\nwhy\n</think>\n\nREASON</think>COMMONCONTENT" in tok.decode(ids_true)
    assert tok.decode([t for t, w in zip(ids_false, w_false) if w == 1.0]) == (
        "REASON</think>COMMONCONTENT<|im_end|>"
    )


def test_qwen_generation_prompt_ignores_supervise_reasoning():
    tok = _QwenCharTokenizer()
    messages = [{"role": "user", "content": "q"}]
    a = Qwen35ReasoningRenderer(tok, supervise_reasoning=True).build_generation_prompt(
        messages
    )
    b = Qwen35ReasoningRenderer(tok, supervise_reasoning=False).build_generation_prompt(
        messages
    )
    assert a.to_ints() == b.to_ints()


def test_glm_reasoning_payload_close_tag_is_inert(tokenizer: CharacterTokenizer):
    messages = [
        Message(role="user", content="q"),
        {
            "role": "assistant",
            "reasoning_content": "A</think>B",
            "content": "ANSWER",
        },
    ]
    (sup_in, sup_w), (msk_in, msk_w), *_ = _glm_pair(tokenizer, messages)
    full = tokenizer.decode(sup_in.to_ints())
    assert msk_in.to_ints() == sup_in.to_ints()
    assert "<think>A</think>B</think>ANSWER" in full
    assert _trained(tokenizer, msk_in, msk_w) == "ANSWER<|user|>"
    assert _trained(tokenizer, sup_in, sup_w) == "A</think>B</think>ANSWER<|user|>"


def test_qwen_reasoning_payload_close_tag_is_inert():
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "reasoning_content": "A</think>B",
            "content": "ANSWER",
        },
    ]
    ids_true, w_true = _supervised_ids_weights(True, messages)
    ids_false, w_false = _supervised_ids_weights(False, messages)
    tok = _QwenCharTokenizer()
    assert ids_true == ids_false
    assert "<think>\nA</think>B\n</think>\n\nANSWER" in tok.decode(ids_true)
    assert tok.decode([t for t, w in zip(ids_false, w_false) if w == 1.0]) == (
        "ANSWER<|im_end|>"
    )
    assert tok.decode([t for t, w in zip(ids_true, w_true) if w == 1.0]) == (
        "A</think>B\n</think>\n\nANSWER<|im_end|>"
    )


def test_glm_strips_visible_answer_qwen_does_not(tokenizer: CharacterTokenizer):
    padded = " ANSWER "
    glm_messages = [
        Message(role="user", content="q"),
        {"role": "assistant", "content": padded},
    ]
    (glm_in, _), *_ = _glm_pair(tokenizer, glm_messages)
    assert " ANSWER " not in tokenizer.decode(glm_in.to_ints())
    assert "</think>ANSWER<|user|>" in tokenizer.decode(glm_in.to_ints())

    qwen_messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": padded},
    ]
    ids, _ = _supervised_ids_weights(True, qwen_messages)
    assert " ANSWER " in _QwenCharTokenizer().decode(ids)


def test_glm_all_tokens_false_still_trains_cot(tokenizer: CharacterTokenizer):
    renderer = GLM52Renderer(tokenizer, supervise_reasoning=False)
    messages = [
        Message(role="user", content="q"),
        {"role": "assistant", "reasoning_content": "REASON", "content": "ANSWER"},
    ]
    model_input, weights = renderer.build_supervised_example(
        messages, train_on_what=TrainOnWhat.ALL_TOKENS
    )
    trained = _trained(tokenizer, model_input, weights)
    assert "REASON" in trained
    assert "ANSWER" in trained


def test_qwen_all_tokens_false_still_trains_cot():
    tok = _QwenCharTokenizer()
    renderer = Qwen35ReasoningRenderer(tok, supervise_reasoning=False)
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "reasoning_content": "REASON", "content": "ANSWER"},
    ]
    model_input, weights = renderer.build_supervised_example(
        messages, train_on_what=TrainOnWhat.ALL_TOKENS
    )
    weight_list = weights.tolist() if hasattr(weights, "tolist") else list(weights)
    trained = tok.decode(
        [t for t, w in zip(model_input.to_ints(), weight_list) if w == 1.0]
    )
    assert "REASON" in trained
    assert "ANSWER" in trained


def test_qwen_keeps_history_think_blocks_and_factory_defaults():
    tok = _QwenCharTokenizer()
    messages = [
        {"role": "user", "content": "q1"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "HIST"},
                {"type": "text", "text": "a1"},
            ],
        },
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    kept = Qwen35ReasoningRenderer(tok)
    stripped = Qwen35ReasoningRenderer(tok, strip_thinking_from_history=True)
    kept_ids, _ = kept.build_supervised_example(
        messages, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES
    )
    stripped_ids, _ = stripped.build_supervised_example(
        messages, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES
    )
    assert "HIST" in tok.decode(kept_ids.to_ints())
    assert "HIST" not in tok.decode(stripped_ids.to_ints())
    assert "a1" in tok.decode(stripped_ids.to_ints())

    factory = get_renderer(QWEN35_RENDERER, tok, model_name="Qwen/Qwen3.5")
    assert isinstance(factory, Qwen35ReasoningRenderer)
    assert factory.strip_thinking_from_history is False


@pytest.fixture
def tokenizer() -> CharacterTokenizer:
    return CharacterTokenizer()
