"""Unit tests for the MinT Qwen3.5 reasoning renderer helpers.

* ``_normalize_assistant_message`` folds ``reasoning_content``, a ThinkingPart,
  or a serving leftover ``REASON</think>ANSWER`` into one thinking part.
* ``_mask_think_span`` / ``_mask_think_open`` move think tokens from ``output``
  into ``header`` without changing the concatenated sequence.
"""

import pytest
import tinker
from tinker_cookbook.exceptions import RendererError
from tinker_cookbook.renderers.base import RenderedMessage

from mint.renderers import qwen35

_CLOSE = 99  # stand-in token id for "</think>"
_WS = 7  # stand-in token id for a whitespace separator


class _FakeTokenizer:
    """Minimal tokenizer: only the ops ``_mask_think_span`` relies on."""

    def encode(self, text, add_special_tokens=False):
        assert text == "</think>"
        return [_CLOSE]

    def decode(self, ids):
        out = []
        for tid in ids:
            if tid == _CLOSE:
                out.append("</think>")
            elif tid == _WS:
                out.append("\n\n")
            else:
                out.append("x")
        return "".join(out)


def _chunk(tokens):
    return tinker.types.EncodedTextChunk(tokens=list(tokens))


def test_promote_reasoning_content_folds_into_thinking_part():
    message = {"role": "assistant", "content": "answer", "reasoning_content": "why"}

    promoted = qwen35._promote_reasoning_content(message)

    assert "reasoning_content" not in promoted
    assert promoted["content"] == [
        {"type": "thinking", "thinking": "why"},
        {"type": "text", "text": "answer"},
    ]
    # original message is untouched
    assert message["reasoning_content"] == "why"


def test_promote_reasoning_content_noop_without_reasoning():
    message = {"role": "assistant", "content": "answer"}
    assert qwen35._promote_reasoning_content(message) == {
        "role": "assistant",
        "content": "answer",
    }


def test_inject_empty_thinking_prepends_blank_thinking_part():
    # No-reasoning answer: empty ThinkingPart puts </think> in supervised output.
    message = {"role": "assistant", "content": "answer"}
    injected = qwen35._inject_empty_thinking(message)
    assert injected["content"] == [
        {"type": "thinking", "thinking": ""},
        {"type": "text", "text": "answer"},
    ]
    assert message["content"] == "answer"


def test_inject_empty_thinking_noop_when_thinking_present():
    message = {
        "role": "assistant",
        "content": [{"type": "thinking", "thinking": "t"}, {"type": "text", "text": "a"}],
    }
    injected = qwen35._inject_empty_thinking(message)
    assert injected["content"][0] == {"type": "thinking", "thinking": "t"}
    assert not any(
        p == {"type": "thinking", "thinking": ""} for p in injected["content"]
    )


def test_normalize_keeps_thinking_part_without_field():
    message = {
        "role": "assistant",
        "content": [{"type": "thinking", "thinking": "already"}, {"type": "text", "text": "a"}],
    }
    reasoning, has_structure, promoted = qwen35._normalize_assistant_message(message)
    assert reasoning == "already"
    assert has_structure is True
    assert promoted["content"][0] == {"type": "thinking", "thinking": "already"}


def test_normalize_literal_think_tags_are_not_structure():
    for content in (
        "<think>REASON</think>ANSWER",
        "<think>\nREASON\n</think>\n\nANSWER",
        "<think>A</think>B<think>C</think>",
        "ANSWER<think>REASON</think>",
        "see <think>x</think> please",
    ):
        message = {"role": "assistant", "content": content}
        reasoning, has_structure, promoted = qwen35._normalize_assistant_message(message)
        assert reasoning == ""
        assert has_structure is False
        assert promoted["content"] == content


@pytest.mark.parametrize(
    "content, text",
    [
        (["<think>REASON</think>ANSWER"], "<think>REASON</think>ANSWER"),
        ([{"text": "<think>REASON</think>ANSWER"}], "<think>REASON</think>ANSWER"),
        ([{"type": "text", "text": "<think>REASON</think>ANSWER"}], "<think>REASON</think>ANSWER"),
    ],
)
def test_normalize_complete_block_list_encodings_stay_answers(content, text):
    original = {"role": "assistant", "content": content}
    reasoning, has_structure, promoted = qwen35._normalize_assistant_message(original)
    assert reasoning == ""
    assert has_structure is False
    assert promoted["content"] == [{"type": "text", "text": text}]
    assert original["content"] == content


@pytest.mark.parametrize(
    "content",
    [
        "REASON</think>COMMONCONTENT",
        [{"type": "text", "text": "REASON</think>COMMONCONTENT"}],
        ["REASON</think>COMMONCONTENT"],
    ],
)
def test_normalize_serving_fragment_becomes_thinking_part(content):
    reasoning, has_structure, promoted = qwen35._normalize_assistant_message(
        {"role": "assistant", "content": content}
    )
    assert reasoning == "REASON"
    assert has_structure is True
    assert promoted["content"] == [
        {"type": "thinking", "thinking": "REASON"},
        {"type": "text", "text": "COMMONCONTENT"},
    ]


def test_normalize_empty_leftover_is_empty_thinking_part():
    reasoning, has_structure, promoted = qwen35._normalize_assistant_message(
        {"role": "assistant", "content": "</think>ANSWER"}
    )
    assert reasoning == ""
    assert has_structure is True
    assert promoted["content"] == [
        {"type": "thinking", "thinking": ""},
        {"type": "text", "text": "ANSWER"},
    ]


def test_normalize_rejects_field_and_thinking_part():
    message = {
        "role": "assistant",
        "content": [{"type": "thinking", "thinking": "already"}, {"type": "text", "text": "a"}],
        "reasoning_content": "also-here",
    }
    with pytest.raises(RendererError, match="not both"):
        qwen35._normalize_assistant_message(message)


def test_normalize_field_keeps_literal_think_tags_as_answer():
    message = {
        "role": "assistant",
        "content": "<think>REASON</think>ANSWER",
        "reasoning_content": "also-here",
    }
    reasoning, has_structure, promoted = qwen35._normalize_assistant_message(message)
    assert reasoning == "also-here"
    assert has_structure is True
    assert promoted["content"] == [
        {"type": "thinking", "thinking": "also-here"},
        {"type": "text", "text": "<think>REASON</think>ANSWER"},
    ]


def test_inject_empty_thinking_keeps_literal_think_tags_as_text():
    message = {"role": "assistant", "content": "<think>REASON</think>ANSWER"}
    _, has_structure, promoted = qwen35._normalize_assistant_message(message)
    assert has_structure is False
    injected = qwen35._inject_empty_thinking(promoted)
    assert injected["content"] == [
        {"type": "thinking", "thinking": ""},
        {"type": "text", "text": "<think>REASON</think>ANSWER"},
    ]


def test_mask_think_span_moves_cot_into_header_without_changing_sequence():
    tokenizer = _FakeTokenizer()
    header = _chunk([1, 2])
    # tokens: [<think>, body, body, </think>, \n\n, answer, answer, <|im_end|>]
    output = [_chunk([10, 11, 12, _CLOSE, _WS, 20, 21, 30])]
    rendered = RenderedMessage(header=header, output=output)

    masked = qwen35._mask_think_span(rendered, tokenizer)

    header_tokens = list(masked.header.tokens)
    output_tokens = [t for chunk in masked.output for t in chunk.tokens]

    # Everything up to and including </think>\n\n is now unsupervised (header).
    assert header_tokens == [1, 2, 10, 11, 12, _CLOSE, _WS]
    # Only the visible answer (+ end token) remains trained.
    assert output_tokens == [20, 21, 30]
    # The concatenated sequence is byte-for-byte identical to the input.
    assert header_tokens + output_tokens == [1, 2, 10, 11, 12, _CLOSE, _WS, 20, 21, 30]


def test_mask_think_span_uses_formatted_block_when_body_contains_close():
    # reasoning_content "REASON</think>STILL" renders a first </think> that is
    # not the block boundary. Matching the formatted prefix keeps STILL off loss.
    block = "<think>\nREASON</think>STILL\n</think>\n\n"
    block_ids = [10, _CLOSE, 11, _CLOSE, _WS]

    class _BlockTokenizer:
        def encode(self, text, add_special_tokens=False):
            if text == block:
                return list(block_ids)
            if text == "</think>":
                return [_CLOSE]
            raise AssertionError(text)

        def decode(self, ids):
            return "".join("</think>" if t == _CLOSE else "x" for t in ids)

    rendered = RenderedMessage(
        header=_chunk([1]),
        output=[_chunk(block_ids + [20, 21])],
    )
    masked = qwen35._mask_think_span(rendered, _BlockTokenizer(), think_block=block)
    assert list(masked.header.tokens) == [1, *block_ids]
    assert [t for chunk in masked.output for t in chunk.tokens] == [20, 21]


class _MergingNlTokenizer:
    """Independent ``\\n\\n`` is 271; four newlines merge to 987 (Qwen3.5 BPE)."""

    _THINK_BLOCK = "<think>\nREASON\n\n</think>\n\n"
    THINK_OPEN = 1
    NL = 2
    REASON = 3
    CLOSE = 99
    NL2 = 271
    NL4 = 987
    ANSWER = 20

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text == "</think>":
            return [self.CLOSE]
        if text == self._THINK_BLOCK:
            return [
                self.THINK_OPEN,
                self.NL,
                self.REASON,
                self.NL2,
                self.CLOSE,
                self.NL2,
            ]
        raise AssertionError(text)

    def decode(self, ids):
        pieces = []
        for token in ids:
            if token == self.THINK_OPEN:
                pieces.append("<think>")
            elif token == self.NL:
                pieces.append("\n")
            elif token == self.REASON:
                pieces.append("REASON")
            elif token == self.CLOSE:
                pieces.append("</think>")
            elif token == self.NL2:
                pieces.append("\n\n")
            elif token == self.NL4:
                pieces.append("\n\n\n\n")
            elif token == self.ANSWER:
                pieces.append("ANSWER")
            else:
                pieces.append("?")
        return "".join(pieces)


def test_mask_think_span_covers_bpe_merged_trailing_newlines():
    # Leftover keeps answer-leading \\n\\n; format already ends with \\n\\n.
    # Merged output last think token is 987, not independently encoded 271.
    block = _MergingNlTokenizer._THINK_BLOCK
    output_ids = [
        _MergingNlTokenizer.THINK_OPEN,
        _MergingNlTokenizer.NL,
        _MergingNlTokenizer.REASON,
        _MergingNlTokenizer.NL2,
        _MergingNlTokenizer.CLOSE,
        _MergingNlTokenizer.NL4,
        _MergingNlTokenizer.ANSWER,
    ]
    rendered = RenderedMessage(header=_chunk([1]), output=[_chunk(output_ids)])
    masked = qwen35._mask_think_span(rendered, _MergingNlTokenizer(), think_block=block)
    assert list(masked.header.tokens) == [1, *output_ids[:-1]]
    assert [t for chunk in masked.output for t in chunk.tokens] == [
        _MergingNlTokenizer.ANSWER
    ]


def test_mask_think_span_raises_when_formatted_block_missing():
    rendered = RenderedMessage(header=_chunk([1]), output=[_chunk([20, 21])])
    with pytest.raises(RendererError, match="could not locate the rendered think block"):
        qwen35._mask_think_span(
            rendered, _MergingNlTokenizer(), think_block=_MergingNlTokenizer._THINK_BLOCK
        )


_OPEN = 88  # stand-in token id for "<think>"
_NL = 5  # stand-in token id for "\n"


class _OpenTokenizer:
    """Tokenizer whose ``<think>\\n`` encodes to two atomic tokens."""

    def encode(self, text, add_special_tokens=False):
        assert text == "<think>\n"
        return [_OPEN, _NL]

    def decode(self, ids):
        return "".join({_OPEN: "<think>", _NL: "\n"}.get(t, "x") for t in ids)


def test_mask_think_open_moves_opening_into_header():
    # cot=True: <think>\n is prefilled at generation, so it must sit in the
    # unsupervised header; the rest (reasoning / </think> / answer) stays trained.
    tokenizer = _OpenTokenizer()
    header = _chunk([1, 2])
    output = [_chunk([_OPEN, _NL, 10, 11, 30])]  # <think>\n THINK... answer
    rendered = RenderedMessage(header=header, output=output)

    masked = qwen35._mask_think_open(rendered, tokenizer)

    assert list(masked.header.tokens) == [1, 2, _OPEN, _NL]
    assert [t for chunk in masked.output for t in chunk.tokens] == [10, 11, 30]


def test_mask_think_open_noop_when_output_does_not_open_with_think():
    tokenizer = _OpenTokenizer()
    rendered = RenderedMessage(header=_chunk([1]), output=[_chunk([10, 20])])
    masked = qwen35._mask_think_open(rendered, tokenizer)
    assert list(masked.header.tokens) == [1]
    assert [t for chunk in masked.output for t in chunk.tokens] == [10, 20]


def test_format_thinking_text_empty_uses_single_newline():
    # Empty reasoning renders the real <think>\n</think> (not the HF template's
    # <think>\n\n</think>) so </think> lands on a clean, supervisable boundary.
    r = qwen35.Qwen35ReasoningRenderer.__new__(qwen35.Qwen35ReasoningRenderer)
    assert r._format_thinking_text("") == "<think>\n</think>\n\n"


def test_assistant_header_suffix_is_assistant_only_and_does_not_validate_payload():
    from types import SimpleNamespace

    r = qwen35.Qwen35ReasoningRenderer.__new__(qwen35.Qwen35ReasoningRenderer)
    ctx = SimpleNamespace(idx=1, last_user_index=0)
    assert r._assistant_header_suffix({"role": "user", "content": "q"}, ctx) == ""
    assert (
        r._assistant_header_suffix({"role": "assistant", "content": "ANSWER"}, ctx)
        == "<think>\n\n</think>\n\n"
    )
    assert (
        r._assistant_header_suffix(
            {"role": "assistant", "content": "<think>\nREASON\n</think>\n\nANSWER"},
            ctx,
        )
        == ""
    )
    assert (
        r._assistant_header_suffix(
            {"role": "assistant", "content": "see <think>x</think> please"},
            ctx,
        )
        == ""
    )
    assert (
        r._assistant_header_suffix(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "see <think>x</think> please"}],
            },
            ctx,
        )
        == ""
    )
    assert (
        r._assistant_header_suffix(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "<think>\nREASON\n</think>\n\nANSWER"}
                ],
            },
            ctx,
        )
        == ""
    )
    assert (
        r._assistant_header_suffix(
            {"role": "assistant", "content": [{"type": "text", "text": "ANSWER"}]},
            ctx,
        )
        == "<think>\n\n</think>\n\n"
    )
    assert (
        r._assistant_header_suffix(
            {"role": "assistant", "content": "</think>ANSWER"},
            ctx,
        )
        == ""
    )
    assert (
        r._assistant_header_suffix(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "</think>ANSWER"},
                    {"type": "image", "image": "x"},
                ],
            },
            ctx,
        )
        == ""
    )
    assert (
        r._assistant_header_suffix(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "ANSWER"},
                    {"type": "image", "image": "x"},
                ],
            },
            ctx,
        )
        == "<think>\n\n</think>\n\n"
    )
    # Malformed thinking payload must not raise here; suffix only looks for a part.
    assert (
        r._assistant_header_suffix(
            {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": 3}],
            },
            ctx,
        )
        == ""
    )


def test_mask_think_span_bails_when_no_close_token():
    tokenizer = _FakeTokenizer()
    header = _chunk([1])
    output = [_chunk([10, 20, 30])]  # no </think>
    rendered = RenderedMessage(header=header, output=output)

    masked = qwen35._mask_think_span(rendered, tokenizer)

    assert list(masked.header.tokens) == [1]
    assert [t for chunk in masked.output for t in chunk.tokens] == [10, 20, 30]


class _QwenCharTokenizer:
    """Reversible tokenizer with Qwen specials kept atomic (for e2e render tests)."""

    _specials = ("<|im_start|>", "<|im_end|>", "</think>", "<think>")

    def encode(self, text, add_special_tokens=False, **_):
        del add_special_tokens
        result: list[int] = []
        pos = 0
        while pos < len(text):
            special = next((s for s in self._specials if text.startswith(s, pos)), None)
            if special is None:
                result.append(ord(text[pos]))
                pos += 1
            else:
                result.append(0x110000 + self._specials.index(special))
                pos += len(special)
        return result

    def decode(self, tokens):
        by_id = {0x110000 + i: s for i, s in enumerate(self._specials)}
        out = []
        for raw in tokens:
            token = int(raw)
            out.append(by_id[token] if token in by_id else chr(token))
        return "".join(out)


def _supervised_ids_weights(train_on_cot: bool, messages):
    from tinker_cookbook.renderers import TrainOnWhat

    renderer = qwen35.Qwen35ReasoningRenderer(
        _QwenCharTokenizer(), supervise_reasoning=train_on_cot
    )
    model_input, weights = renderer.build_supervised_example(
        messages, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES
    )
    return model_input.to_ints(), (
        weights.tolist() if hasattr(weights, "tolist") else list(weights)
    )


def test_no_structure_false_keeps_official_empty_think():
    # Official completed-empty is <think>\n\n</think>\n\n. True injects a
    # serving-aligned <think>\n</think>\n\n; do not force False to match.
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "ANSWER"},
    ]
    ids_true, w_true = _supervised_ids_weights(True, messages)
    ids_false, w_false = _supervised_ids_weights(False, messages)
    tok = _QwenCharTokenizer()
    assert tok.decode(ids_true).count("<think>\n</think>") == 1
    assert tok.decode(ids_false).count("<think>\n\n</think>") == 1
    assert ids_true != ids_false
    assert "</think>" in tok.decode(
        [t for t, w in zip(ids_true, w_true) if w == 1.0]
    )
    assert "</think>" not in tok.decode(
        [t for t, w in zip(ids_false, w_false) if w == 1.0]
    )
    assert "ANSWER" in tok.decode(
        [t for t, w in zip(ids_false, w_false) if w == 1.0]
    )


def test_padded_reasoning_false_does_not_raise():
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "reasoning_content": "\nREASON", "content": "ANSWER"},
    ]
    ids, weights = _supervised_ids_weights(False, messages)
    tok = _QwenCharTokenizer()
    supervised = tok.decode([t for t, w in zip(ids, weights) if w == 1.0])
    assert "REASON" not in supervised
    assert "ANSWER" in supervised


@pytest.mark.parametrize(
    "content",
    [
        "see <think>x</think> please",
        [{"type": "text", "text": "see <think>x</think> please"}],
    ],
)
def test_literal_think_tags_in_answer_are_plain_text(content):
    tok = _QwenCharTokenizer()
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": content},
    ]
    ids_true, w_true = _supervised_ids_weights(True, messages)
    ids_false, w_false = _supervised_ids_weights(False, messages)
    full_true = tok.decode(ids_true)
    full_false = tok.decode(ids_false)
    assert "see <think>x</think> please" in full_true
    assert "see <think>x</think> please" in full_false
    # Visible text already has <think> (string or text part): no second block.
    assert "<think>\n\n</think>" not in full_false
    assert full_false.count("<think>") == 1
    # True still injects the serving-aligned empty block; tags stay in the answer.
    assert full_true.count("<think>\n</think>") == 1
    assert full_true.count("<think>") == 2
    trained_false = tok.decode([t for t, w in zip(ids_false, w_false) if w == 1.0])
    assert trained_false == "see <think>x</think> please<|im_end|>"
    trained_true = tok.decode([t for t, w in zip(ids_true, w_true) if w == 1.0])
    assert trained_true == "</think>\n\nsee <think>x</think> please<|im_end|>"


def test_false_leftover_keeps_answer_leading_newlines():
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "REASON\n</think>\n\nANSWER"},
    ]
    ids_true, w_true = _supervised_ids_weights(True, messages)
    ids_false, w_false = _supervised_ids_weights(False, messages)
    tok = _QwenCharTokenizer()
    assert ids_true == ids_false
    assert "<think>\nREASON\n\n</think>\n\n\n\nANSWER" in tok.decode(ids_true)
    assert tok.decode([t for t, w in zip(ids_false, w_false) if w == 1.0]) == (
        "\n\nANSWER<|im_end|>"
    )
    assert tok.decode([t for t, w in zip(ids_true, w_true) if w == 1.0]) == (
        "REASON\n\n</think>\n\n\n\nANSWER<|im_end|>"
    )


@pytest.mark.parametrize(
    "content",
    [
        ["<think>REASON</think>ANSWER"],
        [{"text": "<think>REASON</think>ANSWER"}],
        [{"type": "text", "text": "<think>REASON</think>ANSWER"}],
    ],
)
def test_complete_block_list_encodings_render_as_plain_answers(content):
    tok = _QwenCharTokenizer()
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": content},
    ]
    ids_true, w_true = _supervised_ids_weights(True, messages)
    ids_false, w_false = _supervised_ids_weights(False, messages)
    visible = "<think>REASON</think>ANSWER"
    assert visible in tok.decode(ids_true)
    assert visible in tok.decode(ids_false)
    assert "<think>\n\n</think>" not in tok.decode(ids_false)
    assert tok.decode([t for t, w in zip(ids_false, w_false) if w == 1.0]) == (
        f"{visible}<|im_end|>"
    )
    assert tok.decode([t for t, w in zip(ids_true, w_true) if w == 1.0]) == (
        f"</think>\n\n{visible}<|im_end|>"
    )


@pytest.mark.parametrize(
    "content",
    [
        "<think>\nREASON\n</think>\n\nANSWER",
        [{"type": "text", "text": "<think>\nREASON\n</think>\n\nANSWER"}],
    ],
)
def test_false_literal_leading_think_block_is_not_double_wrapped(content):
    tok = _QwenCharTokenizer()
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": content},
    ]
    ids, weights = _supervised_ids_weights(False, messages)
    full = tok.decode(ids)
    assert "<think>\n\n</think>\n\n<think>" not in full
    assert full.count("<think>") == 1
    supervised = tok.decode([t for t, w in zip(ids, weights) if w == 1.0])
    assert supervised == "<think>\nREASON\n</think>\n\nANSWER<|im_end|>"


def test_multiple_thinking_parts_false_masks_both():
    tok = _QwenCharTokenizer()
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "RONE"},
                {"type": "thinking", "thinking": "RTWO"},
                {"type": "text", "text": "ANSWER"},
            ],
        },
    ]
    ids, weights = _supervised_ids_weights(False, messages)
    supervised = tok.decode([t for t, w in zip(ids, weights) if w == 1.0])
    assert "ANSWER" in supervised
    assert "RONE" not in supervised and "RTWO" not in supervised


def test_malformed_thinking_part_raises_on_render():
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": 3}]},
    ]
    with pytest.raises(RendererError, match="ThinkingPart"):
        _supervised_ids_weights(True, messages)


def _official_qwen35_tokenizer_dir():
    import os
    from pathlib import Path

    env = os.environ.get("QWEN35_TOKENIZER")
    candidates = []
    if env:
        candidates.append(Path(env))
    repo_root = Path(__file__).resolve().parents[2]
    candidates.append(
        repo_root.parent / "MindForge" / "tests" / "fixtures" / "tokenizers" / "qwen35"
    )
    for path in candidates:
        if (path / "tokenizer.json").is_file():
            return path
    return None


@pytest.fixture(scope="module")
def official_qwen35_tokenizer():
    path = _official_qwen35_tokenizer_dir()
    if path is None:
        pytest.skip("official Qwen3.5 tokenizer fixture not available")
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)


@pytest.mark.parametrize(
    "content",
    [
        "REASON\n</think>\n\nANSWER",
        "</think>\n\nANSWER",
        "REASON</think>\n",
        [{"type": "text", "text": "REASON\n</think>\n\nANSWER"}],
    ],
)
def test_official_leftover_newlines_false_does_not_raise(
    official_qwen35_tokenizer, content
):
    from tinker_cookbook.renderers import TrainOnWhat

    renderer = qwen35.Qwen35ReasoningRenderer(
        official_qwen35_tokenizer, supervise_reasoning=False
    )
    model_input, weights = renderer.build_supervised_example(
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": content},
        ],
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    weight_list = weights.tolist() if hasattr(weights, "tolist") else list(weights)
    trained = official_qwen35_tokenizer.decode(
        [t for t, w in zip(model_input.to_ints(), weight_list) if w == 1.0]
    )
    assert "REASON" not in trained
    if "ANSWER" in str(content):
        assert "ANSWER" in trained


@pytest.mark.parametrize(
    "content",
    [
        ["<think>REASON</think>ANSWER"],
        [{"text": "<think>REASON</think>ANSWER"}],
    ],
)
def test_official_complete_block_list_encodings_render(
    official_qwen35_tokenizer, content
):
    from tinker_cookbook.renderers import TrainOnWhat

    for supervise in (True, False):
        renderer = qwen35.Qwen35ReasoningRenderer(
            official_qwen35_tokenizer, supervise_reasoning=supervise
        )
        model_input, _ = renderer.build_supervised_example(
            [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": content},
            ],
            train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        )
        full = official_qwen35_tokenizer.decode(model_input.to_ints())
        assert "<think>REASON</think>ANSWER" in full
