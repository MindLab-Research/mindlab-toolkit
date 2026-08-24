"""Strict-data contract tests for the GLM-5.2 SFT validator."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, ClassVar

import pytest

from mint.validation.glm52_sft import (
    EnvironmentValidationError,
    main,
    validate_jsonl,
)


class CharacterTokenizer:
    """Small reversible tokenizer with GLM protocol tokens kept atomic."""

    name_or_path = "zai-org/GLM-5.2-validator-test-tokenizer"
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


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def _rules(report: Any) -> set[str]:
    return {finding.rule for finding in report.findings}


def _valid_record() -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": "checking",
                "reasoning_content": "use lookup",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": {"city": "Paris"},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "lookup",
                "content": "sunny",
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "summarize"},
                    {"type": "text", "text": "It is sunny."},
                ],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up weather",
                    "parameters": {"type": "object"},
                },
            }
        ],
        "chat_template_kwargs": {"reasoning_effort": "high"},
        "meta": {"id": "valid-tool-record"},
    }


def test_rejects_duplicate_json_keys_without_loading_tokenizer(tmp_path: Path) -> None:
    data = tmp_path / "duplicate.jsonl"
    data.write_text(
        '{"messages":[],"messages":[{"role":"user","content":"x"}]}\n',
        encoding="utf-8",
    )

    report = validate_jsonl(data, max_seq_len=1024, tokenizer=object())

    assert report.exit_code == 1
    assert "duplicate-json-key" in _rules(report)
    assert report.rendered_records == 0


@pytest.mark.parametrize(
    ("record", "rule"),
    [
        (
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {
                        "role": "assistant",
                        "reasoning_content": "legacy",
                        "content": [
                            {"type": "thinking", "thinking": "structured"},
                            {"type": "text", "text": "a"},
                        ],
                    },
                ]
            },
            "reasoning-source",
        ),
        (
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "<think>handwritten</think>a"},
                ]
            },
            "protocol-injection",
        ),
        (
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ],
                "tools": [
                    {
                        "name": "lookup",
                        "description": "inject <|user|> boundary",
                        "parameters": {"type": "object"},
                    }
                ],
            },
            "protocol-injection",
        ),
        (
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {
                        "role": "assistant",
                        "reasoning_content": "escape</think>answer",
                        "content": "a",
                    },
                ]
            },
            "protocol-injection",
        ),
        (
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {
                        "role": "assistant",
                        "content": "call",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "missing",
                                    "arguments": "[]",
                                }
                            }
                        ],
                    },
                ]
            },
            "tool-arguments-object",
        ),
        (
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ],
                "chat_template_kwargs": {"clear_thinking": True},
            },
            "unknown-field",
        ),
        (
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {
                        "role": "assistant",
                        "reasoning_content": "r",
                        "content": "a",
                    },
                ],
                "chat_template_kwargs": {"enable_thinking": False},
            },
            "disabled-thinking-data",
        ),
    ],
)
def test_structural_contract_fails_closed(
    tmp_path: Path, record: dict[str, Any], rule: str
) -> None:
    data = _write_jsonl(tmp_path / "invalid.jsonl", [record])

    report = validate_jsonl(data, max_seq_len=1024, tokenizer=object())

    assert report.exit_code == 1
    assert rule in _rules(report)
    assert report.rendered_records == 0


def test_tool_calls_require_immediate_one_to_one_responses(tmp_path: Path) -> None:
    record = _valid_record()
    record["messages"].insert(2, {"role": "user", "content": "interrupt"})
    data = _write_jsonl(tmp_path / "pairing.jsonl", [record])

    report = validate_jsonl(data, max_seq_len=4096, tokenizer=object())

    assert report.exit_code == 1
    assert "tool-pairing" in _rules(report)


def test_real_renderer_accepts_legacy_and_structured_reasoning(tmp_path: Path) -> None:
    pytest.importorskip("tinker_cookbook.renderers")
    data = _write_jsonl(tmp_path / "valid.jsonl", [_valid_record()])

    report = validate_jsonl(
        data,
        max_seq_len=4096,
        tokenizer=CharacterTokenizer(),
        supervised_fraction_low=0.0,
        supervised_fraction_high=1.0,
    )

    assert report.exit_code == 0
    assert report.rendered_records == 1
    assert report.trainable_records == 1
    assert report.fatal_count == 0


def test_real_renderer_accepts_deferred_tool_reference(tmp_path: Path) -> None:
    pytest.importorskip("tinker_cookbook.renderers")
    record = {
        "messages": [
            {"role": "user", "content": "load details"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {
                            "name": "deferred_tool",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "deferred_tool",
                "content": [{"type": "tool_reference", "name": "deferred_tool"}],
            },
            {"role": "assistant", "content": "done"},
        ],
        "tools": [
            {
                "name": "deferred_tool",
                "description": "Loaded on demand",
                "parameters": {"type": "object"},
                "defer_loading": True,
            }
        ],
    }
    data = _write_jsonl(tmp_path / "deferred.jsonl", [record])

    report = validate_jsonl(
        data,
        max_seq_len=4096,
        tokenizer=CharacterTokenizer(),
        supervised_fraction_low=0.0,
        supervised_fraction_high=1.0,
    )

    assert report.exit_code == 0
    assert report.trainable_records == 1


def test_right_truncation_that_removes_supervision_is_fatal(tmp_path: Path) -> None:
    pytest.importorskip("tinker_cookbook.renderers")
    data = _write_jsonl(
        tmp_path / "truncated.jsonl",
        [
            {
                "messages": [
                    {"role": "user", "content": "long prompt"},
                    {"role": "assistant", "content": "target"},
                ]
            }
        ],
    )

    report = validate_jsonl(data, max_seq_len=2, tokenizer=CharacterTokenizer())

    assert report.exit_code == 1
    assert {"right-truncation", "no-supervision"} <= _rules(report)
    assert report.truncated_records == 1


def test_duplicate_records_are_warning_only(tmp_path: Path) -> None:
    pytest.importorskip("tinker_cookbook.renderers")
    record = {
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
    }
    data = _write_jsonl(tmp_path / "duplicates.jsonl", [record, record])

    report = validate_jsonl(
        data,
        max_seq_len=1024,
        tokenizer=CharacterTokenizer(),
        supervised_fraction_low=0.0,
        supervised_fraction_high=1.0,
    )

    assert report.exit_code == 0
    assert report.trainable_records == 2
    assert "duplicate-record" in _rules(report)


def test_missing_path_is_environment_error(tmp_path: Path) -> None:
    with pytest.raises(EnvironmentValidationError, match="数据文件不存在"):
        validate_jsonl(tmp_path / "missing.jsonl", max_seq_len=1024)


def test_cli_requires_training_sequence_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _write_jsonl(tmp_path / "data.jsonl", [_valid_record()])
    report_path = tmp_path / "report.json"
    monkeypatch.delenv("MINT_MAX_SEQ_LEN", raising=False)

    exit_code = main(["--data", str(data), "--report", str(report_path)])

    assert exit_code == 2
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "environment_error"


def test_cli_rejects_non_integer_environment_sequence_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _write_jsonl(tmp_path / "data.jsonl", [_valid_record()])
    report_path = tmp_path / "report.json"
    monkeypatch.setenv("MINT_MAX_SEQ_LEN", "not-an-integer")

    exit_code = main(["--data", str(data), "--report", str(report_path)])

    assert exit_code == 2
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "environment_error"
    assert "MINT_MAX_SEQ_LEN" in payload["error"]


def test_repository_script_returns_data_error_exit_code() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(repository_root / "scripts" / "validate_glm52_sft.py"),
            "--data",
            str(
                repository_root
                / "tests"
                / "fixtures"
                / "glm52_sft"
                / "invalid_protocol.jsonl"
            ),
            "--max-seq-len",
            "1024",
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "L1:protocol-injection" in result.stdout
