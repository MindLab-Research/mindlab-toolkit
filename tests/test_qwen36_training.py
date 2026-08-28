import importlib.util
import json

import pytest


spec = importlib.util.spec_from_file_location("train_qwen36_sft", "scripts/train_qwen36_sft.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_conversations_preserves_multiturn_records(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        json.dumps({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}) + "\n",
        encoding="utf-8",
    )

    assert list(module.conversations(path)) == [
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    ]


def test_conversations_rejects_missing_messages(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"text": "not chat"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="messages"):
        list(module.conversations(path))


def test_conversations_accepts_rollout(tmp_path):
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    path = tmp_path / "rollout.jsonl"
    path.write_text(json.dumps({"rollout": messages}) + "\n", encoding="utf-8")

    assert list(module.conversations(path)) == [messages]


def test_resolve_data_path_maps_windows_drive(monkeypatch, tmp_path):
    mapped = tmp_path / "c" / "Users" / "trots" / "Downloads" / "data.jsonl"
    mapped.parent.mkdir(parents=True)
    mapped.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(module, "WINDOWS_MOUNT", tmp_path)
    assert module.resolve_data_path(r"C:\Users\trots\Downloads\data.jsonl") == mapped


def test_assistant_prefixes_expands_multiturn_conversation():
    messages = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "first"},
        {"role": "user", "content": "two"},
        {"role": "assistant", "content": "second"},
    ]

    assert list(module.assistant_prefixes(messages)) == [messages[:2], messages]
