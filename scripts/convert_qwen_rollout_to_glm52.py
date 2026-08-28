#!/usr/bin/env python3
"""Convert inline Qwen <think> blocks to GLM-5.2 structured content."""

import argparse
import json
import re
from pathlib import Path

THINK_BLOCK = re.compile(r"\s*<think>(.*?)</think>\s*(.*)\s*", re.S)


def convert(source: Path, destination: Path) -> tuple[int, int]:
    rows = converted = 0
    with source.open(encoding="utf-8") as inp, destination.open("w", encoding="utf-8") as out:
        for line_number, line in enumerate(inp, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages") or row.get("rollout")
            if not isinstance(messages, list):
                raise ValueError(f"{source}:{line_number}: expected messages or rollout list")
            for message in messages:
                content = message.get("content")
                if message.get("role") != "assistant" or not isinstance(content, str):
                    continue
                match = THINK_BLOCK.fullmatch(content)
                if match:
                    message["content"] = [
                        {"type": "thinking", "thinking": match.group(1)},
                        {"type": "text", "text": match.group(2)},
                    ]
                    converted += 1
            row["messages"] = messages
            row.pop("rollout", None)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows += 1
    return rows, converted


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    rows, converted = convert(args.source, args.destination)
    print(f"rows={rows} converted_assistant_messages={converted}")
