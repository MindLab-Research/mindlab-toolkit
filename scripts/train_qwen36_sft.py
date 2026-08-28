#!/usr/bin/env python3
"""Run one-from-scratch Qwen3.6 SFT job through Tinker or MinT.

The ``tinker`` backend is the reference path.  The ``mint`` backend exercises
MinT's compatibility layer with the exact same model, renderer, and datums.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
from pathlib import Path
from typing import Any, Iterator

MODEL = "Qwen/Qwen3.6-35B-A3B"
BASE_URL = "https://mintcn.macaron.xin/train"
DEFAULT_DATA = r"C:\Users\trots\Downloads\sft__merged__balanced_v3__multiturn(1).jsonl"
WINDOWS_MOUNT = Path("/mnt")


def resolve_data_path(value: str) -> Path:
    """Resolve the requested Windows path through WSL's /mnt/c mapping."""
    path = Path(value)
    if path.is_file():
        return path
    if len(value) >= 3 and value[1:3] == ":\\":
        mapped = WINDOWS_MOUNT / value[0].lower() / Path(value[3:].replace("\\", "/"))
        if mapped.is_file():
            return mapped
    return path


def conversations(path: Path) -> Iterator[list[dict[str, Any]]]:
    """Read Tinker Cookbook's strict ``{"messages": [...]}`` JSONL shape."""
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            messages = (row.get("messages") or row.get("rollout")) if isinstance(row, dict) else None
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{path}:{line_number}: expected non-empty messages list")
            for message in messages:
                if not isinstance(message, dict) or not isinstance(message.get("role"), str):
                    raise ValueError(f"{path}:{line_number}: every message needs a role")
                if "content" not in message:
                    raise ValueError(f"{path}:{line_number}: every message needs content")
            yield messages


def assistant_prefixes(messages: list[dict[str, Any]]) -> Iterator[list[dict[str, Any]]]:
    for index, message in enumerate(messages):
        if message["role"] == "assistant":
            yield messages[: index + 1]


def make_datums(
    path: Path,
    tokenizer,
    max_length: int,
    limit: int | None = None,
    objective: str = "sft",
):
    from tinker_cookbook.renderers import TrainOnWhat, get_renderer
    from tinker_cookbook.supervised.data import conversation_to_datum

    renderer = get_renderer("qwen3", tokenizer, model_name=MODEL)
    if objective == "pretrain":
        examples = conversations(path)
        train_on_what = TrainOnWhat.ALL_TOKENS
    else:
        examples = (
            prefix
            for messages in conversations(path)
            for prefix in assistant_prefixes(messages)
        )
        train_on_what = TrainOnWhat.LAST_ASSISTANT_MESSAGE
    return [
        conversation_to_datum(
            messages,
            renderer,
            max_length=max_length,
            train_on_what=train_on_what,
            reduction="mean",
        )
        for messages in itertools.islice(examples, limit)
    ]


def run(args: argparse.Namespace) -> None:
    api_key = os.environ.get("MINT_API_KEY") or os.environ.get("TINKER_API_KEY")
    if not api_key:
        raise RuntimeError("set MINT_API_KEY (or TINKER_API_KEY) before training")

    base_url = os.environ.get("MINT_BASE_URL") or os.environ.get("TINKER_BASE_URL") or BASE_URL
    os.environ.setdefault("TINKER_API_KEY", api_key)
    os.environ.setdefault("TINKER_BASE_URL", base_url)
    if args.backend == "mint":
        os.environ.setdefault("MINT_API_KEY", api_key)
        os.environ.setdefault("MINT_BASE_URL", base_url)
        import mint as sdk
    else:
        import mint  # Apply MinT's sk-key compatibility patch to Tinker.
        import tinker as sdk

    data_path = resolve_data_path(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(f"training data not found: {data_path}")

    service = sdk.ServiceClient(api_key=api_key, base_url=base_url)
    capabilities = service.get_server_capabilities()
    supported = {model.model_name for model in capabilities.supported_models}
    if MODEL not in supported:
        raise RuntimeError(f"{MODEL} is not available; server offers {sorted(supported)}")

    client = asyncio.run(
        asyncio.wait_for(
            service.create_lora_training_client_async(
                base_model=MODEL,
                rank=args.rank,
                seed=args.seed,
                train_mlp=True,
                train_attn=True,
                train_unembed=True,
            ),
            timeout=args.timeout,
        )
    )
    datum_limit = args.max_steps * args.batch_size if args.max_steps else None
    datums = make_datums(
        data_path,
        client.get_tokenizer(),
        args.max_length,
        datum_limit,
        args.objective,
    )
    if not datums:
        raise ValueError("training data contains no records")

    completed_steps = 0
    for epoch in range(args.epochs):
        for step, start in enumerate(range(0, len(datums), args.batch_size), 1):
            batch = datums[start : start + args.batch_size]
            fb = client.forward_backward(batch, "cross_entropy").result(timeout=args.timeout)
            optim = client.optim_step(sdk.AdamParams(learning_rate=args.learning_rate)).result(timeout=args.timeout)
            completed_steps += 1
            metrics = fb.metrics
            print(
                f"{args.backend} epoch={epoch + 1} step={step}/{(len(datums) + args.batch_size - 1) // args.batch_size} "
                f"loss_mean={metrics.get('loss:mean', float('nan')):.6f} "
                f"loss_sum={metrics.get('loss:sum', float('nan')):.6f} "
                f"tokens={metrics.get('num_tokens:sum', 0)} "
                f"grad_norm={(optim.metrics or {}).get('grad_norm', float('nan')):.6f}",
                flush=True,
            )
            if args.max_steps and completed_steps >= args.max_steps:
                break
        if args.max_steps and completed_steps >= args.max_steps:
            break

    if args.save == "state":
        saved = client.save_state(args.output).result(timeout=args.timeout)
    elif args.save == "sampler":
        saved = client.save_weights_for_sampler(args.output).result(timeout=args.timeout)
    else:
        return
    print(f"saved: {getattr(saved, 'path', saved)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--backend", choices=("tinker", "mint"), default="tinker")
    parser.add_argument("--objective", choices=("sft", "pretrain"), default="sft")
    parser.add_argument("--max-length", type=int, default=65536)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--output", default="qwen36-sft-from-scratch")
    parser.add_argument("--save", choices=("state", "sampler", "none"), default="none")
    parser.add_argument("--timeout", type=float, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
