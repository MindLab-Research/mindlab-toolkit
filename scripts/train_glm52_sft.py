#!/usr/bin/env python3
"""Run reproducible GLM-5.2 MinT SFT smoke tests."""

import argparse
import asyncio
import json
import os
from pathlib import Path

import mint
import mint.renderers  # Registers MindLab/glm52.
from tinker_cookbook.renderers import TrainOnWhat, get_renderer
from tinker_cookbook.supervised.data import conversation_to_datum

MODEL = "zai-org/GLM-5.2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    key, url = os.getenv("MINT_API_KEY"), os.getenv("MINT_BASE_URL")
    if not key or not url:
        raise RuntimeError("set MINT_API_KEY and MINT_BASE_URL")
    rows = [json.loads(line)["messages"] for line in args.data.open(encoding="utf-8") if line.strip()]
    service = mint.ServiceClient(api_key=key, base_url=url)
    for run in range(1, args.runs + 1):
        client = asyncio.run(service.create_lora_training_client_async(
            base_model=MODEL, rank=args.rank, seed=args.seed,
            train_mlp=True, train_attn=True, train_unembed=True,
        ))
        print(f"run={run} model_id={client.model_id}", flush=True)
        renderer = get_renderer("MindLab/glm52", client.get_tokenizer())
        datums = [conversation_to_datum(
            messages, renderer, max_length=args.max_length,
            train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES, reduction="mean",
        ) for messages in rows[:args.max_steps]]
        adam = mint.AdamParams(learning_rate=1e-4, beta1=.9, beta2=.999, eps=1e-8)
        for step, datum in enumerate(datums, 1):
            fb = client.forward_backward([datum], "cross_entropy").result(timeout=300)
            optim = client.optim_step(adam).result(timeout=300)
            print(f"run={run} step={step} loss={fb.metrics.get('loss:mean')} "
                  f"grad_norm={(optim.metrics or {}).get('grad_norm')}", flush=True)


if __name__ == "__main__":
    main()
