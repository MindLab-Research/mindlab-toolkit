<p align="center">
  <img src="docs/assets/mint-icon.jpg" alt="MinT" width="120" height="120">
</p>

# Mind Lab Toolkit (MinT)

The Open Infrastructure for Experiential Intelligence.

MinT is the reinforcement learning infrastructure for agents and models to learn from real experience. It focuses on the engineering and algorithmic realization of RL across multiple models and tasks, with emphasis on making LoRA RL simple, stable, and efficient.

Visit the [MinT website](https://macaron.im/mindlab/mint).

## Installation

From a source checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

When `mindlab-toolkit==0.2.0` is available on your configured package index, the equivalent package installation is:

```bash
python -m pip install 'mindlab-toolkit==0.2.0'
```

MinT pins and installs its validated SDK dependency automatically:

```text
mindlab-toolkit==0.2.0
tinker==0.22.0
```

Do not use `pip install --no-deps`. That option bypasses dependency installation. `import mint` checks the installed Tinker distribution before importing Tinker internals and reports how to install `tinker==0.22.0` when it is missing or incompatible.

The Python package version (`0.2.0`) and this repository's release tags (for example `v2.6.5`) are currently separate version identifiers.

## Usage

China endpoint, which is also the default:

```bash
export MINT_API_KEY=tml-...
export MINT_BASE_URL=https://mintcn.macaron.xin/train
```

Global endpoint:

```bash
export MINT_API_KEY=sk-...
export MINT_BASE_URL=https://mint.macaron.im/train
```

Then create clients through MinT's Tinker-compatible surface:

```python
import mint

service_client = mint.ServiceClient()
```

### Qwen3.6 SFT from scratch

Install the optional Tinker Cookbook renderer, then run the same training
loop against either SDK surface. `--backend tinker` is the reference path;
`--backend mint` uses MinT with identical datums and optimizer settings.

```bash
python -m pip install -e '.[glm52-renderer]'
export MINT_API_KEY=tml-...
python scripts/train_qwen36_sft.py \
  --data 'C:\Users\trots\Downloads\sft__merged__balanced_v3__multiturn(1).jsonl' \
  --backend mint
```

For a GLM-5.2 reproducibility check, first convert inline Qwen thinking blocks,
then run two independent 10-step MinT clients:

```bash
python scripts/convert_qwen_rollout_to_glm52.py input.jsonl glm52.jsonl
export MINT_API_KEY=tml-...
export MINT_BASE_URL=http://your-mint-endpoint
python scripts/train_glm52_sft.py --data glm52.jsonl --runs 2 --max-steps 10
```

A non-empty `MINT_API_KEY` takes precedence over `TINKER_API_KEY` for the process. A non-empty `MINT_BASE_URL` similarly takes precedence over `TINKER_BASE_URL`; an explicit client constructor `base_url` remains the final override. Both supported deployment URLs retain their `/train` prefix.

Top-level `import mint` exposes the public Tinker 0.22.0 API directly and mirrors it in `mint.tinker`.

## Lowest-Friction Tinker Migration

If your existing code starts with `import tinker`, the smallest working MinT migration is:

```python
import mint as tinker
```

Then switch your credentials and endpoint to MinT.

Why this matters:

- raw upstream `import tinker` keeps Tinker's original behavior: `tml-` and `eyJ...` are accepted, while `sk-` is rejected
- MinT keys may use `sk-`; the supplied `tml-` key also follows Tinker's native path
- `import mint` validates Tinker 0.22.0 and patches its confirmed standard client construction paths for MinT keys
- Tinker's original `ApiKeyAuthProvider` class remains unchanged and still rejects `sk-` when called directly

If you must keep the exact `import tinker` statement, import `mint` earlier in the same process before constructing Tinker clients. Importing only `tinker` does not load MinT compatibility.

## MinT Extension Namespace (`mintx`)

MinT-only APIs live under `mint.mint`. The intended usage is:

```python
import mint
import mint.mint as mintx

training_client = ...  # a mint/tinker TrainingClient
mintx.forward_backward_reverse_kl(
    training_client,
    reference_model_path="mint://teacher-step-0010",
    data=[...],
)
```

Use this namespace for MinT-specific extensions that should not appear in the default top-level `mint` surface. The current extensions are the MinT-only endpoints `forward_backward_reverse_kl` and `interpolate_checkpoints`.

## GLM-5.2 Tinker Renderer

Install the optional renderer dependencies:

```bash
python -m pip install 'mindlab-toolkit[glm52-renderer]'
```

The extra selects Tinker Cookbook 0.4.0/0.4.1, whose Renderer API is compatible with this release's validated `tinker==0.22.0` pin.

Importing `mint.renderers` registers three namespaced renderers with Tinker Cookbook:

- `MindLab/glm52`: thinking enabled, `Reasoning Effort: Max`
- `MindLab/glm52_high_reasoning`: thinking enabled, `Reasoning Effort: High`
- `MindLab/glm52_disable_thinking`: thinking disabled with `<think></think>` pre-filled

```python
from mint.renderers import GLM52_RENDERER
from tinker_cookbook.renderers import Message, get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

model_name = "zai-org/GLM-5.2"
tokenizer = get_tokenizer(model_name)
renderer = get_renderer(
    GLM52_RENDERER,
    tokenizer,
    model_name=model_name,
)

prompt = renderer.build_generation_prompt(
    [Message(role="user", content="Who are you?")]
)
```

Use Tinker's structured `ThinkingPart` for reasoning-bearing assistant messages:

```python
assistant = Message(
    role="assistant",
    content=[
        {"type": "thinking", "thinking": "I should answer briefly."},
        {"type": "text", "text": "I am GLM-5.2."},
    ],
)
```

Existing records from the previous GLM-5.2 SFT pipeline can use the legacy OpenAI-style shape instead, without preprocessing:

```python
legacy_assistant = {
    "role": "assistant",
    "reasoning_content": "I should answer briefly.",
    "content": "I am GLM-5.2.",
}
```

Use either a `ThinkingPart` in `content` or the legacy string `reasoning_content`, never both on the same message. New code should prefer `ThinkingPart`. The disable-thinking renderer rejects either representation because its inference prompt explicitly disables reasoning.

Tool definitions are added with `renderer.create_conversation_prefix_with_tools(...)`. The renderer emits and parses GLM-5.2's `<tool_call>...<arg_key>...<arg_value>...</tool_call>` format and uses `<|observation|>` / `<|user|>` as sampling stops. Its SFT builder supervises those assistant-produced boundaries while keeping role headers and the pre-filled `<think>` scaffold at zero weight.

Install the test extra before running the repository's default test suite:

```bash
python -m pip install -e '.[test]'
python -m pytest tests/
```

The default suite excludes the external GLM-5.2 SFT parity tests. To run them, clone `MindLab-Research/agent-model-training-mono` next to this repository (or set `GLM52_SFT_REFERENCE_DIR` to its `glm52_sft` directory), then override the default marker selection:

```bash
python -m pytest -o addopts='' -m integration tests/renderers/test_glm52_sft_parity.py
```

### Strict GLM-5.2 dataset validation

Validate every JSONL record through both the strict on-disk contract and the real Tinker renderer before training. `--max-seq-len` is required and must equal the training setting; the validator fails closed when it is unknown.

```bash
python -m pip install -e '.[glm52-renderer]'
mint-validate-glm52-sft \
    --data training.glm52_sft.jsonl \
    --max-seq-len 98304 \
    --report validation-report.json
```

The command rejects malformed JSONL, duplicate keys, unknown fields, protocol-marker injection, ambiguous reasoning fields, broken tool pairing, undeclared tools, renderer failures, invalid masks, and samples with no supervised tokens. It reports truncation, duplicate records, unusual supervised-token ratios, and sequences near the configured limit. Exit code `0` means the dataset is trainable, `1` means fatal data violations, and `2` means an environment/configuration error.

Repository agents can use the checked-in [`glm52-sft-validator` skill](.agents/skills/glm52-sft-validator/SKILL.md) for the fail-closed validation workflow and report interpretation.

## Documentation

Read the MinT documentation at [mint-doc.macaron.im](https://mint-doc.macaron.im).

## License

MIT

---

A [Mind Lab](https://macaron.im/mindlab) Contribution - A Lab for Experiential Intelligence.
