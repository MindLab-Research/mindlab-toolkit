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

Global endpoint, which is also the default:

```bash
export MINT_API_KEY=sk-...
export MINT_BASE_URL=https://mint.macaron.im/train
```

China endpoint:

```bash
export MINT_API_KEY=sk-...
export MINT_BASE_URL=https://mintcn.macaron.xin/train
```

Then create clients through MinT's Tinker-compatible surface:

```python
import mint

service_client = mint.ServiceClient()
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
- MinT keys start with `sk-`
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

## Documentation

Read the MinT documentation at [mint-doc.macaron.im](https://mint-doc.macaron.im).

## License

MIT

---

A [Mind Lab](https://macaron.im/mindlab) Contribution - A Lab for Experiential Intelligence.
