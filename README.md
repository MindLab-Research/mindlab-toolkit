# Mind Lab Toolkit (MinT)

The Open Infrastructure for Experiential Intelligence.

MinT is the reinforcement learning infrastructure for agents and models to learn from real experience. It focuses on the engineering and algorithmic realization of RL across multiple models and tasks, with emphasis on making LoRA RL simple, stable, and efficient.

## Installation

```bash
cd mindlab-toolkit
pip install -e .
```

## Usage

```python
import mint

# Set API key via environment variable MINT_API_KEY.
# You can keep both MINT_* and TINKER_* variables in the same .env.
# Importing mint makes MINT_* take precedence for this process; set MINT_BASE_URL
# if you want a non-default endpoint.
# Default base URL: https://mint.macaron.xin
# Mainland China endpoint override: https://mint-cn.macaron.xin/

service_client = mint.ServiceClient()
```

Top-level `import mint` is the Tinker-compatible surface. All public Tinker APIs are available directly from `mint` and mirrored in `mint.tinker`.

## MinT Extension Namespace (`mintx`)

MinT-only APIs live under `mint.mint`. The intended usage is:

```python
import mint
import mint.mint as mintx

service_client = mint.ServiceClient()
openpi_client = mintx.create_openpi_training_client(service_client)
```

Use this namespace for MinT-specific extensions that should not appear in the default top-level `mint` surface. OpenPI / VLA helpers are the first concrete example of that rule.

The current OpenPI helper is intentionally narrow: it is pinned to `openpi/pi0-fast-libero-low-mem-finetune` with LoRA rank `16`.

## Documentation

Read the MinT documentation at [mint-doc.macaron.im](https://mint-doc.macaron.im).

## License

MIT

---

A [Mind Lab](https://macaron.im/mindlab) Contribution - A Lab for Experiential Intelligence.
