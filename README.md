# Mind Lab Toolkit (MinT)

The Open Infrastructure for Experiential Intelligence.

MinT is the reinforcement learning infrastructure for agents and models to learn from real experience. It focuses on the engineering and algorithmic realization of RL across multiple models and tasks, with emphasis on making LoRA RL simple, stable, and efficient.

## Installation

```bash
pip install mindlab-toolkit
```

## Usage

```python
import mint

# Set API key via environment variable MINT_API_KEY.
# You can keep both MINT_* and TINKER_* variables in the same .env.
# Importing mint makes MINT_* take precedence for this process; set MINT_BASE_URL
# if you want a non-default endpoint.
# Default base URL: https://mint.macaron.im

client = mint.TrainingClient()
```

All tinker APIs are available directly from mint.

## License

MIT

---

A [Mind Lab](https://macaron.im/mindlab) Contribution - A Lab for Experiential Intelligence.
