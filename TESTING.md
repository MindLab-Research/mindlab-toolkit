# Testing Guide

## Quick Test

Verify the MinT SDK is working correctly:

```bash
# Set credentials
export MINT_BASE_URL=https://mintcn.macaron.xin/train
export MINT_API_KEY=your-api-key

# Run example
python examples/basic_training_setup.py
```

Expected output:
```
MinT SDK Example - v0.2.0
============================================================
Endpoint: https://mintcn.macaron.xin/train
API Key: your-api-key...

1. Creating ServiceClient...
   ✓ ServiceClient created

2. Getting available models...
   ✓ Found 1 model(s):
     - Qwen/Qwen3.6-35B-A3B (context: 65536)

3. Creating LoRA TrainingClient...
   Model: Qwen/Qwen3.6-35B-A3B
   LoRA rank: 16
   ✓ TrainingClient created

4. Loading tokenizer...
   ✓ Tokenizer ready: Qwen2Tokenizer
   Example: 'Hello, world!' → 4 tokens

============================================================
✓ Setup complete!
```

## Test Results (2026-08-13)

### Tested Endpoints

| Endpoint | Status | Notes |
|----------|--------|-------|
| `https://mintcn.macaron.xin/train` | ✓ Working | Default endpoint |
| `https://mint.macaron.im/train` | ✓ Working | Global endpoint |
| `https://mint-cn.macaron.xin/` | ⚠️ Auth issues | Key registration required |

### Tested Functionality

- ✓ ServiceClient creation
- ✓ API authentication (both `sk-` and `tml-` prefixes)
- ✓ Server capabilities query
- ✓ Model list retrieval
- ✓ TrainingClient creation
- ✓ Tokenizer loading

### Known Working Models

- `Qwen/Qwen3.6-35B-A3B` (context: 65536)

Use `get_server_capabilities()` to check currently available models on your endpoint.

## Installation

```bash
# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# Verify installation
python -c "import mint; print(f'mint {mint.__version__}')"
```

## API Key Formats

The SDK supports both key formats:
- `sk-...` (MinT standard keys)
- `tml-...` (Tinker/legacy keys)

## Troubleshooting

### AuthenticationError: "must start with the 'tml-' prefix"

This was a bug in v0.2.0 that has been fixed. The auth patch now handles both code paths:
1. `resolve_auth_provider` (JWT-enabled flow)
2. Direct `ApiKeyAuthProvider` instantiation (non-JWT flow)

Update to the latest version to resolve this issue.

### TrainingClient creation timeout

Check:
1. Your API key has training permissions
2. The model name matches an available model from `get_server_capabilities()`
3. Your account has sufficient quota/balance

### Empty supported_models list

The endpoint may require:
- Account activation
- Billing setup
- Different base URL

Contact MinT support for assistance.
