#!/usr/bin/env python3
"""
Example: Create a MinT training client and prepare for fine-tuning.

This example demonstrates:
1. Connecting to MinT with API credentials
2. Checking available models
3. Creating a LoRA training client
4. Loading the tokenizer

Prerequisites:
    export MINT_BASE_URL=https://mintcn.macaron.xin/train
    export MINT_API_KEY=your-api-key-here
"""

import os
import mint


def main():
    # Check environment variables
    base_url = os.environ.get('MINT_BASE_URL')
    api_key = os.environ.get('MINT_API_KEY')

    if not base_url or not api_key:
        print("Error: MINT_BASE_URL and MINT_API_KEY must be set")
        print("\nExample:")
        print("  export MINT_BASE_URL=https://mintcn.macaron.xin/train")
        print("  export MINT_API_KEY=your-key-here")
        return 1

    print(f"MinT SDK Example - v{mint.__version__}")
    print("=" * 60)
    print(f"Endpoint: {base_url}")
    print(f"API Key: {api_key[:20]}...")
    print()

    # Step 1: Create ServiceClient
    print("1. Creating ServiceClient...")
    service_client = mint.ServiceClient()
    print("   ✓ ServiceClient created")

    # Step 2: Get available models
    print("\n2. Getting available models...")
    caps = service_client.get_server_capabilities()
    models = caps.supported_models

    if not models:
        print("   ✗ No models available")
        return 1

    print(f"   ✓ Found {len(models)} model(s):")
    for model in models:
        print(f"     - {model.model_name} (context: {model.max_context_length})")

    model_name = "Qwen/Qwen3.6-35B-A3B"
    if model_name not in {model.model_name for model in models}:
        print(f"   ✗ Required model is unavailable: {model_name}")
        return 1

    # Step 3: Create TrainingClient
    print(f"\n3. Creating LoRA TrainingClient...")
    print(f"   Model: {model_name}")
    print(f"   LoRA rank: 16")

    training_client = service_client.create_lora_training_client(
        base_model=model_name,
        rank=16,
        train_mlp=True,
        train_attn=True,
        train_unembed=True,
    )
    print("   ✓ TrainingClient created")

    # Step 4: Get tokenizer
    print("\n4. Loading tokenizer...")
    tokenizer = training_client.get_tokenizer()
    print(f"   ✓ Tokenizer ready: {type(tokenizer).__name__}")

    # Test tokenization
    test_text = "Hello, world!"
    tokens = tokenizer.encode(test_text)
    print(f"   Example: '{test_text}' → {len(tokens)} tokens")

    print("\n" + "=" * 60)
    print("✓ Setup complete!")
    print("\nYour training client is ready. Next steps:")
    print("  1. Prepare your training data as mint.types.Datum objects")
    print("  2. Call training_client.forward_backward(data, loss_fn='cross_entropy')")
    print("  3. Call training_client.optim_step(mint.types.AdamParams(learning_rate=1e-4))")
    print("  4. Save checkpoints with training_client.save_weights_for_sampler()")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
