#!/usr/bin/env python3
"""
Single inference test for quick model validation.

This script provides a simple way to test the fine-tuned MLX model by:
1. Loading the model with its fine-tuned adapter
2. Selecting a random test example
3. Generating a completion and comparing it to the expected output

Use this for quick smoke testing after training or when debugging model behavior.

Dependencies:
- mlx.core: MLX framework core
- mlx_lm: MLX language model utilities (load, generate)
- json: Parse JSONL test data
- random: Random test example selection

Usage:
    python test_single_inference.py

    The script uses hardcoded paths:
    - Model: models/mistral-7b-mlx
    - Adapter: adapters/contact-center-mlx-small
    - Test data: data/mlx_small/test.jsonl

Related Files:
- validate_model.py: Comprehensive validation with perplexity metrics
- convert_to_mlx_format.py: Prepares test data in MLX format
"""

import json
import mlx.core as mx
from mlx_lm import load, generate
import random

def test_single_inference():
    """
    Run a single inference test showing input, output, and expected result.

    This function:
    1. Loads the fine-tuned model
    2. Randomly selects one test example
    3. Generates a completion
    4. Displays prompt, generated output, and expected output
    5. Shows basic statistics (character counts)

    Note:
        Uses hardcoded paths - modify source code to change model/data paths.
    """
    print("=" * 80)
    print("SINGLE INFERENCE TEST")
    print("=" * 80)

    # Load base model with fine-tuned LoRA adapter
    # The adapter contains task-specific weights learned during training
    print("\nLoading model...")
    model, tokenizer = load(
        "models/mistral-7b-mlx",  # Base model directory
        adapter_path="adapters/contact-center-mlx-small"  # Fine-tuned adapter
    )
    print("Model loaded successfully!\n")

    # Load test data (JSONL format, one JSON object per line)
    print("Loading test data...")
    with open("data/mlx_small/test.jsonl", 'r') as f:
        test_examples = [json.loads(line) for line in f]

    # Randomly select one example for testing
    example = random.choice(test_examples)
    full_text = example['text']

    # Parse the example into prompt and expected completion
    # MLX format uses <|assistant|> as the delimiter
    if '<|assistant|>' in full_text:
        parts = full_text.split('<|assistant|>')
        prompt = parts[0] + '<|assistant|>'  # Include delimiter in prompt
        expected = parts[1].strip() if len(parts) > 1 else ""
    else:
        # Fallback if format doesn't match expected pattern
        prompt = full_text[:500]
        expected = "N/A"

    # Display the input prompt
    print("=" * 80)
    print("INPUT PROMPT")
    print("=" * 80)
    print(prompt)
    print()

    print("=" * 80)
    print("GENERATING RESPONSE...")
    print("=" * 80)

    # Generate completion using the fine-tuned model
    # max_tokens limits output length to prevent runaway generation
    # verbose=False suppresses token-by-token output
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=500,  # Maximum tokens to generate
        verbose=False    # Suppress generation progress
    )

    # Extract only the new generated text (remove echoed prompt)
    # The generate() function returns prompt + completion
    if prompt in response:
        generated = response.replace(prompt, '').strip()
    else:
        generated = response

    # Display the model's generated output
    print("\n" + "=" * 80)
    print("MODEL OUTPUT")
    print("=" * 80)
    print(generated)
    print()

    # Display the expected output from test data for comparison
    print("=" * 80)
    print("EXPECTED OUTPUT (from test data)")
    print("=" * 80)
    # Truncate very long expected outputs for readability
    print(expected[:1000] + ('...' if len(expected) > 1000 else ''))
    print()

    # Show basic statistics about the inference
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print(f"Prompt length: {len(prompt)} characters")
    print(f"Generated length: {len(generated)} characters")
    print(f"Expected length: {len(expected)} characters")
    print("=" * 80)

if __name__ == "__main__":
    test_single_inference()
