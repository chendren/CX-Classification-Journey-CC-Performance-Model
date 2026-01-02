#!/usr/bin/env python3
"""
Validate the fine-tuned MLX model on the test set.

This script performs comprehensive validation of a fine-tuned MLX model by:
1. Calculating perplexity metrics on a sample of test data
2. Generating sample outputs to assess qualitative performance
3. Comparing generated responses against expected outputs

Dependencies:
- mlx.core, mlx.nn: Apple's MLX framework for ML on Apple Silicon
- mlx_lm: MLX language model utilities (load, generate)
- numpy: Statistical calculations
- tqdm: Progress bars

Usage:
    python validate_model.py --model models/mistral-7b-mlx \
                            --adapter adapters/contact-center-mlx-small/PRODUCTION_MODEL.safetensors \
                            --test-data data/mlx_small/test.jsonl \
                            --num-samples 100 \
                            --num-generations 5

Related Files:
- test_single_inference.py: Quick single-example inference testing
- convert_to_mlx_format.py: Converts training data to MLX format
- train_mac.py: Training script that produces the adapters validated here
"""

import json
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
from pathlib import Path
import numpy as np
from tqdm import tqdm
import time

def load_test_data(test_file: str) -> list:
    """
    Load test data from JSONL file.

    Args:
        test_file: Path to JSONL file containing test examples.
                  Each line should be a JSON object with a "text" field.

    Returns:
        List of dictionaries, each containing a test example.

    Example:
        >>> test_examples = load_test_data("data/mlx_small/test.jsonl")
        >>> len(test_examples)
        500
    """
    test_examples = []
    with open(test_file, 'r') as f:
        for line in f:
            example = json.loads(line)
            test_examples.append(example)
    return test_examples

def calculate_perplexity(model, tokenizer, text: str, max_tokens: int = 2048) -> float:
    """
    Calculate perplexity for a text sequence using the given model.

    Perplexity measures how well the model predicts the text. Lower perplexity
    indicates better performance. It's calculated as exp(cross_entropy_loss).

    Args:
        model: The MLX language model to evaluate
        tokenizer: Tokenizer corresponding to the model
        text: Input text to evaluate
        max_tokens: Maximum number of tokens to process (default: 2048)
                   Longer sequences are truncated to prevent memory issues

    Returns:
        float: Perplexity score. Returns inf on error.
               - Excellent: < 5
               - Very Good: 5-10
               - Good: 10-20
               - Fair: 20-50
               - Needs Improvement: > 50

    Note:
        Uses next-token prediction paradigm: model predicts token[i+1] from
        tokens[0:i]. This is why logits and labels are shifted.
    """
    try:
        # Tokenize the input text into integer token IDs
        tokens = tokenizer.encode(text)

        # Truncate if sequence exceeds max length to prevent OOM errors
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]

        # Convert to MLX array with batch dimension [1, seq_len]
        tokens_mx = mx.array([tokens])

        # Forward pass: get model predictions (logits) for each token position
        # Shape: [batch_size, seq_len, vocab_size]
        logits = model(tokens_mx)

        # Prepare for next-token prediction task:
        # - shift_logits: predictions for tokens [0, seq_len-1]
        # - shift_labels: actual tokens [1, seq_len]
        # This aligns predictions with targets (predict next token)
        shift_logits = logits[0, :-1, :]  # Remove last prediction (no target for it)
        shift_labels = tokens_mx[0, 1:]    # Remove first token (no prediction for it)

        # Calculate cross-entropy loss (negative log-likelihood)
        # This measures how well the predicted distribution matches actual tokens
        loss = nn.losses.cross_entropy(
            shift_logits,
            shift_labels,
            reduction='mean'  # Average loss across all token positions
        )

        # Convert loss to perplexity: perplexity = exp(loss)
        # Perplexity can be interpreted as the "effective vocabulary size"
        # the model is uncertain about at each step
        perplexity = float(mx.exp(loss))

        return perplexity
    except Exception as e:
        print(f"Error calculating perplexity: {e}")
        return float('inf')  # Return infinity to indicate failure

def generate_sample_output(model, tokenizer, prompt: str, max_tokens: int = 512) -> str:
    """
    Generate text completion from the model given a prompt.

    Args:
        model: The MLX language model
        tokenizer: Tokenizer corresponding to the model
        prompt: Input prompt to complete
        max_tokens: Maximum number of tokens to generate (default: 512)

    Returns:
        str: Generated text (includes the prompt). Returns error message on failure.

    Note:
        Uses temperature=0.7 for balanced creativity vs coherence.
        Lower temperature (e.g., 0.3) makes output more deterministic.
        Higher temperature (e.g., 1.0) makes output more creative/random.
    """
    try:
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=0.7,  # Moderate randomness for natural but consistent responses
            verbose=False  # Suppress generation progress output
        )
        return response
    except Exception as e:
        return f"Error: {e}"

def validate_model(
    model_path: str,
    adapter_path: str,
    test_file: str,
    num_samples: int = 100,
    num_generation_samples: int = 5
):
    """
    Comprehensive validation of a fine-tuned MLX model.

    Performs two types of evaluation:
    1. Quantitative: Calculates perplexity statistics on test samples
    2. Qualitative: Generates and displays sample responses

    Args:
        model_path: Path to base model directory (e.g., "models/mistral-7b-mlx")
        adapter_path: Path to LoRA adapter weights (.safetensors file)
        test_file: Path to test data JSONL file
        num_samples: Number of examples to use for perplexity calculation (default: 100)
                    Higher values give more stable metrics but take longer
        num_generation_samples: Number of sample generations to display (default: 5)

    Outputs:
        Prints validation report including:
        - Perplexity statistics (mean, median, std, min, max)
        - Quality assessment based on perplexity
        - Sample generations with expected vs actual comparison

    Note:
        Perplexity is calculated on a random sample to balance accuracy and speed.
        Full test set evaluation would be more accurate but much slower.
    """

    print("=" * 80)
    print("MLX Model Validation")
    print("=" * 80)
    print(f"\nModel: {model_path}")
    print(f"Adapter: {adapter_path}")
    print(f"Test Data: {test_file}\n")

    # Load base model with LoRA adapter applied
    # The adapter contains fine-tuned weights learned during training
    print("Loading model...")
    start_time = time.time()
    model, tokenizer = load(model_path, adapter_path=adapter_path)
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f}s\n")

    # Load test data (held-out examples not seen during training)
    print("Loading test data...")
    test_examples = load_test_data(test_file)
    print(f"Loaded {len(test_examples)} test examples\n")

    # Randomly sample a subset for perplexity calculation
    # Full evaluation would be too slow for iterative development
    sample_size = min(num_samples, len(test_examples))
    test_sample = np.random.choice(test_examples, sample_size, replace=False)

    print(f"Calculating perplexity on {sample_size} samples...")
    perplexities = []

    # Calculate perplexity for each sampled example
    for example in tqdm(test_sample, desc="Calculating perplexity"):
        text = example.get('text', '')
        if text:
            ppl = calculate_perplexity(model, tokenizer, text)
            # Only include valid perplexity values (exclude errors/inf)
            if ppl != float('inf'):
                perplexities.append(ppl)

    # Calculate and display statistical metrics
    if perplexities:
        mean_ppl = np.mean(perplexities)
        median_ppl = np.median(perplexities)
        std_ppl = np.std(perplexities)
        min_ppl = np.min(perplexities)
        max_ppl = np.max(perplexities)

        print("\n" + "=" * 80)
        print("PERPLEXITY RESULTS")
        print("=" * 80)
        print(f"Samples evaluated: {len(perplexities)}")
        print(f"Mean Perplexity:   {mean_ppl:.4f}")
        print(f"Median Perplexity: {median_ppl:.4f}")
        print(f"Std Deviation:     {std_ppl:.4f}")
        print(f"Min Perplexity:    {min_ppl:.4f}")
        print(f"Max Perplexity:    {max_ppl:.4f}")
    else:
        print("\nNo valid perplexity calculations")

    # Generate and display sample outputs for qualitative assessment
    print("\n" + "=" * 80)
    print("SAMPLE GENERATIONS")
    print("=" * 80)

    # Randomly select examples to generate completions for
    generation_samples = np.random.choice(
        test_examples,
        min(num_generation_samples, len(test_examples)),
        replace=False
    )

    for i, example in enumerate(generation_samples, 1):
        text = example.get('text', '')

        # Extract user prompt and expected response
        # MLX format uses <|assistant|> as delimiter between user and assistant
        if '<|assistant|>' in text:
            # Split at delimiter to separate prompt from expected completion
            user_prompt = text.split('<|assistant|>')[0] + '<|assistant|>'
            expected_response = text.split('<|assistant|>')[1].strip()
        else:
            # Fallback if format is unexpected
            user_prompt = text[:200]  # Just use first 200 chars
            expected_response = "N/A"

        print(f"\n{'─' * 80}")
        print(f"SAMPLE {i}")
        print(f"{'─' * 80}")
        print(f"\nPrompt:\n{user_prompt}\n")

        # Generate model response and measure latency
        print("Generating response...")
        start_time = time.time()
        generated = generate_sample_output(model, tokenizer, user_prompt, max_tokens=500)
        gen_time = time.time() - start_time

        # Extract only the newly generated text (remove echoed prompt)
        # The generate function returns prompt + completion
        if user_prompt in generated:
            generated_response = generated.replace(user_prompt, '').strip()
        else:
            generated_response = generated

        print(f"\nGenerated Response ({gen_time:.2f}s):\n{generated_response}\n")

        # Display expected response for manual comparison
        if expected_response != "N/A":
            # Truncate long responses for readability
            print(f"Expected Response:\n{expected_response[:500]}{'...' if len(expected_response) > 500 else ''}\n")

    # Print validation summary with quality assessment
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"✅ Model loaded successfully")
    print(f"✅ Test set: {len(test_examples)} examples")
    print(f"✅ Perplexity evaluated on {len(perplexities)} samples")
    if perplexities:
        print(f"✅ Mean Perplexity: {mean_ppl:.4f}")

        # Interpret perplexity score to assess model quality
        # These thresholds are heuristic and domain-specific
        if mean_ppl < 5:
            quality = "Excellent"
            emoji = "🟢"
        elif mean_ppl < 10:
            quality = "Very Good"
            emoji = "🟢"
        elif mean_ppl < 20:
            quality = "Good"
            emoji = "🟡"
        elif mean_ppl < 50:
            quality = "Fair"
            emoji = "🟡"
        else:
            quality = "Needs Improvement"
            emoji = "🔴"

        print(f"{emoji} Quality Assessment: {quality}")
    print(f"✅ Generated {num_generation_samples} sample responses")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate MLX fine-tuned model")
    parser.add_argument(
        "--model",
        default="models/mistral-7b-mlx",
        help="Path to base model"
    )
    parser.add_argument(
        "--adapter",
        default="adapters/contact-center-mlx-small/PRODUCTION_MODEL.safetensors",
        help="Path to LoRA adapter"
    )
    parser.add_argument(
        "--test-data",
        default="data/mlx_small/test.jsonl",
        help="Path to test data"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples for perplexity calculation"
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=5,
        help="Number of generation samples"
    )

    args = parser.parse_args()

    validate_model(
        model_path=args.model,
        adapter_path=args.adapter,
        test_file=args.test_data,
        num_samples=args.num_samples,
        num_generation_samples=args.num_generations
    )
