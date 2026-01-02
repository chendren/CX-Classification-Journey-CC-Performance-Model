#!/usr/bin/env python3
"""
Create smaller training dataset for fast iteration and testing.

During model development, it's often useful to work with smaller datasets to:
1. Reduce training time for quick experiments
2. Test the training pipeline end-to-end
3. Debug issues without waiting for full training runs
4. Validate hyperparameters before full-scale training

This script creates a small subset of the training data while keeping
validation and test sets at their original size for consistent evaluation.

Dependencies:
- json: Parse JSONL format
- random: Random sampling
- pathlib: File path handling

Usage:
    python create_small_dataset.py

    By default:
    - Input: data/train_temporal.jsonl
    - Output: data/train_temporal_small.jsonl
    - Size: 1,500 examples

Related Files:
- split_temporal_data.py: Creates the full training splits
- add_temporal_features.py: Adds temporal features to data
- train_mac.py: Training script that can use the small dataset
"""

import json
import random
from pathlib import Path

def create_small_dataset(input_file: str, output_file: str, size: int = 1500):
    """
    Randomly sample a smaller training set from full training data.

    Args:
        input_file: Path to full training JSONL file
        output_file: Path where small dataset will be written
        size: Number of examples to sample (default: 1500)
              If input has fewer examples, all will be used

    Note:
        Uses fixed random seed (42) for reproducibility. The same subset
        will be selected each time the script is run with the same inputs.

        Validation and test sets should NOT be reduced - keep them at full
        size for accurate performance measurement.

    Example:
        >>> create_small_dataset("data/train.jsonl", "data/train_small.jsonl", size=1000)
        Original size: 10,000 examples
        Sampled size: 1,000 examples
        ✅ Created data/train_small.jsonl
    """

    # Load all examples from JSONL file
    with open(input_file, 'r') as f:
        examples = [json.loads(line) for line in f]

    print(f"Original size: {len(examples):,} examples")

    # Randomly sample subset (without replacement)
    # Fixed seed ensures reproducibility
    random.seed(42)  # For reproducibility
    # Use min() to handle case where input has fewer than 'size' examples
    sampled = random.sample(examples, min(size, len(examples)))

    print(f"Sampled size: {len(sampled):,} examples")

    # Write sampled dataset in JSONL format
    with open(output_file, 'w') as f:
        for example in sampled:
            f.write(json.dumps(example) + '\n')

    print(f"✅ Created {output_file}")

if __name__ == "__main__":
    create_small_dataset(
        "data/train_temporal.jsonl",
        "data/train_temporal_small.jsonl",
        size=1500
    )
