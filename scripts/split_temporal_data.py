#!/usr/bin/env python3
"""
Split temporal training data into train/validation/test sets.

This script takes a single JSONL file containing all training examples
and splits it into three sets following standard ML practice:
- 80% training: Used to train the model
- 10% validation: Used for hyperparameter tuning and early stopping
- 10% test: Held-out set for final evaluation

The split is randomized with a fixed seed for reproducibility.

Dependencies:
- json: Parse JSONL format
- random: Randomize split
- pathlib: File path handling

Usage:
    python split_temporal_data.py

    By default processes data/train_temporal.jsonl
    Edit the __main__ block to change input file.

Related Files:
- add_temporal_features.py: Generates the input temporal data
- create_small_dataset.py: Creates smaller subsets for fast iteration
"""

import json
import random
from pathlib import Path

def split_data(input_file: str, output_dir: str = "data"):
    """
    Split JSONL data into train/validation/test sets.

    Args:
        input_file: Path to input JSONL file containing all examples
        output_dir: Directory where split files will be written (default: "data")

    Outputs:
        Three JSONL files in output_dir:
        - train_temporal.jsonl (80% of data)
        - validation_temporal.jsonl (10% of data)
        - test_temporal.jsonl (10% of data)

    Note:
        Uses fixed random seed (42) for reproducibility. Running this script
        multiple times with the same input will produce identical splits.

        Test set gets the remainder to ensure all data is used (may be slightly
        larger than exactly 10% due to rounding).
    """

    # Load all examples from JSONL file (one JSON object per line)
    with open(input_file, 'r') as f:
        examples = [json.loads(line) for line in f]

    total = len(examples)
    print(f"Total examples: {total:,}")

    # Shuffle for random split
    # Fixed seed ensures reproducibility across runs
    random.seed(42)  # For reproducibility
    random.shuffle(examples)

    # Calculate split sizes
    train_size = int(0.8 * total)  # 80% for training
    val_size = int(0.1 * total)    # 10% for validation
    # test gets the remainder to ensure we use all data (no examples discarded)

    # Split into three non-overlapping sets
    train_data = examples[:train_size]
    val_data = examples[train_size:train_size + val_size]
    test_data = examples[train_size + val_size:]  # Remainder goes to test

    # Display split statistics
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_data):,} ({len(train_data)/total*100:.1f}%)")
    print(f"  Validation: {len(val_data):,} ({len(val_data)/total*100:.1f}%)")
    print(f"  Test: {len(test_data):,} ({len(test_data)/total*100:.1f}%)")

    # Prepare output directory
    output_path = Path(output_dir)

    # Define output file paths
    train_file = output_path / "train_temporal.jsonl"
    val_file = output_path / "validation_temporal.jsonl"
    test_file = output_path / "test_temporal.jsonl"

    print(f"\nWriting files...")

    # Write training set (largest split, used for learning)
    with open(train_file, 'w') as f:
        for example in train_data:
            f.write(json.dumps(example) + '\n')
    print(f"  ✅ {train_file} ({len(train_data):,} examples)")

    # Write validation set (used during training for hyperparameter tuning)
    with open(val_file, 'w') as f:
        for example in val_data:
            f.write(json.dumps(example) + '\n')
    print(f"  ✅ {val_file} ({len(val_data):,} examples)")

    # Write test set (held-out for final evaluation only)
    with open(test_file, 'w') as f:
        for example in test_data:
            f.write(json.dumps(example) + '\n')
    print(f"  ✅ {test_file} ({len(test_data):,} examples)")

    print(f"\n✅ Split complete!")

if __name__ == "__main__":
    split_data("data/train_temporal.jsonl")
