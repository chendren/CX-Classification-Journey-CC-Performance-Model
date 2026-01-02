#!/usr/bin/env python3
"""
Convert temporal training data to MLX format.

MLX (Apple's ML framework) requires a specific data format for fine-tuning:
- JSONL files (one JSON object per line)
- Each object must have a "text" field containing the full conversation
- Conversation must use special tokens: <|user|> and <|assistant|>

This script converts from the ChatML message format to MLX's flat text format.

Input format (ChatML):
    {
        "messages": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ],
        "temporal_features": {...}
    }

Output format (MLX):
    {
        "text": "<|user|>\\n...\\n<|assistant|>\\n..."
    }

Dependencies:
- json: Parse and write JSONL
- pathlib: File path handling

Usage:
    python convert_to_mlx_format.py

    By default converts:
    - data/train_temporal.jsonl -> data/mlx/train.jsonl
    - data/validation_temporal.jsonl -> data/mlx/valid.jsonl
    - data/test_temporal.jsonl -> data/mlx/test.jsonl

Related Files:
- add_temporal_features.py: Creates the input temporal data
- train_mac.py: MLX training script that uses this format
- validate_model.py: Validation script that expects MLX format
"""

import json
from pathlib import Path

def convert_to_mlx_format(input_file: str, output_file: str):
    """
    Convert ChatML format training data to MLX flat text format.

    Args:
        input_file: Path to input JSONL file in ChatML format
                   Each line should have a "messages" field with user/assistant turns
        output_file: Path where MLX format JSONL will be written
                    Each line will have a "text" field with formatted conversation

    Note:
        - Examples without exactly 2 messages (user + assistant) are skipped
        - Temporal features and other metadata are discarded (MLX only uses text)
        - The special tokens <|user|> and <|assistant|> are required by MLX

    Format Details:
        MLX expects conversations in this format:
        <|user|>
        [user message]
        <|assistant|>
        [assistant response]

        During training, MLX will learn to predict tokens after <|assistant|>
    """

    # Process files line by line (streaming to handle large files)
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            example = json.loads(line)

            # Extract messages array from example
            messages = example.get('messages', [])

            # Skip examples that don't have exactly 2 messages (user + assistant)
            # MLX fine-tuning expects single-turn conversations
            if len(messages) < 2:
                continue

            # Extract user and assistant content
            user_msg = messages[0]['content']
            assistant_msg = messages[1]['content']

            # Format as MLX conversation with special tokens
            # \n separators help the model learn turn boundaries
            text = f"<|user|>\n{user_msg}\n<|assistant|>\n{assistant_msg}"

            # Create MLX format example (just a text field)
            mlx_example = {"text": text}

            # Write as JSONL (one JSON object per line)
            f_out.write(json.dumps(mlx_example) + '\n')

    print(f"Converted {input_file} -> {output_file}")

if __name__ == "__main__":
    # Create output directory for MLX-formatted data
    mlx_data_dir = Path("data/mlx")
    mlx_data_dir.mkdir(exist_ok=True)

    # Convert all three data splits to MLX format
    # MLX expects specific filenames: train.jsonl, valid.jsonl, test.jsonl
    print("Converting training data to MLX format...\n")

    convert_to_mlx_format("data/train_temporal.jsonl", "data/mlx/train.jsonl")
    convert_to_mlx_format("data/validation_temporal.jsonl", "data/mlx/valid.jsonl")
    convert_to_mlx_format("data/test_temporal.jsonl", "data/mlx/test.jsonl")

    print("\n✅ MLX data ready!")
    print("\nNext steps:")
    print("  1. Run: python scripts/train_mac.py")
    print("  2. Validate: python scripts/validate_model.py")
