#!/usr/bin/env python3
"""
Replicate Training Script for Contact Center Analytics

Easiest way to fine-tune - no GPU needed, pay only for training time.

Setup:
    1. pip install replicate
    2. export REPLICATE_API_TOKEN="your-token-here"
    3. Upload training data to public URL (or use Replicate's file hosting)

Usage:
    python train_replicate.py \\
        --model "mistralai/mistral-7b-instruct-v0.3" \\
        --train_data_url "https://your-url/train_chatml.jsonl" \\
        --val_data_url "https://your-url/validation_chatml.jsonl" \\
        --destination "your-username/contact-center-mistral"
"""

import os
import argparse
import replicate
from pathlib import Path


def upload_data_to_replicate(file_path: str) -> str:
    """
    Upload a file to Replicate's file hosting
    Returns the public URL
    """
    print(f"Uploading {file_path} to Replicate...")
    with open(file_path, 'rb') as f:
        file = replicate.files.create(f)
    print(f"✅ Uploaded: {file.urls['get']}")
    return file.urls['get']


def start_training(
    model: str,
    train_data_url: str,
    val_data_url: str,
    destination: str,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    batch_size: int = 4
):
    """Start fine-tuning on Replicate"""

    print("=" * 80)
    print("CONTACT CENTER ANALYTICS - REPLICATE TRAINING")
    print("=" * 80)
    print(f"Base model: {model}")
    print(f"Destination: {destination}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print()

    # Create training
    print("🚀 Starting training...")
    training = replicate.trainings.create(
        destination=destination,
        model=model,
        input={
            "train_data": train_data_url,
            "num_train_epochs": num_epochs,
            "learning_rate": learning_rate,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "gradient_accumulation_steps": 4,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
        }
    )

    print(f"\n✅ Training started!")
    print(f"   Training ID: {training.id}")
    print(f"   Status: {training.status}")
    print(f"   View at: https://replicate.com/trainings/{training.id}")
    print()
    print("You can monitor progress at the URL above.")
    print("Training typically takes 3-5 hours for 3 epochs.")
    print()
    print("To check status:")
    print(f"  training = replicate.trainings.get('{training.id}')")
    print(f"  print(training.status)")
    print()
    print("When complete, you can use your model:")
    print(f"  model = replicate.models.get('{destination}')")
    print(f"  prediction = model.predict(input={{...}})")

    return training


def main():
    parser = argparse.ArgumentParser(description="Train contact center model on Replicate")
    parser.add_argument('--model', type=str, default="mistralai/mistral-7b-instruct-v0.3",
                        help='Base model to fine-tune')
    parser.add_argument('--train_data_url', type=str,
                        help='Public URL to training data (JSONL)')
    parser.add_argument('--val_data_url', type=str,
                        help='Public URL to validation data (JSONL)')
    parser.add_argument('--train_data_file', type=str,
                        help='Local training data file (will upload)')
    parser.add_argument('--val_data_file', type=str,
                        help='Local validation data file (will upload)')
    parser.add_argument('--destination', type=str, required=True,
                        help='Destination model name (e.g., username/model-name)')
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size per device')

    args = parser.parse_args()

    # Check if API token is set
    if not os.getenv('REPLICATE_API_TOKEN'):
        print("❌ Error: REPLICATE_API_TOKEN environment variable not set")
        print("   Get your token at: https://replicate.com/account/api-tokens")
        print("   Then: export REPLICATE_API_TOKEN='your-token-here'")
        return

    # Handle data URLs
    train_url = args.train_data_url
    val_url = args.val_data_url

    if args.train_data_file:
        train_url = upload_data_to_replicate(args.train_data_file)

    if args.val_data_file:
        val_url = upload_data_to_replicate(args.val_data_file)

    if not train_url:
        print("❌ Error: Must provide either --train_data_url or --train_data_file")
        return

    # Start training
    training = start_training(
        model=args.model,
        train_data_url=train_url,
        val_data_url=val_url,
        destination=args.destination,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
