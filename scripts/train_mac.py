#!/usr/bin/env python3
"""
Mac Apple Silicon Training Script for Contact Center Analytics

Optimized for M1/M2/M3/M4 Macs with MPS (Metal Performance Shaders)

Requirements:
    - Apple Silicon Mac (M1/M2/M3/M4)
    - 32GB+ RAM (64GB recommended)
    - macOS 13.0+

Usage:
    python train_mac.py --config ../configs/mistral_7b_mac.yaml
"""

import os
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model


class MacContactCenterTrainer:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model_name = self.config['model']['name']
        self.output_dir = self.config['training']['output_dir']

        # Check for Apple Silicon
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("✅ Apple Silicon GPU (MPS) detected")
        else:
            self.device = "cpu"
            print("⚠️  MPS not available, using CPU (will be slower)")

        print("=" * 80)
        print(f"CONTACT CENTER ANALYTICS - MAC TRAINING (M4 Max)")
        print("=" * 80)
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Output: {self.output_dir}")
        print(f"RAM: 64GB")

    def load_model_and_tokenizer(self):
        """Load model optimized for Mac"""
        print("\nLoading model and tokenizer...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # Load model in 16-bit (faster, better quality, fits in 64GB RAM)
        print("Loading model in 16-bit precision...")

        # Load model (for MPS, load without device_map and manually move to device)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )

        # Move model to MPS device
        self.model = self.model.to(self.device)

        # Enable gradient checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable()

        # Apply LoRA
        if self.config['lora']['enabled']:
            lora_config = LoraConfig(
                r=self.config['lora']['r'],
                lora_alpha=self.config['lora']['lora_alpha'],
                lora_dropout=self.config['lora']['lora_dropout'],
                bias=self.config['lora']['bias'],
                task_type=self.config['lora']['task_type'],
                target_modules=self.config['lora']['target_modules']
            )
            self.model = get_peft_model(self.model, lora_config)

            # Print trainable parameters
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            all_params = sum(p.numel() for p in self.model.parameters())
            print(f"✅ LoRA enabled: {trainable_params:,} / {all_params:,} parameters trainable "
                  f"({100 * trainable_params / all_params:.2f}%)")

        print(f"✅ Model loaded successfully")

        # Estimate memory usage
        model_size_mb = sum(p.nelement() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
        print(f"   Model size: ~{model_size_mb:.0f} MB")

    def load_datasets(self):
        """Load training, validation, and test datasets"""
        print("\nLoading datasets...")

        data_config = self.config['data']

        # Load datasets
        dataset = load_dataset('json', data_files={
            'train': data_config['train_file'],
            'validation': data_config['validation_file'],
            'test': data_config['test_file']
        })

        print(f"  Training:   {len(dataset['train']):,} examples")
        print(f"  Validation: {len(dataset['validation']):,} examples")
        print(f"  Test:       {len(dataset['test']):,} examples")

        # Tokenize datasets
        def tokenize_function(examples):
            # Format: <s>[INST] {prompt} [/INST] {completion} </s>
            formatted = []
            for messages in examples['messages']:
                text = ""
                for msg in messages:
                    if msg['role'] == 'user':
                        text += f"[INST] {msg['content']} [/INST] "
                    elif msg['role'] == 'assistant':
                        text += f"{msg['content']}"
                formatted.append(text)

            tokenized = self.tokenizer(
                formatted,
                truncation=True,
                max_length=data_config['max_length'],
                padding='max_length'
            )
            tokenized['labels'] = tokenized['input_ids'].copy()
            return tokenized

        print("Tokenizing datasets...")
        self.train_dataset = dataset['train'].map(
            tokenize_function,
            batched=True,
            remove_columns=dataset['train'].column_names,
            desc="Tokenizing training set"
        )

        self.eval_dataset = dataset['validation'].map(
            tokenize_function,
            batched=True,
            remove_columns=dataset['validation'].column_names,
            desc="Tokenizing validation set"
        )

        self.test_dataset = dataset['test'].map(
            tokenize_function,
            batched=True,
            remove_columns=dataset['test'].column_names,
            desc="Tokenizing test set"
        )

        print("✅ Datasets tokenized successfully")

    def train(self):
        """Run training"""
        print("\nStarting training...")
        print("⏱️  Estimated time: 4-8 hours on M4 Max")
        print("💾 Model will be saved to:", self.output_dir)
        print()

        training_config = self.config['training']

        # Training arguments optimized for Mac
        training_args = TrainingArguments(
            output_dir=training_config['output_dir'],
            num_train_epochs=training_config['num_train_epochs'],
            per_device_train_batch_size=training_config['per_device_train_batch_size'],
            per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            learning_rate=training_config['learning_rate'],
            warmup_ratio=training_config['warmup_ratio'],
            weight_decay=training_config['weight_decay'],
            max_grad_norm=training_config['max_grad_norm'],
            lr_scheduler_type=training_config['lr_scheduler_type'],
            logging_steps=training_config['logging_steps'],
            eval_steps=training_config['eval_steps'],
            save_steps=training_config['save_steps'],
            save_total_limit=training_config['save_total_limit'],
            eval_strategy=training_config['evaluation_strategy'],
            load_best_model_at_end=training_config['load_best_model_at_end'],
            metric_for_best_model=training_config['metric_for_best_model'],
            # Mac-specific settings
            fp16=False,  # MPS doesn't support fp16 training yet
            gradient_checkpointing=training_config.get('gradient_checkpointing', True),
            dataloader_num_workers=0,  # Mac MPS works best with 0 workers
            report_to="none",  # Disable wandb for local training
            use_cpu=False  # Let PyTorch auto-detect MPS
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator
        )

        # Train
        print("🚀 Training started...")
        print("   You can monitor progress in the terminal")
        print("   Training will checkpoint every 500 steps")
        print()

        trainer.train()

        # Save final model
        trainer.save_model(training_config['output_dir'])
        self.tokenizer.save_pretrained(training_config['output_dir'])

        print(f"\n✅ Training complete!")
        print(f"   Model saved to: {training_config['output_dir']}")

        # Evaluate on test set
        print("\n📊 Evaluating on test set...")
        test_results = trainer.evaluate(self.test_dataset)
        print(f"Test perplexity: {torch.exp(torch.tensor(test_results['eval_loss'])):.2f}")

        # Save test results
        with open(f"{training_config['output_dir']}/test_results.json", 'w') as f:
            json.dump(test_results, f, indent=2)

        print("\n🎉 All done! Your model is ready to use.")

    def run(self):
        """Run full training pipeline"""
        self.load_model_and_tokenizer()
        self.load_datasets()
        self.train()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune contact center model on Mac")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    args = parser.parse_args()

    trainer = MacContactCenterTrainer(args.config)
    trainer.run()


if __name__ == "__main__":
    main()
