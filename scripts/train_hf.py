#!/usr/bin/env python3
"""
HuggingFace Training Script for Contact Center Analytics

Fine-tune Mistral 7B / Llama 3 8B on multi-task contact center data.

Usage:
    python train_hf.py --config ../configs/mistral_7b.yaml
    python train_hf.py --config ../configs/llama3_8b.yaml
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
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import evaluate


class ContactCenterTrainer:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model_name = self.config['model']['name']
        self.output_dir = self.config['training']['output_dir']

        print("=" * 80)
        print(f"CONTACT CENTER ANALYTICS - MODEL FINE-TUNING")
        print("=" * 80)
        print(f"Model: {self.model_name}")
        print(f"Output: {self.output_dir}")

    def load_model_and_tokenizer(self):
        """Load model with 4-bit quantization and LoRA"""
        print("\nLoading model and tokenizer...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )

        # Prepare for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)

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

        training_config = self.config['training']

        # Training arguments
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
            evaluation_strategy=training_config['evaluation_strategy'],
            load_best_model_at_end=training_config['load_best_model_at_end'],
            metric_for_best_model=training_config['metric_for_best_model'],
            bf16=training_config['bf16'],
            gradient_checkpointing=training_config['gradient_checkpointing'],
            report_to="wandb" if self.config['wandb']['enabled'] else "none"
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

    def run(self):
        """Run full training pipeline"""
        self.load_model_and_tokenizer()
        self.load_datasets()
        self.train()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune contact center analytics model")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    args = parser.parse_args()

    trainer = ContactCenterTrainer(args.config)
    trainer.run()


if __name__ == "__main__":
    main()
