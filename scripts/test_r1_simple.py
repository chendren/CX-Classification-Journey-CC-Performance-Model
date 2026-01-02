#!/usr/bin/env python3
"""
Simple R1 Test - Single threaded test to verify MLX R1 works
"""

import json
import mlx.core as mx
from mlx_lm import load, generate

print("Loading DeepSeek R1...")
model, tokenizer = load("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
print("✅ Model loaded!")

# Load one example
with open('data/train_chatml.jsonl', 'r') as f:
    example = json.loads(f.readline())

user_msg = example['messages'][0]['content']
print(f"\n📝 Input length: {len(user_msg)} chars")

# Simple prompt
prompt = f"{user_msg}\n\nProvide a brief analysis:"

print("\n🔮 Generating response...")
response = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=500,
    verbose=True
)

print(f"\n\n✅ Generated response ({len(response)} chars):")
print("="*80)
print(response)
print("="*80)
