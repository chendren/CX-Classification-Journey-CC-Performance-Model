# Training on Mac (M4 Max with 64GB RAM)

## Your Setup

- **CPU:** Apple M4 Max
- **RAM:** 64GB
- **Architecture:** Apple Silicon (arm64)
- **Perfect for:** Local fine-tuning with 4-bit quantization

## Quick Start

### 1. Install Dependencies

```bash
cd /Users/chadhendren/contact-center-analytics/fine-tuning

# Install training dependencies
pip install torch transformers datasets accelerate peft bitsandbytes pyyaml
```

### 2. Start Training

```bash
python scripts/train_mac.py --config configs/mistral_7b_mac.yaml
```

That's it! Training will start automatically.

## What to Expect

### Training Time
- **Estimated:** 4-8 hours for 3 epochs
- **Progress:** You'll see updates every 10 steps
- **Checkpoints:** Saved every 500 steps to `models/mistral-7b-contact-center/`

### Memory Usage
- **Model:** ~4GB (4-bit quantization)
- **Training:** ~20-30GB total
- **Safe with:** Your 64GB RAM

### Performance Monitoring

You'll see output like this:
```
Step 10/4848  Loss: 1.234  Time: 2.5s/batch
Step 20/4848  Loss: 1.198  Time: 2.4s/batch
...
Evaluation:   eval_loss: 0.987  perplexity: 2.68
```

## Training Configuration

**Optimized for M4 Max:**
- Batch size: 2 per device
- Gradient accumulation: 8 steps (effective batch size: 16)
- 4-bit quantization (saves memory)
- LoRA (only trains 0.1% of parameters)
- Gradient checkpointing (memory efficient)

## Output

After training completes, you'll have:

```
models/mistral-7b-contact-center/
├── adapter_config.json       # LoRA adapter config
├── adapter_model.bin          # Trained LoRA weights (~150MB)
├── config.json                # Model config
├── tokenizer_config.json      # Tokenizer config
├── special_tokens_map.json    # Special tokens
└── test_results.json          # Final test metrics
```

## Using Your Trained Model

After training, you can use it immediately:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    load_in_4bit=True,
    device_map="auto"
)

# Load your fine-tuned adapter
model = PeftModel.from_pretrained(
    base_model,
    "models/mistral-7b-contact-center"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("models/mistral-7b-contact-center")

# Use it!
prompt = "[INST] Analyze this conversation and predict CSAT... [/INST]"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=512)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Tips for Mac Training

### Memory Management
- **Close other apps** during training to free RAM
- **Monitor memory:** Use Activity Monitor to check usage
- **If you run out of memory:**
  - Reduce `per_device_train_batch_size` from 2 to 1
  - Reduce `max_length` from 2048 to 1024

### Speed Optimization
- **Plug in your Mac** - training uses significant power
- **Good ventilation** - M4 Max will get warm
- **Background tasks** - Close browsers, Slack, etc.

### Stopping and Resuming
If you need to stop training:
- Press `Ctrl+C` to stop gracefully
- Training will save a checkpoint
- To resume: Run the same command again (it will resume from last checkpoint)

## Cost

- **Training:** Free (runs on your Mac)
- **Time:** 4-8 hours (overnight training recommended)
- **Electricity:** ~$0.50-1.00 total

## Troubleshooting

### "Out of memory" error
```bash
# Edit configs/mistral_7b_mac.yaml
# Change: per_device_train_batch_size: 1
# Change: gradient_accumulation_steps: 16
```

### "MPS not available"
```bash
# Check macOS version (needs 13.0+)
sw_vers

# Make sure you have the latest PyTorch
pip install --upgrade torch
```

### Slow training (>5s/batch)
```bash
# Make sure other apps are closed
# Check Activity Monitor for memory pressure
# Consider reducing max_length to 1024
```

## Next Steps

After training:
1. **Test the model** on a few sample conversations
2. **Compare results** to GPT-4/Claude prompting
3. **Deploy** as a local API (see inference/ directory)
4. **Iterate** - try different hyperparameters if needed

## Questions?

- Check `README.md` for general training info
- Review `configs/mistral_7b_mac.yaml` for parameter details
- Training logs will be in `models/mistral-7b-contact-center/`
