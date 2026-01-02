# Contact Center Analytics - Model Fine-Tuning

## Overview

This directory contains everything needed to fine-tune smaller, faster models for contact center analytics tasks using the synthetic training data we generated.

## Training Data

**Location:** `data/`
- **Training set:** 25,790 examples (80%)
- **Validation set:** 3,223 examples (10%)
- **Test set:** 3,225 examples (10%)
- **Total:** 32,238 multi-task examples

### Tasks Covered
1. **Quality Scoring** - Overall quality score (0-100) + effectiveness rating
2. **CSAT Prediction** - Customer satisfaction prediction
3. **Issue Classification** - Primary issue categorization
4. **Churn Risk** - Customer churn risk assessment (0-100)
5. **Journey Type** - Customer journey classification
6. **Coaching Recommendations** - Agent coaching insights

## Recommended Models

### Open Source Models (7-8B parameters)
- **Mistral 7B Instruct v0.3** - Best overall performance
- **Llama 3 8B Instruct** - Strong generalization
- **Qwen 2.5 7B Instruct** - Excellent for structured output

## Fine-Tuning Options

### Option 1: HuggingFace (Local/Cloud)

**Requirements:**
- GPU with 24GB+ VRAM (A10G, A100, H100)
- Python 3.10+
- transformers, datasets, peft, bitsandbytes

**Cost:** ~$1-2/hour on cloud GPUs

```bash
# Install dependencies
pip install transformers datasets peft bitsandbytes accelerate

# Run training
python scripts/train_hf.py \\
    --model_name "mistralai/Mistral-7B-Instruct-v0.3" \\
    --train_file data/train_chatml.jsonl \\
    --val_file data/validation_chatml.jsonl \\
    --output_dir models/mistral-7b-contact-center \\
    --num_epochs 3 \\
    --batch_size 4 \\
    --learning_rate 2e-5
```

### Option 2: Replicate (Easiest - No GPU needed)

**Cost:** ~$0.002/second of training (~$7-15 total)

```bash
# Upload data to public URL or use Replicate's data hosting
# Then create training via API or web UI

# Example with Replicate API
import replicate

training = replicate.trainings.create(
    destination="your-username/contact-center-mistral",
    model="mistralai/mistral-7b-instruct-v0.3",
    input={
        "train_data": "https://your-data-url/train_chatml.jsonl",
        "val_data": "https://your-data-url/validation_chatml.jsonl",
        "num_train_epochs": 3,
        "learning_rate": 2e-5
    }
)
```

### Option 3: Modal (Serverless GPU)

**Cost:** ~$0.50-2/hour, only pay for actual training time

```bash
# Deploy training job to Modal
modal deploy scripts/train_modal.py

# Run training
modal run scripts/train_modal.py --config configs/mistral_7b.yaml
```

## Training Configuration

### Recommended Hyperparameters

```yaml
model:
  name: "mistralai/Mistral-7B-Instruct-v0.3"
  max_seq_length: 2048

training:
  num_epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 4  # Effective batch size: 16
  learning_rate: 2e-5
  warmup_ratio: 0.03
  weight_decay: 0.01

lora:  # Parameter-efficient fine-tuning
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
```

## Expected Training Time

- **Mistral 7B on A100:** ~3-4 hours
- **Llama 3 8B on A100:** ~4-5 hours
- **On T4 GPU:** ~12-18 hours

## Expected Performance

Based on similar fine-tuning tasks, you should expect:

- **Quality Scoring:** 85-90% accuracy within ±10 points
- **CSAT Prediction:** 80-85% accuracy within ±15 points
- **Issue Classification:** 90-95% accuracy
- **Churn Risk:** 75-80% accuracy
- **Journey Type:** 85-90% accuracy
- **Coaching:** 80-85% relevance score

## After Fine-Tuning

### Inference Speed Comparison

| Approach | Latency | Cost per 1K calls |
|----------|---------|-------------------|
| GPT-4 Prompting | 5-8 seconds | $40 |
| Claude Sonnet 3.5 | 3-5 seconds | $24 |
| **Fine-tuned Mistral 7B** | **0.5-1 second** | **$0.50** |

**80x cheaper, 5-10x faster**

### Deployment

See `../inference/` directory for:
- FastAPI inference server
- Docker deployment
- Replicate deployment
- Modal serverless deployment

## Next Steps

1. Choose your fine-tuning platform (HuggingFace/Replicate/Modal)
2. Review training config in `configs/`
3. Run training script from `scripts/`
4. Evaluate on test set
5. Deploy to production (see `../inference/`)

## Cost Estimate

**Full training pipeline:**
- Data upload/storage: $0-5
- Training (3 epochs): $7-20
- Model storage: $0-10/month
- **Total to production:** $15-35

**Production inference:**
- Self-hosted (T4 GPU): ~$0.35/hour = $250/month
- Serverless (Modal/Replicate): Pay per call, ~$0.0005/call

## Support

For questions or issues:
1. Check `scripts/` for example training code
2. Review `configs/` for hyperparameter templates
3. See HuggingFace docs: https://huggingface.co/docs/transformers/training
