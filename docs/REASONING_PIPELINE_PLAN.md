# Contact Center Reasoning Model - Complete Pipeline Plan

## Overview
Train a comprehensive reasoning-capable model for contact center analytics using locally generated reasoning chains.

## Current Status
- ✅ Training data prepared: 25,790 examples (6 specialized tasks)
- ✅ HuggingFace transformers reasoning generator created
- ⏳ Qwen2.5-7B-Instruct downloading (~14GB)
- ⏳ Test generation pending
- ⏳ Full reasoning generation pending

## Pipeline Steps

### Phase 1: Reasoning Data Generation (Current)
**Goal:** Generate 25,790 high-quality reasoning chains using Qwen2.5-7B-Instruct

**Process:**
1. Download Qwen2.5-7B-Instruct (one-time, ~14GB)
2. Test with 1 example to validate quality
3. Generate reasoning for all 25,790 training examples

**Reasoning Structure:**
```
<thinking>
Step 1 - Customer Sentiment Analysis
Step 2 - Agent Performance Evaluation
Step 3 - Interaction Quality Assessment
Step 4 - Root Cause Analysis
Step 5 - Business Impact Analysis
</thinking>

<analysis>
Quality Metrics, Key Strengths, Critical Issues, Patterns & Trends
</analysis>

<insights>
Immediate Coaching, Process Improvements, Business Actions, Strategic Insights
</insights>
```

**Quality Validation:**
- Required sections: <thinking>, <analysis>, <insights>
- Minimum length: 500 characters
- Step-by-step structure required
- Explicit reasoning with "because" statements
- Actionable recommendations
- Business impact analysis
- Quality threshold: 70/100 score
- Auto-retry on poor quality

**Estimated Time:**
- Sequential: 36-72 hours (depends on generation speed)
- Per example: ~5-10 seconds
- Total: 25,790 examples

**Output:**
- File: `data/train_reasoning.jsonl`
- Format: ChatML with reasoning-enhanced responses
- Quality: Validated, high-scoring reasoning chains

### Phase 2: Model Training
**Goal:** Train Mistral-7B or Qwen2.5-7B with reasoning-enhanced data

**Model Options:**
- **Option A:** Mistral-7B-v0.1 (already downloaded, 13GB)
- **Option B:** Qwen2.5-7B-Instruct (downloading, 14GB)

**Training Configuration:**
- LoRA fine-tuning (parameter efficient)
- 16-bit precision (FP16)
- Apple Silicon MPS backend
- Batch size: 2 per device
- Gradient accumulation: 8 steps
- Effective batch size: 16
- Learning rate: 2e-5
- Epochs: 3
- Max length: 2048 tokens

**Estimated Training Time:**
- With LoRA: 12-24 hours on M4 Max
- Full fine-tune: 2-4 days (not recommended)

### Phase 3: Evaluation & Deployment
**Goal:** Validate model quality and prepare for inference

**Evaluation Metrics:**
- Reasoning structure compliance
- Quality score distribution
- Task-specific accuracy (CSAT, quality, churn prediction)
- Response coherence and actionability

**Deployment:**
- Local inference with MPS acceleration
- Batch processing capability
- API integration ready

## File Locations

### Scripts
- `/Users/chadhendren/contact-center-analytics/fine-tuning/scripts/generate_reasoning_local.py` - Reasoning generator
- `/Users/chadhendren/contact-center-analytics/fine-tuning/scripts/train_mac.py` - Training script

### Data
- `/Users/chadhendren/contact-center-analytics/fine-tuning/data/train_chatml.jsonl` - Original training data (25,790)
- `/Users/chadhendren/contact-center-analytics/fine-tuning/data/train_reasoning.jsonl` - Reasoning-enhanced (pending)
- `/Users/chadhendren/contact-center-analytics/fine-tuning/data/validation_chatml.jsonl` - Validation (3,223)
- `/Users/chadhendren/contact-center-analytics/fine-tuning/data/test_chatml.jsonl` - Test (3,225)

### Models
- `~/.cache/huggingface/hub/` - Downloaded models cache
- `/Users/chadhendren/contact-center-analytics/models/` - Trained models output

### Configs
- `/Users/chadhendren/contact-center-analytics/fine-tuning/configs/mistral_7b_mac.yaml` - Training config

## Next Actions (Auto-Execute)

1. **Wait for Qwen2.5 download to complete** ⏳
2. **Validate test generation quality** → If quality score ≥75, proceed
3. **Launch full reasoning generation** → Background process, ~36-72 hours
4. **Monitor progress** → Check quality distribution
5. **Start model training** → Once reasoning data ready
6. **Evaluate trained model** → Test set performance
7. **Deploy for inference** → Production-ready model

## Success Criteria

### Reasoning Data Quality
- ✅ Average quality score ≥75/100
- ✅ >80% examples with score ≥70
- ✅ All required sections present
- ✅ Structured step-by-step reasoning
- ✅ Actionable business insights

### Trained Model Quality
- ✅ Generates structured reasoning chains
- ✅ Maintains domain knowledge
- ✅ Provides actionable recommendations
- ✅ Accurate quality/CSAT predictions
- ✅ Fast inference (M4 Max optimized)

## Timeline Estimate

| Phase | Duration | Status |
|-------|----------|--------|
| Qwen download & test | 30-60 min | In Progress |
| Reasoning generation | 36-72 hours | Pending |
| Model training | 12-24 hours | Pending |
| Evaluation | 2-4 hours | Pending |
| **Total** | **2-4 days** | **~50% Complete** |

## Resource Requirements

### Disk Space
- Qwen2.5-7B model: ~14GB
- Reasoning dataset: ~500MB-1GB
- Training checkpoints: ~30-40GB
- Final model: ~13-14GB
- **Total:** ~60-70GB

### Memory (M4 Max, 64GB RAM)
- Model loading: ~16-18GB
- Training: ~30-40GB peak
- Inference: ~16-18GB
- **Recommended:** 64GB RAM (✅ Available)

### Compute
- Generation: CPU/GPU hybrid, ~36-72 hours
- Training: MPS (GPU), ~12-24 hours
- Total: ~2-4 days wall time

## Notes

- Qwen2.5-7B chosen for superior reasoning capabilities
- HuggingFace transformers ensures proper tokenization
- Quality validation prevents low-quality data
- LoRA training is parameter-efficient (0.58% trainable)
- M4 Max optimization for Apple Silicon
- Checkpoint saving for resume capability
