#!/usr/bin/env python3
"""
Generate Reasoning-Enhanced Training Data using Local DeepSeek R1

Uses DeepSeek-R1-Distill-Llama-8B running locally with MLX for fast, parallel reasoning generation.
Ensures high quality through validation and multi-pass generation.

Features:
- Parallel processing with multiple workers
- Quality validation and scoring
- Resume capability (checkpoint progress)
- MLX-optimized for M4 Max
"""

import os
import json
import mlx.core as mx
from mlx_lm import load, generate
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

class R1ReasoningGenerator:
    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"):
        """Initialize R1 model with MLX"""
        print(f"Loading {model_name}...")
        self.model, self.tokenizer = load(model_name)
        print("✅ Model loaded successfully")

    def create_reasoning_prompt(self, transcript: str, analysis: Dict) -> str:
        """Create a detailed prompt for R1 to generate reasoning chains"""

        prompt = f"""You are an expert contact center quality analyst. Analyze this customer service interaction and provide detailed, step-by-step reasoning leading to actionable insights.

TRANSCRIPT:
{transcript}

ANALYSIS CONTEXT:
{json.dumps(analysis, indent=2)}

Provide a comprehensive analysis following this structure:

<thinking>
Step 1 - Customer Sentiment Analysis:
- Identify specific emotional indicators in customer's language
- Track how sentiment evolves through the conversation
- Note trigger phrases that changed emotional state
- Assess overall customer satisfaction trajectory

Step 2 - Agent Performance Evaluation:
- Communication effectiveness (clarity, empathy, professionalism)
- Technical knowledge and problem-solving ability
- Adherence to best practices and protocols
- Missed opportunities or areas for improvement

Step 3 - Interaction Quality Assessment:
- First-call resolution capability
- Efficiency metrics (hold times, transfers, overall duration)
- Process adherence and compliance
- Customer effort score indicators

Step 4 - Root Cause Analysis:
- Distinguish between agent skill gaps vs. systemic issues
- Identify process or policy problems
- Determine if this indicates broader trends
- Assess prevention opportunities

Step 5 - Business Impact Analysis:
- Customer retention risk assessment
- Operational efficiency implications
- Revenue impact potential
- Brand perception effects
</thinking>

<analysis>
Based on systematic evaluation:

Quality Metrics:
- Overall Quality Score: [X/100] because [specific reasons with transcript references]
- CSAT Prediction: [X/5] due to [emotional markers and resolution quality]
- First Call Resolution: [Achieved/Not Achieved] - [explanation]
- Customer Effort: [High/Medium/Low] - [justification]

Key Strengths Identified:
1. [Specific strength with transcript example]
2. [Specific strength with transcript example]
3. [Specific strength with transcript example]

Critical Issues:
1. [Issue with severity level and transcript reference]
2. [Issue with severity level and transcript reference]
3. [Issue with severity level and transcript reference]

Patterns & Trends:
- [Pattern observed that suggests broader implications]
- [Connection to operational metrics or common issues]
</analysis>

<insights>
Immediate Coaching Recommendations:
1. [Specific, actionable coaching point targeting identified gap]
   - What to improve: [specific behavior]
   - How to improve: [concrete technique]
   - Expected outcome: [measurable result]

2. [Second coaching recommendation following same structure]

3. [Third coaching recommendation following same structure]

Process Improvements:
- [Systemic fix that would prevent this issue]
- [Policy or workflow enhancement]
- [Training program suggestion]

Business Actions:
- Priority Level: [High/Medium/Low]
- Immediate Action Required: [Yes/No - what action]
- Follow-up Needed: [customer callback, escalation, etc.]
- Monitoring Recommendation: [what metrics to track]

Long-term Strategic Insights:
- Customer experience trend: [what this reveals]
- Operational efficiency opportunity: [how to optimize]
- Training investment needed: [specific skills or knowledge]
</insights>

Provide your detailed analysis now:"""

        return prompt

    def generate_reasoning(self, transcript: str, analysis: Dict, max_tokens: int = 2000) -> str:
        """Generate reasoning chain using R1"""

        prompt = self.create_reasoning_prompt(transcript, analysis)

        try:
            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=False
            )
            return response
        except Exception as e:
            print(f"Error generating reasoning: {e}")
            return ""

    def validate_quality(self, reasoning: str) -> tuple[bool, float, List[str]]:
        """
        Validate reasoning quality
        Returns: (is_valid, quality_score, issues)
        """
        issues = []
        score = 100.0

        # Check for required sections
        required_sections = ['<thinking>', '<analysis>', '<insights>']
        for section in required_sections:
            if section not in reasoning:
                issues.append(f"Missing {section} section")
                score -= 30

        # Check for substantive content
        if len(reasoning) < 500:
            issues.append("Response too short (< 500 chars)")
            score -= 20

        # Check for step-by-step reasoning
        if 'Step 1' not in reasoning or 'Step 2' not in reasoning:
            issues.append("Missing step-by-step reasoning structure")
            score -= 15

        # Check for specific examples
        if 'because' not in reasoning.lower():
            issues.append("Lacks explicit reasoning with 'because' statements")
            score -= 10

        # Check for actionable insights
        if 'Recommendation' not in reasoning and 'Action' not in reasoning:
            issues.append("Missing actionable recommendations")
            score -= 15

        # Check for business impact
        if 'impact' not in reasoning.lower():
            issues.append("Missing business impact analysis")
            score -= 10

        is_valid = score >= 70.0  # Require 70% quality threshold

        return is_valid, max(0, score), issues

    def enhance_example(self, example: Dict, retry_on_failure: bool = True) -> Dict:
        """Enhance a single training example with R1 reasoning"""

        # Parse the original example
        messages = example.get('messages', [])
        if not messages or len(messages) < 2:
            return None

        user_message = messages[0]['content']
        assistant_message = messages[1]['content']

        # Extract transcript (assuming ChatML format)
        transcript = user_message.replace('[INST]', '').replace('[/INST]', '').strip()

        # Try to parse existing analysis
        try:
            existing_analysis = json.loads(assistant_message)
        except:
            existing_analysis = {"response": assistant_message}

        # Generate reasoning with R1
        reasoning = self.generate_reasoning(transcript, existing_analysis)

        if not reasoning:
            return None

        # Validate quality
        is_valid, quality_score, issues = self.validate_quality(reasoning)

        # Retry once if quality is poor
        if not is_valid and retry_on_failure:
            reasoning = self.generate_reasoning(transcript, existing_analysis, max_tokens=2500)
            is_valid, quality_score, issues = self.validate_quality(reasoning)

        # Create enhanced example
        enhanced = {
            "messages": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": reasoning}
            ],
            "original_response": assistant_message,
            "reasoning_enhanced": True,
            "quality_score": quality_score,
            "validation_issues": issues if issues else None
        }

        return enhanced


def process_batch(batch_data):
    """Process a batch of examples (for parallel processing)"""
    examples, model_name, batch_id = batch_data

    # Each worker loads its own model instance
    generator = R1ReasoningGenerator(model_name)

    results = []
    for i, example in enumerate(examples):
        enhanced = generator.enhance_example(example)
        if enhanced:
            results.append(enhanced)

    return results


def parallel_generate(input_file: str, output_file: str, checkpoint_file: str,
                     num_workers: int = 4, batch_size: int = 10,
                     max_examples: Optional[int] = None):
    """
    Generate reasoning chains in parallel with checkpointing
    """

    print(f"\n{'='*80}")
    print(f"R1 REASONING GENERATION - PARALLEL MODE")
    print(f"{'='*80}")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Workers: {num_workers}")
    print(f"Batch size: {batch_size}")
    print(f"{'='*80}\n")

    # Load existing data
    with open(input_file, 'r') as f:
        examples = [json.loads(line) for line in f]

    if max_examples:
        examples = examples[:max_examples]

    # Check for existing checkpoint
    completed = []
    if os.path.exists(checkpoint_file):
        print(f"📁 Found checkpoint: {checkpoint_file}")
        with open(checkpoint_file, 'r') as f:
            completed = [json.loads(line) for line in f]
        print(f"   Resuming from {len(completed)} completed examples\n")
        examples = examples[len(completed):]

    if not examples:
        print("✅ All examples already processed!")
        return

    print(f"Processing {len(examples)} examples...")
    print(f"Estimated time: {len(examples) * 3 / 60 / num_workers:.1f} hours\n")

    # Split into batches for parallel processing
    batches = []
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i+batch_size]
        batches.append((batch, "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", i//batch_size))

    # Process in parallel
    all_results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all batches
        future_to_batch = {executor.submit(process_batch, batch): batch for batch in batches}

        # Collect results with progress bar
        with tqdm(total=len(batches), desc="Processing batches") as pbar:
            for future in as_completed(future_to_batch):
                try:
                    results = future.result()
                    all_results.extend(results)

                    # Checkpoint progress
                    with open(checkpoint_file, 'a') as f:
                        for result in results:
                            f.write(json.dumps(result) + '\n')

                    pbar.update(1)
                    pbar.set_postfix({'completed': len(completed) + len(all_results)})

                except Exception as e:
                    print(f"\n❌ Batch failed: {e}")

    # Combine completed and new results
    final_results = completed + all_results

    # Save final output
    with open(output_file, 'w') as f:
        for result in final_results:
            f.write(json.dumps(result) + '\n')

    # Quality report
    quality_scores = [r.get('quality_score', 0) for r in final_results if 'quality_score' in r]
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

    print(f"\n{'='*80}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"✅ Total examples: {len(final_results)}")
    print(f"📊 Average quality score: {avg_quality:.1f}/100")
    print(f"🎯 High quality (>80): {sum(1 for s in quality_scores if s > 80)}")
    print(f"⚠️  Medium quality (60-80): {sum(1 for s in quality_scores if 60 <= s <= 80)}")
    print(f"❌ Low quality (<60): {sum(1 for s in quality_scores if s < 60)}")
    print(f"💾 Saved to: {output_file}")
    print(f"{'='*80}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate reasoning with local R1")
    parser.add_argument('--input', type=str, required=True, help='Input training file')
    parser.add_argument('--output', type=str, required=True, help='Output file')
    parser.add_argument('--checkpoint', type=str, default='reasoning_checkpoint.jsonl',
                       help='Checkpoint file for resume capability')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--batch-size', type=int, default=10, help='Examples per batch')
    parser.add_argument('--max-examples', type=int, help='Max examples (for testing)')
    parser.add_argument('--test', action='store_true', help='Test mode (10 examples)')

    args = parser.parse_args()

    if args.test:
        args.max_examples = 10
        args.workers = 2
        print("🧪 TEST MODE: Processing 10 examples with 2 workers\n")

    parallel_generate(
        args.input,
        args.output,
        args.checkpoint,
        num_workers=args.workers,
        batch_size=args.batch_size,
        max_examples=args.max_examples
    )


if __name__ == "__main__":
    main()
