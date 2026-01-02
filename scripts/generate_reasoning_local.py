#!/usr/bin/env python3
"""
Local Reasoning Generation using HuggingFace Transformers
Uses DeepSeek R1 or Qwen2.5 for high-quality reasoning chains
Optimized for M4 Max with proper tokenization
"""

import os
import json
import torch
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

class LocalReasoningGenerator:
    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"):
        """Initialize local reasoning model with transformers"""
        print(f"Loading {model_name}...")
        print("This may take a few minutes on first run (14B model, ~28GB)...")

        # Setup device
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("✅ Using Apple Silicon GPU (MPS)")
        else:
            self.device = "cpu"
            print("⚠️  Using CPU (slower)")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Load model in 16-bit for M4 Max
        # For MPS, we need to load without device_map and manually move to device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        # Move model to MPS device
        self.model = self.model.to(self.device)

        print(f"✅ Model loaded successfully on {self.device}")

    def create_reasoning_prompt(self, transcript: str, analysis: Dict) -> str:
        """Create detailed reasoning prompt"""

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
        """Generate reasoning chain"""

        prompt = self.create_reasoning_prompt(transcript, analysis)

        try:
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096
            ).to(self.device)

            # Generate with proper sampling
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode with proper spacing
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )

            return response.strip()

        except Exception as e:
            print(f"Error generating reasoning: {e}")
            return ""

    def validate_quality(self, reasoning: str) -> tuple[bool, float, List[str]]:
        """Validate reasoning quality"""
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

        is_valid = score >= 70.0
        return is_valid, max(0, score), issues

    def enhance_example(self, example: Dict, retry_on_failure: bool = True) -> Optional[Dict]:
        """Enhance a single training example with reasoning"""

        messages = example.get('messages', [])
        if not messages or len(messages) < 2:
            return None

        user_message = messages[0]['content']
        assistant_message = messages[1]['content']

        # Extract transcript
        transcript = user_message.replace('[INST]', '').replace('[/INST]', '').strip()

        # Parse existing analysis
        try:
            existing_analysis = json.loads(assistant_message)
        except:
            existing_analysis = {"response": assistant_message}

        # Generate reasoning
        print(f"  Generating reasoning (transcript length: {len(transcript)} chars)...")
        reasoning = self.generate_reasoning(transcript, existing_analysis)

        if not reasoning:
            return None

        # Validate quality
        is_valid, quality_score, issues = self.validate_quality(reasoning)

        print(f"  Quality score: {quality_score:.1f}/100 {'✅' if is_valid else '❌'}")
        if issues:
            print(f"  Issues: {', '.join(issues)}")

        # Retry once if quality is poor
        if not is_valid and retry_on_failure:
            print(f"  Retrying with higher max_tokens...")
            reasoning = self.generate_reasoning(transcript, existing_analysis, max_tokens=2500)
            is_valid, quality_score, issues = self.validate_quality(reasoning)
            print(f"  Retry quality score: {quality_score:.1f}/100 {'✅' if is_valid else '❌'}")

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


def main():
    parser = argparse.ArgumentParser(description="Generate reasoning with local model")
    parser.add_argument('--input', type=str, required=True, help='Input training file')
    parser.add_argument('--output', type=str, required=True, help='Output file')
    parser.add_argument('--model', type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                       help='Model to use (default: DeepSeek-R1-Distill-Qwen-14B)')
    parser.add_argument('--max-examples', type=int, help='Max examples (for testing)')
    parser.add_argument('--test', action='store_true', help='Test mode (1 example)')

    args = parser.parse_args()

    if args.test:
        args.max_examples = 1
        print("🧪 TEST MODE: Processing 1 example\n")

    # Load examples
    with open(args.input, 'r') as f:
        examples = [json.loads(line) for line in f]

    if args.max_examples:
        examples = examples[:args.max_examples]

    print(f"Processing {len(examples)} examples...")
    print(f"Using model: {args.model}\n")

    # Initialize generator
    generator = LocalReasoningGenerator(args.model)

    # Process examples
    results = []
    for i, example in enumerate(tqdm(examples, desc="Generating reasoning")):
        print(f"\nExample {i+1}/{len(examples)}:")
        enhanced = generator.enhance_example(example)
        if enhanced:
            results.append(enhanced)

    # Save results
    with open(args.output, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    # Quality report
    quality_scores = [r.get('quality_score', 0) for r in results]
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

    print(f"\n{'='*80}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"✅ Total examples: {len(results)}")
    print(f"📊 Average quality score: {avg_quality:.1f}/100")
    print(f"🎯 High quality (>80): {sum(1 for s in quality_scores if s > 80)}")
    print(f"⚠️  Medium quality (60-80): {sum(1 for s in quality_scores if 60 <= s <= 80)}")
    print(f"❌ Low quality (<60): {sum(1 for s in quality_scores if s < 60)}")
    print(f"💾 Saved to: {args.output}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
