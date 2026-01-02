#!/usr/bin/env python3
"""
Generate Reasoning-Enhanced Training Data for Contact Center Analytics

Uses Claude API to add step-by-step reasoning chains to existing training examples.
Creates a reasoning model that can explain its analysis and provide actionable insights.
"""

import os
import json
import anthropic
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

class ReasoningChainGenerator:
    def __init__(self, api_key: str = None):
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.model = "claude-3-5-sonnet-20241022"

    def generate_reasoning_chain(self, transcript: str, analysis: Dict) -> str:
        """Generate a detailed reasoning chain for a contact center analysis"""

        prompt = f"""You are an expert contact center analyst. Analyze this transcript and provide detailed reasoning leading to insights.

TRANSCRIPT:
{transcript}

EXPECTED ANALYSIS:
{json.dumps(analysis, indent=2)}

Please provide a detailed analysis with this structure:

<thinking>
1. Customer Sentiment Analysis:
   - Identify emotional indicators in customer language
   - Track sentiment evolution through conversation
   - Note specific trigger phrases or events

2. Agent Performance Evaluation:
   - Assess communication effectiveness
   - Identify strengths (empathy, knowledge, resolution)
   - Note weaknesses or missed opportunities

3. Interaction Quality Assessment:
   - Evaluate first-call resolution potential
   - Assess efficiency (hold times, transfers, resolution time)
   - Check compliance and process adherence

4. Root Cause Analysis:
   - Identify systemic vs. individual issues
   - Determine if issue is agent knowledge, process, or policy-related
   - Assess whether similar issues are likely recurring
</thinking>

<analysis>
Based on the above reasoning:
- Quality Score: [score with justification]
- CSAT Prediction: [score with rationale]
- Key Strengths: [specific examples from transcript]
- Areas for Improvement: [specific, actionable items]
- Patterns Identified: [broader trends this suggests]
</analysis>

<insights>
Coaching Recommendations:
1. [Specific coaching point with transcript reference]
2. [Process improvement suggestion]
3. [Training need identified]

Business Impact:
- Customer Satisfaction Risk: [High/Medium/Low with reasoning]
- Operational Efficiency: [Assessment with specific metrics]
- Recommended Actions: [Immediate and long-term steps]
</insights>

Provide your analysis now:"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            print(f"Error generating reasoning chain: {e}")
            return ""

    def enhance_training_example(self, example: Dict) -> Dict:
        """Enhance a single training example with reasoning"""

        # Parse the original example
        messages = example.get('messages', [])
        if not messages or len(messages) < 2:
            return example

        user_message = messages[0]['content']
        assistant_message = messages[1]['content']

        # Extract transcript from user message
        # Assuming format: "[INST] Analyze this transcript: <transcript> [/INST]"
        transcript = user_message.split('[/INST]')[0].replace('[INST]', '').strip()

        # Try to parse existing analysis
        try:
            existing_analysis = json.loads(assistant_message)
        except:
            existing_analysis = {"response": assistant_message}

        # Generate reasoning chain
        reasoning_output = self.generate_reasoning_chain(transcript, existing_analysis)

        if not reasoning_output:
            return example

        # Create enhanced example with reasoning
        enhanced_messages = [
            {
                "role": "user",
                "content": user_message
            },
            {
                "role": "assistant",
                "content": reasoning_output
            }
        ]

        return {
            "messages": enhanced_messages,
            "original_response": assistant_message,
            "reasoning_enhanced": True
        }

    def process_training_file(self, input_file: str, output_file: str, max_examples: int = None):
        """Process entire training file and add reasoning chains"""

        print(f"\nProcessing: {input_file}")
        print(f"Output: {output_file}")

        # Load existing data
        with open(input_file, 'r') as f:
            examples = [json.loads(line) for line in f]

        if max_examples:
            examples = examples[:max_examples]

        print(f"Loaded {len(examples)} examples")

        # Process each example
        enhanced_examples = []
        with tqdm(total=len(examples), desc="Generating reasoning chains") as pbar:
            for example in examples:
                enhanced = self.enhance_training_example(example)
                enhanced_examples.append(enhanced)
                pbar.update(1)

        # Save enhanced data
        with open(output_file, 'w') as f:
            for example in enhanced_examples:
                f.write(json.dumps(example) + '\n')

        print(f"\n✅ Saved {len(enhanced_examples)} enhanced examples to {output_file}")

        # Calculate stats
        reasoning_count = sum(1 for ex in enhanced_examples if ex.get('reasoning_enhanced'))
        print(f"   {reasoning_count} examples enhanced with reasoning ({100*reasoning_count/len(enhanced_examples):.1f}%)")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate reasoning-enhanced training data")
    parser.add_argument('--input', type=str, required=True, help='Input training file (JSONL)')
    parser.add_argument('--output', type=str, required=True, help='Output file for enhanced data')
    parser.add_argument('--max-examples', type=int, help='Maximum examples to process (for testing)')
    parser.add_argument('--api-key', type=str, help='Claude API key (or set ANTHROPIC_API_KEY)')

    args = parser.parse_args()

    # Verify API key
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ Error: ANTHROPIC_API_KEY not set")
        print("   Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        return

    generator = ReasoningChainGenerator(api_key=api_key)
    generator.process_training_file(args.input, args.output, args.max_examples)


if __name__ == "__main__":
    main()
