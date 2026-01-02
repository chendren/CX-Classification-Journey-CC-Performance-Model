#!/usr/bin/env python3
"""
Training Data Preparation for Contact Center Analytics Fine-Tuning

Converts synthetic data into multi-task training format for fine-tuning
smaller, faster models (Mistral 7B, Llama 3 8B, Qwen 7B)

Multi-Task Approach:
1. Quality Scoring (regression + classification)
2. CSAT Prediction (regression)
3. Issue Classification (multi-label classification)
4. Sentiment Analysis (classification)
5. Churn Risk Prediction (regression + classification)
6. Customer Journey Type (classification)
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

class TrainingDataPreparator:
    """
    Prepares multi-task training data from contact center transcripts and analyses.

    This class transforms raw conversation data into structured training examples
    for fine-tuning language models on contact center analytics tasks.

    Attributes:
        transcript_dir: Directory containing raw conversation transcripts
        analysis_dir: Directory containing quality analysis JSON files
        journey_dir: Directory containing customer journey insights
        output_dir: Directory where training data will be saved
        train_ratio: Proportion of data for training (0.8 = 80%)
        val_ratio: Proportion of data for validation (0.1 = 10%)
        test_ratio: Proportion of data for testing (0.1 = 10%)
    """

    def __init__(self):
        """
        Initialize data preparation paths and split ratios.

        Note:
            Creates output_dir if it doesn't exist.
            Paths are hardcoded - modify source to change locations.
        """
        # Input directories containing source data
        self.transcript_dir = Path("/Users/chadhendren/contact-center-analytics/transcripts")
        self.analysis_dir = Path("/Users/chadhendren/contact-center-analytics/analyses")
        self.journey_dir = Path("/Users/chadhendren/contact-center-analytics/journey_insights")

        # Output directory for prepared training data
        self.output_dir = Path("/Users/chadhendren/contact-center-analytics/fine-tuning/data")

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training split ratios (should sum to 1.0)
        self.train_ratio = 0.8   # 80% for training
        self.val_ratio = 0.1     # 10% for validation (hyperparameter tuning)
        self.test_ratio = 0.1    # 10% for final evaluation

    def extract_conversation_only(self, transcript_path):
        """
        Extract clean conversation text from transcript markdown file.

        Transcript files may contain metadata and formatting. This function
        extracts just the dialogue portion for use in training examples.

        Args:
            transcript_path: Path to transcript markdown file

        Returns:
            str: Cleaned conversation text with dialogue turns

        Note:
            Handles two formats:
            1. Markdown with "## TRANSCRIPT" section header
            2. Plain format with timestamped dialogue lines like "[00:01:23]: Hello"
        """
        with open(transcript_path, 'r') as f:
            content = f.read()

        # Try to extract from markdown section
        if "## TRANSCRIPT" in content:
            parts = content.split("## TRANSCRIPT")
            if len(parts) > 1:
                # Get content after "## TRANSCRIPT" header, before next "##" section
                conversation = parts[1].split("##")[0]
                return conversation.strip()

        # Fallback: extract lines that look like dialogue
        # Format: [timestamp]: speaker message
        lines = content.split('\n')
        conv_lines = [line for line in lines if line.strip().startswith('[') and ']:' in line]
        return '\n'.join(conv_lines) if conv_lines else content

    def create_quality_scoring_task(self, transcript_text, analysis):
        """
        Create training example for quality scoring task.

        Teaches the model to evaluate conversation quality and provide ratings.

        Args:
            transcript_text: Raw conversation dialogue
            analysis: Dictionary containing quality analysis with keys:
                     - quality_scores.overall_quality: numeric score 0-100
                     - quality_scores.effectiveness_rating: text rating
                     - key_strengths: list of identified strengths
                     - critical_weaknesses: list of identified weaknesses

        Returns:
            dict: Training example with 'prompt' and 'completion' keys

        Example Output Format:
            {
                "quality_score": 85,
                "effectiveness_rating": "good",
                "justification": "Based on 3 strengths and 1 weaknesses identified."
            }
        """
        # Extract quality metrics from analysis
        quality_score = analysis.get("quality_scores", {}).get("overall_quality", 0)
        rating = analysis.get("quality_scores", {}).get("effectiveness_rating", "")

        # Create instructional prompt
        prompt = f"""Analyze this contact center conversation and provide a quality score.

Conversation:
{transcript_text}

Provide:
1. Overall quality score (0-100)
2. Effectiveness rating (exceptional/good/adequate/needs_improvement/poor)
3. Brief justification (2-3 sentences)

Format as JSON."""

        # Create expected response in structured JSON format
        response = json.dumps({
            "quality_score": quality_score,
            "effectiveness_rating": rating,
            "justification": f"Based on {len(analysis.get('key_strengths', []))} strengths and {len(analysis.get('critical_weaknesses', []))} weaknesses identified."
        })

        return {"prompt": prompt, "completion": response}

    def create_csat_prediction_task(self, transcript_text, analysis):
        """
        Create training example for CSAT (Customer Satisfaction) prediction.

        Teaches the model to predict customer satisfaction based on conversation analysis.

        Args:
            transcript_text: Raw conversation dialogue
            analysis: Dictionary with customer_journey.satisfaction_indicators containing:
                     - predicted_csat: numeric score 0-100
                     - satisfaction_signals: list of positive/negative indicators

        Returns:
            dict: Training example with 'prompt' and 'completion' keys

        Note:
            Confidence is derived heuristically: scores at extremes (>80 or <30)
            indicate high confidence, moderate scores indicate medium confidence.
        """
        # Extract CSAT score from nested analysis structure
        csat = analysis.get("customer_journey", {}).get("satisfaction_indicators", {}).get("predicted_csat", 0)

        prompt = f"""Predict the customer satisfaction (CSAT) score for this conversation.

Conversation:
{transcript_text}

Predict CSAT score (0-100) and explain key factors."""

        # Build structured response with confidence assessment
        response = json.dumps({
            "predicted_csat": csat,
            # Extreme scores (very high/low) have higher confidence
            "confidence": "high" if csat > 80 or csat < 30 else "medium",
            # Include top 3 satisfaction signals as supporting evidence
            "key_factors": analysis.get("customer_journey", {}).get("satisfaction_indicators", {}).get("satisfaction_signals", [])[:3]
        })

        return {"prompt": prompt, "completion": response}

    def create_issue_classification_task(self, transcript_text, analysis):
        """Task 3: Classify interaction issues"""
        issue_category = analysis.get("issue_category", "")

        prompt = f"""Classify the primary issue category for this contact center interaction.

Conversation:
{transcript_text}

Identify: primary issue category and sub-categories."""

        response = json.dumps({
            "primary_category": issue_category,
            "detected_issues": analysis.get("detected_issues", [])[:3],
            "complexity": analysis.get("interaction_complexity", "medium")
        })

        return {"prompt": prompt, "completion": response}

    def create_churn_risk_task(self, transcript_text, journey):
        """Task 4: Predict churn risk from journey insights"""
        if not journey:
            return None

        churn_score = journey.get("value_signals", {}).get("churn_risk_score", 0)

        prompt = f"""Analyze this customer conversation and predict churn risk.

Conversation:
{transcript_text}

Predict:
1. Churn risk score (0-100)
2. Risk level (low/medium/high/critical)
3. Top 3 risk factors"""

        risk_factors = journey.get("value_signals", {}).get("churn_risk_factors", [])
        # Handle case where risk_factors might be a list of dicts or other structure
        if isinstance(risk_factors, list):
            factors = [f.get("factor", "") if isinstance(f, dict) else str(f) for f in risk_factors[:3]]
        else:
            factors = []

        response = json.dumps({
            "churn_risk_score": churn_score,
            "risk_level": "critical" if churn_score > 75 else "high" if churn_score > 50 else "medium" if churn_score > 25 else "low",
            "risk_factors": factors
        })

        return {"prompt": prompt, "completion": response}

    def create_journey_type_task(self, transcript_text, journey):
        """Task 5: Classify customer journey type"""
        if not journey:
            return None

        journey_type = journey.get("journey_metadata", {}).get("journey_type", "")

        prompt = f"""Classify the customer journey type for this interaction.

Conversation:
{transcript_text}

Identify journey type and key characteristics."""

        # journey_stages is a dict with stage names as keys (awareness, consideration, decision, retention)
        journey_stages = journey.get("journey_stages", {})
        if isinstance(journey_stages, dict):
            # Extract stage names from dict keys where present=True
            stages_list = [stage_name for stage_name, stage_data in journey_stages.items()
                          if isinstance(stage_data, dict) and stage_data.get("present", False)]
        else:
            stages_list = []

        response = json.dumps({
            "journey_type": journey_type,
            "journey_stages": stages_list,
            "sentiment_trajectory": journey.get("sentiment_journey", {}).get("sentiment_trajectory", "")
        })

        return {"prompt": prompt, "completion": response}

    def create_coaching_recommendations_task(self, transcript_text, analysis):
        """Task 6: Generate coaching recommendations"""
        coaching = analysis.get("coaching_recommendations", {})

        prompt = f"""Analyze this agent conversation and provide coaching recommendations.

Conversation:
{transcript_text}

Provide:
1. Immediate coaching priorities
2. Long-term development areas
3. Specific examples from the conversation"""

        response = json.dumps({
            "immediate_priorities": coaching.get("immediate", [])[:3],
            "long_term": coaching.get("long_term", [])[:2],
            "strengths_to_leverage": analysis.get("key_strengths", [])[:2]
        })

        return {"prompt": prompt, "completion": response}

    def prepare_all_tasks(self):
        """
        Main orchestration method that prepares all training examples.

        Process:
        1. Iterates through all analysis files
        2. Loads corresponding transcripts and journey insights
        3. Creates training examples for each task type
        4. Splits data into train/validation/test sets
        5. Saves in multiple formats (JSONL, ChatML)

        The method creates 6 types of training tasks:
        - Quality scoring: Overall conversation quality assessment
        - CSAT prediction: Customer satisfaction forecasting
        - Issue classification: Problem categorization
        - Churn risk: Customer retention risk assessment
        - Journey type: Customer journey stage identification
        - Coaching: Agent improvement recommendations

        Note:
            Progress is printed every 500 files to avoid excessive output.
            Errors in individual files are logged but don't stop processing.
        """
        print("=" * 80)
        print("TRAINING DATA PREPARATION - Multi-Task Contact Center Analytics")
        print("=" * 80)

        # Find all analysis files (these drive the data preparation)
        # Each analysis file corresponds to one transcript
        analyses = sorted(self.analysis_dir.glob("analysis_*.json"))

        all_examples = []  # Will hold all training examples
        # Track how many examples created per task type
        tasks_created = {
            "quality_scoring": 0,
            "csat_prediction": 0,
            "issue_classification": 0,
            "churn_risk": 0,
            "journey_type": 0,
            "coaching": 0
        }

        print(f"\nProcessing {len(analyses)} analysis files...")

        for i, analysis_file in enumerate(analyses, 1):
            try:
                # Extract numeric ID from filename (e.g., "analysis_00042.json" -> 42)
                num = int(analysis_file.stem.split('_')[1])

                # Construct paths to matching transcript and journey files
                # All files with same number should correspond to same conversation
                transcript_file = self.transcript_dir / f"transcript_{num:05d}.md"
                journey_file = self.journey_dir / f"journey_insight_{num:05d}.json"

                # Skip if transcript doesn't exist (analysis without source data)
                if not transcript_file.exists():
                    continue

                # Extract clean conversation text
                conversation = self.extract_conversation_only(transcript_file)

                # Load analysis JSON (handle potential double-encoding)
                # Some files may have JSON-encoded strings instead of objects
                with open(analysis_file, 'r') as f:
                    analysis = json.load(f)
                    # If analysis is a string, parse it again
                    if isinstance(analysis, str):
                        analysis = json.loads(analysis)

                # Load journey insights if available
                journey = None
                if journey_file.exists():
                    with open(journey_file, 'r') as f:
                        journey = json.load(f)

                # Create training examples for each task
                try:
                    tasks = [
                        ("quality_scoring", self.create_quality_scoring_task(conversation, analysis)),
                        ("csat_prediction", self.create_csat_prediction_task(conversation, analysis)),
                        ("issue_classification", self.create_issue_classification_task(conversation, analysis)),
                        ("coaching", self.create_coaching_recommendations_task(conversation, analysis))
                    ]
                except AttributeError as e:
                    print(f"  Error in task creation for {analysis_file.name}:")
                    print(f"    conversation type: {type(conversation)}")
                    print(f"    analysis type: {type(analysis)}")
                    print(f"    error: {e}")
                    continue

                # Add journey-based tasks if available
                if journey:
                    tasks.extend([
                        ("churn_risk", self.create_churn_risk_task(conversation, journey)),
                        ("journey_type", self.create_journey_type_task(conversation, journey))
                    ])

                # Add examples
                for task_name, task_data in tasks:
                    if task_data:
                        task_data["task_type"] = task_name
                        task_data["source_id"] = f"transcript_{num:05d}"
                        all_examples.append(task_data)
                        tasks_created[task_name] += 1

                if i % 500 == 0:
                    print(f"  Progress: {i}/{len(analyses)} ({i/len(analyses)*100:.1f}%)")

            except Exception as e:
                import traceback
                print(f"  Error processing {analysis_file.name}: {e}")
                print(f"    {traceback.format_exc()}")
                continue

        print(f"\n✅ Created {len(all_examples):,} training examples")
        print("\nExamples per task:")
        for task, count in tasks_created.items():
            print(f"  {task}: {count:,}")

        # Shuffle examples
        random.shuffle(all_examples)

        # Split into train/val/test
        n = len(all_examples)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)

        train_examples = all_examples[:train_end]
        val_examples = all_examples[train_end:val_end]
        test_examples = all_examples[val_end:]

        print(f"\nDataset split:")
        print(f"  Training:   {len(train_examples):,} ({len(train_examples)/n*100:.1f}%)")
        print(f"  Validation: {len(val_examples):,} ({len(val_examples)/n*100:.1f}%)")
        print(f"  Test:       {len(test_examples):,} ({len(test_examples)/n*100:.1f}%)")

        # Save datasets
        self.save_dataset(train_examples, "train")
        self.save_dataset(val_examples, "validation")
        self.save_dataset(test_examples, "test")

        # Create metadata file
        metadata = {
            "created_at": datetime.now().isoformat(),
            "total_examples": len(all_examples),
            "tasks": tasks_created,
            "splits": {
                "train": len(train_examples),
                "validation": len(val_examples),
                "test": len(test_examples)
            },
            "source": "contact-center-analytics synthetic data",
            "recommended_models": [
                "mistralai/Mistral-7B-Instruct-v0.3",
                "meta-llama/Llama-3-8B-Instruct",
                "Qwen/Qwen2.5-7B-Instruct"
            ]
        }

        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print("\n✅ Training data preparation complete!")
        print(f"   Output directory: {self.output_dir}")
        print("=" * 80)

    def save_dataset(self, examples, split_name):
        """Save dataset in multiple formats"""
        # JSONL format (for HuggingFace/Replicate)
        jsonl_file = self.output_dir / f"{split_name}.jsonl"
        with open(jsonl_file, 'w') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')

        # ChatML format (for instruction tuning)
        chatml_file = self.output_dir / f"{split_name}_chatml.jsonl"
        with open(chatml_file, 'w') as f:
            for example in examples:
                chatml = {
                    "messages": [
                        {"role": "user", "content": example["prompt"]},
                        {"role": "assistant", "content": example["completion"]}
                    ],
                    "task_type": example["task_type"]
                }
                f.write(json.dumps(chatml) + '\n')

        print(f"  Saved {len(examples):,} examples to {split_name}.jsonl")

def main():
    preparator = TrainingDataPreparator()
    preparator.prepare_all_tasks()

if __name__ == "__main__":
    main()
