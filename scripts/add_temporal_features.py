#!/usr/bin/env python3
"""
Add Temporal Features to Training Data

Enhances training data with time-aware features to enable temporal pattern recognition:
- Time of day (morning/afternoon/evening/night)
- Day of week (Monday-Sunday)
- Business hours indicators
- Holiday/peak season flags
- Queue time patterns
"""

import os
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

class TemporalFeatureEnhancer:
    def __init__(self):
        """Initialize temporal feature generator"""

        # US Federal Holidays 2024-2025 (approximate)
        self.holidays = [
            "2024-01-01", "2024-01-15", "2024-02-19", "2024-05-27",
            "2024-07-04", "2024-09-02", "2024-10-14", "2024-11-11",
            "2024-11-28", "2024-12-25",
            "2025-01-01", "2025-01-20", "2025-02-17", "2025-05-26",
            "2025-07-04", "2025-09-01", "2025-10-13", "2025-11-11",
            "2025-11-27", "2025-12-25"
        ]

        # Peak contact center periods
        self.peak_seasons = {
            "holiday_shopping": ["2024-11-15", "2025-01-05"],
            "tax_season": ["2024-03-01", "2024-04-15", "2025-03-01", "2025-04-15"],
            "back_to_school": ["2024-08-15", "2024-09-15", "2025-08-15", "2025-09-15"],
            "year_end": ["2024-12-15", "2024-12-31", "2025-12-15", "2025-12-31"]
        }

    def generate_realistic_timestamp(self) -> datetime:
        """
        Generate a realistic contact center interaction timestamp.

        Creates timestamps with realistic distribution patterns matching actual
        contact center traffic: higher volume during business hours, lower at night.

        Returns:
            datetime: Random timestamp in 2024 with weighted hour distribution

        Note:
            Hour weights reflect typical contact center patterns:
            - Peak hours: 9am-3pm (weights 20-25)
            - Business hours: 8am-5pm (weights 15-25)
            - Early morning/evening: 6am-8am, 6pm-9pm (weights 5-15)
            - Night hours: 9pm-6am (weights 1-3)

            This ensures the model learns temporal patterns that match real data.
        """

        # Random date in 2024 (uniform distribution across the year)
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 12, 31)

        # Calculate random date
        time_between = end_date - start_date
        days_between = time_between.days
        random_days = random.randrange(days_between)
        random_date = start_date + timedelta(days=random_days)

        # Weighted distribution for contact center hours
        # Weights represent relative probability for each hour of the day
        # Higher weights = more likely to generate timestamps in that hour
        hour_weights = [
            1,  # 00:00 - very rare (overnight)
            1,  # 01:00
            1,  # 02:00
            1,  # 03:00
            1,  # 04:00
            2,  # 05:00 - early birds
            5,  # 06:00 - increasing
            10, # 07:00 - morning ramp-up
            15, # 08:00 - business hours start
            20, # 09:00 - peak begins
            25, # 10:00 - peak hours
            25, # 11:00 - peak hours
            20, # 12:00 - lunch dip
            25, # 13:00 - afternoon peak
            25, # 14:00 - peak hours
            25, # 15:00 - peak hours
            20, # 16:00 - declining
            15, # 17:00 - business hours end
            10, # 18:00 - evening
            8,  # 19:00 - declining
            5,  # 20:00 - late evening
            3,  # 21:00 - rare
            2,  # 22:00 - very rare
            1   # 23:00 - very rare
        ]

        # Select hour using weighted random choice
        hour = random.choices(range(24), weights=hour_weights)[0]
        minute = random.randint(0, 59)
        second = random.randint(0, 59)

        # Combine date and time
        timestamp = random_date.replace(hour=hour, minute=minute, second=second)
        return timestamp

    def extract_temporal_features(self, timestamp: datetime) -> Dict:
        """
        Extract comprehensive temporal features from a timestamp.

        Derives multiple time-based features that help the model learn
        temporal patterns in contact center interactions.

        Args:
            timestamp: Datetime object to extract features from

        Returns:
            dict: Dictionary containing temporal features including:
                - timestamp: ISO format timestamp
                - time_of_day: morning/afternoon/evening/night
                - day_of_week: Monday-Sunday
                - is_business_hours: boolean
                - is_weekend: boolean
                - is_holiday: boolean
                - peak_season: holiday_shopping/tax_season/etc or None
                - queue_wait_seconds/minutes: simulated wait time
                - week_of_year, month, quarter: calendar features

        Note:
            Queue wait times are simulated based on realistic patterns:
            - Longer during peak seasons (1.5-3x multiplier)
            - Longer during business hours (1.2-1.8x multiplier)
            - Shorter off-hours and weekends (0.5-0.8x multiplier)
        """

        date_str = timestamp.strftime("%Y-%m-%d")

        # Classify time of day into broad categories
        # These align with typical contact center shift patterns
        hour = timestamp.hour
        if 5 <= hour < 12:
            time_of_day = "morning"
        elif 12 <= hour < 17:
            time_of_day = "afternoon"
        elif 17 <= hour < 21:
            time_of_day = "evening"
        else:
            time_of_day = "night"

        # Day of week features
        day_of_week = timestamp.strftime("%A")
        day_of_week_num = timestamp.weekday()  # 0=Monday, 6=Sunday

        # Determine if during business hours (M-F, 8am-6pm)
        # Important feature for predicting interaction patterns
        is_business_hours = (
            day_of_week_num < 5 and  # Monday-Friday
            8 <= hour < 18           # 8am-6pm
        )

        # Weekend flag (Saturday=5, Sunday=6)
        is_weekend = day_of_week_num >= 5

        # Check if date is a US federal holiday
        is_holiday = date_str in self.holidays

        # Determine if in a peak contact center season
        # Peak seasons have higher volume and different interaction patterns
        peak_season = None
        for season, date_ranges in self.peak_seasons.items():
            # Date ranges are stored as pairs: [start1, end1, start2, end2, ...]
            for i in range(0, len(date_ranges), 2):
                start = datetime.strptime(date_ranges[i], "%Y-%m-%d")
                end = datetime.strptime(date_ranges[i+1], "%Y-%m-%d")
                if start <= timestamp <= end:
                    peak_season = season
                    break

        # Simulate realistic queue wait times based on temporal factors
        # Real contact centers have longer waits during peak times
        base_wait = random.randint(30, 180)  # Base: 30s - 3min

        # Apply multipliers based on temporal context
        if peak_season:
            # Peak seasons = 1.5-3x longer waits
            base_wait *= random.uniform(1.5, 3.0)
        if is_business_hours and time_of_day in ["morning", "afternoon"]:
            # Peak business hours = 1.2-1.8x longer waits
            base_wait *= random.uniform(1.2, 1.8)
        if is_weekend or not is_business_hours:
            # Off-hours = 0.5-0.8x shorter waits (less volume)
            base_wait *= random.uniform(0.5, 0.8)

        queue_wait_seconds = int(base_wait)

        # First contact of day (more likely in morning)
        is_first_contact = (
            time_of_day == "morning" and
            random.random() < 0.3
        )

        # Follow-up call (more likely in afternoon)
        is_followup = (
            time_of_day in ["afternoon", "evening"] and
            random.random() < 0.25
        )

        return {
            "timestamp": timestamp.isoformat(),
            "date": date_str,
            "time_of_day": time_of_day,
            "hour": hour,
            "day_of_week": day_of_week,
            "day_of_week_num": day_of_week_num,
            "is_business_hours": is_business_hours,
            "is_weekend": is_weekend,
            "is_holiday": is_holiday,
            "peak_season": peak_season,
            "queue_wait_seconds": queue_wait_seconds,
            "queue_wait_minutes": round(queue_wait_seconds / 60, 1),
            "is_first_contact": is_first_contact,
            "is_followup": is_followup,
            "week_of_year": timestamp.isocalendar()[1],
            "month": timestamp.month,
            "month_name": timestamp.strftime("%B"),
            "quarter": (timestamp.month - 1) // 3 + 1
        }

    def enhance_example(self, example: Dict) -> Dict:
        """Add temporal features to a training example"""

        # Generate realistic timestamp
        timestamp = self.generate_realistic_timestamp()

        # Extract temporal features
        temporal_features = self.extract_temporal_features(timestamp)

        # Add to example metadata
        enhanced = example.copy()
        enhanced["temporal_features"] = temporal_features

        # Also add temporal context to the user message
        messages = enhanced.get("messages", [])
        if messages and len(messages) > 0:
            user_message = messages[0].get("content", "")

            # Add temporal context header
            temporal_context = f"""[TEMPORAL CONTEXT]
Timestamp: {temporal_features['timestamp']}
Time of day: {temporal_features['time_of_day']}
Day: {temporal_features['day_of_week']}
Business hours: {'Yes' if temporal_features['is_business_hours'] else 'No'}
Queue wait: {temporal_features['queue_wait_minutes']} minutes
{f"Peak season: {temporal_features['peak_season']}" if temporal_features['peak_season'] else ""}
{f"Holiday period" if temporal_features['is_holiday'] else ""}

"""

            # Prepend to user message
            messages[0]["content"] = temporal_context + user_message
            enhanced["messages"] = messages

        return enhanced


def enhance_dataset(input_file: str, output_file: str):
    """Enhance entire dataset with temporal features"""

    print(f"\n{'='*80}")
    print(f"TEMPORAL FEATURE ENHANCEMENT")
    print(f"{'='*80}")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"{'='*80}\n")

    # Load data
    print("Loading training data...")
    with open(input_file, 'r') as f:
        examples = [json.loads(line) for line in f]

    print(f"Loaded {len(examples)} examples\n")

    # Initialize enhancer
    enhancer = TemporalFeatureEnhancer()

    # Process examples
    print("Adding temporal features...")
    enhanced_examples = []

    for example in tqdm(examples, desc="Processing"):
        enhanced = enhancer.enhance_example(example)
        enhanced_examples.append(enhanced)

    # Save enhanced data
    print(f"\nSaving enhanced data to {output_file}...")
    with open(output_file, 'w') as f:
        for example in enhanced_examples:
            f.write(json.dumps(example) + '\n')

    # Statistics
    print(f"\n{'='*80}")
    print(f"ENHANCEMENT COMPLETE")
    print(f"{'='*80}")
    print(f"✅ Total examples: {len(enhanced_examples)}")

    # Temporal distribution stats
    time_of_day_counts = {}
    day_of_week_counts = {}
    business_hours_count = 0
    holiday_count = 0
    peak_season_counts = {}

    for ex in enhanced_examples:
        tf = ex.get("temporal_features", {})

        tod = tf.get("time_of_day")
        if tod:
            time_of_day_counts[tod] = time_of_day_counts.get(tod, 0) + 1

        dow = tf.get("day_of_week")
        if dow:
            day_of_week_counts[dow] = day_of_week_counts.get(dow, 0) + 1

        if tf.get("is_business_hours"):
            business_hours_count += 1

        if tf.get("is_holiday"):
            holiday_count += 1

        ps = tf.get("peak_season")
        if ps:
            peak_season_counts[ps] = peak_season_counts.get(ps, 0) + 1

    print(f"\n📊 Temporal Distribution:")
    print(f"\nTime of Day:")
    for tod, count in sorted(time_of_day_counts.items()):
        pct = (count / len(enhanced_examples)) * 100
        print(f"  {tod}: {count} ({pct:.1f}%)")

    print(f"\nDay of Week:")
    for dow in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
        count = day_of_week_counts.get(dow, 0)
        pct = (count / len(enhanced_examples)) * 100
        print(f"  {dow}: {count} ({pct:.1f}%)")

    print(f"\nBusiness Hours: {business_hours_count} ({(business_hours_count/len(enhanced_examples))*100:.1f}%)")
    print(f"Holidays: {holiday_count} ({(holiday_count/len(enhanced_examples))*100:.1f}%)")

    if peak_season_counts:
        print(f"\nPeak Seasons:")
        for season, count in sorted(peak_season_counts.items()):
            pct = (count / len(enhanced_examples)) * 100
            print(f"  {season}: {count} ({pct:.1f}%)")

    # Average queue wait time
    avg_queue_wait = sum(ex.get("temporal_features", {}).get("queue_wait_minutes", 0)
                         for ex in enhanced_examples) / len(enhanced_examples)
    print(f"\n⏱️  Average Queue Wait: {avg_queue_wait:.1f} minutes")

    print(f"\n💾 Saved to: {output_file}")
    print(f"{'='*80}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Add temporal features to training data")
    parser.add_argument('--input', type=str, default='data/train_chatml.jsonl',
                       help='Input training file')
    parser.add_argument('--output', type=str, default='data/train_temporal.jsonl',
                       help='Output file with temporal features')
    parser.add_argument('--validation', action='store_true',
                       help='Also process validation set')
    parser.add_argument('--test', action='store_true',
                       help='Also process test set')

    args = parser.parse_args()

    # Process training data
    enhance_dataset(args.input, args.output)

    # Process validation if requested
    if args.validation:
        enhance_dataset('data/validation_chatml.jsonl', 'data/validation_temporal.jsonl')

    # Process test if requested
    if args.test:
        enhance_dataset('data/test_chatml.jsonl', 'data/test_temporal.jsonl')


if __name__ == "__main__":
    main()
