#!/usr/bin/env python3
"""
Agent Performance Fingerprinting

Analyzes patterns across all transcripts to create unique performance profiles for each agent.
Uses clustering to identify behavior patterns, strengths, weaknesses, and coaching needs.

Features:
- Extract agent-specific metrics from training data
- Cluster agents by behavioral patterns
- Identify strengths and weaknesses
- Generate personalized coaching recommendations
- Create agent performance dashboards
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
from tqdm import tqdm
import re

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  scikit-learn not available. Install with: pip install scikit-learn")


class AgentFingerprinter:
    def __init__(self):
        """Initialize agent fingerprinting system"""
        self.agent_data = defaultdict(lambda: {
            'interactions': [],
            'quality_scores': [],
            'csat_scores': [],
            'resolution_rates': [],
            'first_call_resolution': [],
            'avg_handle_time': [],
            'empathy_scores': [],
            'technical_scores': [],
            'communication_scores': [],
            'escalations': [],
            'transfers': [],
            'issues': defaultdict(int),
            'strengths': defaultdict(int),
            'weaknesses': defaultdict(int)
        })

    def extract_agent_metrics(self, examples: List[Dict]) -> Dict:
        """
        Extract metrics for each agent from training examples

        Args:
            examples: List of training examples with transcripts and analyses

        Returns:
            Dictionary mapping agent IDs to their metrics
        """

        print("Extracting agent metrics from training data...")

        for example in tqdm(examples, desc="Processing examples"):
            # Try to extract agent ID from the data
            agent_id = self._extract_agent_id(example)

            if not agent_id:
                continue  # Skip if no agent ID found

            # Extract metrics from the response
            messages = example.get('messages', [])
            if len(messages) < 2:
                continue

            # Parse assistant response
            assistant_response = messages[1].get('content', '')

            try:
                # Try to parse as JSON
                analysis = json.loads(assistant_response)
            except:
                # If not JSON, extract from text
                analysis = self._extract_from_text(assistant_response)

            # Add interaction to agent profile
            self.agent_data[agent_id]['interactions'].append(example)

            # Extract quality score
            quality_score = self._extract_quality_score(analysis)
            if quality_score:
                self.agent_data[agent_id]['quality_scores'].append(quality_score)

            # Extract CSAT
            csat = self._extract_csat(analysis)
            if csat:
                self.agent_data[agent_id]['csat_scores'].append(csat)

            # Extract resolution status
            resolution = self._extract_resolution(analysis)
            if resolution is not None:
                self.agent_data[agent_id]['first_call_resolution'].append(resolution)

            # Extract empathy, technical, communication scores
            self._extract_skill_scores(analysis, agent_id)

            # Extract issues and strengths
            self._extract_patterns(analysis, agent_id)

        return dict(self.agent_data)

    def _extract_agent_id(self, example: Dict) -> Optional[str]:
        """Extract agent ID from example"""

        # Try to find agent ID in metadata
        if 'metadata' in example:
            if 'agent_id' in example['metadata']:
                return example['metadata']['agent_id']

        # Try to extract from transcript
        messages = example.get('messages', [])
        if messages:
            transcript = messages[0].get('content', '')

            # Look for agent name patterns
            agent_patterns = [
                r'Agent:\s*([A-Z][a-z]+)',
                r'Representative:\s*([A-Z][a-z]+)',
                r'Hello.*my name is ([A-Z][a-z]+)',
                r'This is ([A-Z][a-z]+)'
            ]

            for pattern in agent_patterns:
                match = re.search(pattern, transcript)
                if match:
                    return match.group(1)

        # Generate synthetic agent ID based on interaction characteristics
        # This ensures we can still do clustering even without explicit agent IDs
        return f"agent_{hash(str(example)) % 1000:03d}"

    def _extract_quality_score(self, analysis: Dict) -> Optional[float]:
        """Extract quality score from analysis"""

        if isinstance(analysis, dict):
            # Try different keys
            for key in ['quality_score', 'overall_quality', 'score']:
                if key in analysis:
                    val = analysis[key]
                    if isinstance(val, (int, float)):
                        return float(val)

            # Try nested
            if 'quality_scores' in analysis:
                qs = analysis['quality_scores']
                if isinstance(qs, dict) and 'overall_quality' in qs:
                    return float(qs['overall_quality'])

        # Extract from text
        if isinstance(analysis, str):
            match = re.search(r'quality.*?(\d+)/100', analysis, re.IGNORECASE)
            if match:
                return float(match.group(1))

        return None

    def _extract_csat(self, analysis: Dict) -> Optional[float]:
        """Extract CSAT score from analysis"""

        if isinstance(analysis, dict):
            for key in ['csat_score', 'csat', 'satisfaction']:
                if key in analysis:
                    val = analysis[key]
                    if isinstance(val, (int, float)):
                        return float(val)

        if isinstance(analysis, str):
            match = re.search(r'csat.*?(\d+)/5', analysis, re.IGNORECASE)
            if match:
                return float(match.group(1))

        return None

    def _extract_resolution(self, analysis: Dict) -> Optional[bool]:
        """Extract resolution status"""

        if isinstance(analysis, dict):
            for key in ['resolution', 'resolved', 'first_call_resolution']:
                if key in analysis:
                    val = analysis[key]
                    if isinstance(val, bool):
                        return val
                    if isinstance(val, str):
                        return val.lower() in ['yes', 'true', 'resolved', 'achieved']

        if isinstance(analysis, str):
            if re.search(r'first.?call.?resolution.*achieved', analysis, re.IGNORECASE):
                return True
            if re.search(r'first.?call.?resolution.*not achieved', analysis, re.IGNORECASE):
                return False

        return None

    def _extract_skill_scores(self, analysis: Dict, agent_id: str):
        """Extract empathy, technical, communication scores"""

        if isinstance(analysis, dict):
            # Empathy
            if 'empathy' in analysis:
                self.agent_data[agent_id]['empathy_scores'].append(float(analysis['empathy']))

            # Technical
            if 'technical_knowledge' in analysis or 'technical' in analysis:
                score = analysis.get('technical_knowledge') or analysis.get('technical')
                if score:
                    self.agent_data[agent_id]['technical_scores'].append(float(score))

            # Communication
            if 'communication' in analysis:
                self.agent_data[agent_id]['communication_scores'].append(float(analysis['communication']))

    def _extract_patterns(self, analysis: Dict, agent_id: str):
        """Extract issues and strengths from analysis"""

        text = json.dumps(analysis) if isinstance(analysis, dict) else str(analysis)
        text_lower = text.lower()

        # Common issues
        issue_patterns = {
            'poor_empathy': r'lack.*empathy|not empathetic|cold|dismissive',
            'knowledge_gap': r'lack.*knowledge|didn.?t know|unsure|uncertain',
            'poor_communication': r'poor.*communication|unclear|confusing',
            'long_hold': r'long.*hold|excessive.*wait',
            'multiple_transfers': r'multiple.*transfer|transferred.*times',
            'failed_resolution': r'failed.*resolve|not.*resolved|unresolved'
        }

        for issue, pattern in issue_patterns.items():
            if re.search(pattern, text_lower):
                self.agent_data[agent_id]['issues'][issue] += 1
                self.agent_data[agent_id]['weaknesses'][issue] += 1

        # Common strengths
        strength_patterns = {
            'excellent_empathy': r'excellent.*empathy|very empathetic|compassionate',
            'strong_knowledge': r'strong.*knowledge|expert|well.?informed',
            'clear_communication': r'clear.*communication|articulate|well.?explained',
            'quick_resolution': r'quick.*resolution|efficiently.*resolved|fast',
            'first_call_resolution': r'first.?call.?resolution.*achieved',
            'great_rapport': r'great.*rapport|built.*relationship|personable'
        }

        for strength, pattern in strength_patterns.items():
            if re.search(pattern, text_lower):
                self.agent_data[agent_id]['strengths'][strength] += 1

    def _extract_from_text(self, text: str) -> Dict:
        """Extract structured data from text response"""
        return {'raw_text': text}

    def compute_agent_profiles(self) -> Dict:
        """
        Compute comprehensive profiles for each agent

        Returns:
            Dictionary mapping agent IDs to their performance profiles
        """

        print("\nComputing agent performance profiles...")

        profiles = {}

        for agent_id, data in tqdm(self.agent_data.items(), desc="Creating profiles"):
            profile = {
                'agent_id': agent_id,
                'total_interactions': len(data['interactions']),
                'metrics': {},
                'strengths': [],
                'weaknesses': [],
                'coaching_priorities': []
            }

            # Calculate averages
            if data['quality_scores']:
                profile['metrics']['avg_quality_score'] = np.mean(data['quality_scores'])
                profile['metrics']['quality_std'] = np.std(data['quality_scores'])

            if data['csat_scores']:
                profile['metrics']['avg_csat'] = np.mean(data['csat_scores'])
                profile['metrics']['csat_std'] = np.std(data['csat_scores'])

            if data['first_call_resolution']:
                profile['metrics']['fcr_rate'] = np.mean(data['first_call_resolution']) * 100

            if data['empathy_scores']:
                profile['metrics']['avg_empathy'] = np.mean(data['empathy_scores'])

            if data['technical_scores']:
                profile['metrics']['avg_technical'] = np.mean(data['technical_scores'])

            if data['communication_scores']:
                profile['metrics']['avg_communication'] = np.mean(data['communication_scores'])

            # Top strengths
            if data['strengths']:
                top_strengths = sorted(data['strengths'].items(), key=lambda x: x[1], reverse=True)[:3]
                profile['strengths'] = [{'skill': s[0], 'count': s[1]} for s in top_strengths]

            # Top weaknesses
            if data['weaknesses']:
                top_weaknesses = sorted(data['weaknesses'].items(), key=lambda x: x[1], reverse=True)[:3]
                profile['weaknesses'] = [{'skill': w[0], 'count': w[1]} for w in top_weaknesses]

                # Coaching priorities based on weaknesses
                for weakness, count in top_weaknesses:
                    coaching = self._get_coaching_for_weakness(weakness)
                    profile['coaching_priorities'].append(coaching)

            # Performance tier
            avg_quality = profile['metrics'].get('avg_quality_score', 0)
            if avg_quality >= 85:
                profile['performance_tier'] = 'Excellent'
            elif avg_quality >= 75:
                profile['performance_tier'] = 'Good'
            elif avg_quality >= 65:
                profile['performance_tier'] = 'Needs Improvement'
            else:
                profile['performance_tier'] = 'Critical'

            profiles[agent_id] = profile

        return profiles

    def _get_coaching_for_weakness(self, weakness: str) -> Dict:
        """Generate coaching recommendation for a specific weakness"""

        coaching_map = {
            'poor_empathy': {
                'skill': 'Empathy & Emotional Intelligence',
                'focus_area': 'Active listening and emotional acknowledgment',
                'action': 'Practice acknowledging customer emotions explicitly before problem-solving',
                'training': 'Customer empathy workshop'
            },
            'knowledge_gap': {
                'skill': 'Product/Service Knowledge',
                'focus_area': 'Technical and procedural expertise',
                'action': 'Complete product knowledge certification and maintain resource guide',
                'training': 'Advanced product training'
            },
            'poor_communication': {
                'skill': 'Communication Clarity',
                'focus_area': 'Clear, structured explanations',
                'action': 'Use structured frameworks (e.g., situation-action-result)',
                'training': 'Effective communication workshop'
            },
            'long_hold': {
                'skill': 'Efficiency & Time Management',
                'focus_area': 'Reducing hold times and improving workflow',
                'action': 'Pre-prepare common resources, minimize hold times',
                'training': 'Time management and efficiency training'
            },
            'multiple_transfers': {
                'skill': 'Issue Resolution & Ownership',
                'focus_area': 'First-call resolution capability',
                'action': 'Take ownership of issues, consult resources before transferring',
                'training': 'Problem-solving and escalation protocols'
            },
            'failed_resolution': {
                'skill': 'Problem-Solving & Follow-Through',
                'focus_area': 'Complete issue resolution',
                'action': 'Verify resolution with customer, schedule follow-up if needed',
                'training': 'Advanced troubleshooting skills'
            }
        }

        return coaching_map.get(weakness, {
            'skill': weakness.replace('_', ' ').title(),
            'focus_area': 'General improvement needed',
            'action': 'Review best practices and seek mentor guidance',
            'training': 'General skills development'
        })

    def cluster_agents(self, profiles: Dict, n_clusters: int = 5) -> Dict:
        """
        Cluster agents by behavioral patterns

        Args:
            profiles: Agent performance profiles
            n_clusters: Number of clusters

        Returns:
            Cluster assignments and characteristics
        """

        if not SKLEARN_AVAILABLE:
            print("⚠️  Clustering requires scikit-learn")
            return {}

        print(f"\nClustering agents into {n_clusters} behavioral groups...")

        # Prepare feature matrix
        agent_ids = []
        features = []

        for agent_id, profile in profiles.items():
            metrics = profile.get('metrics', {})

            feature_vector = [
                metrics.get('avg_quality_score', 70),
                metrics.get('avg_csat', 3.5),
                metrics.get('fcr_rate', 70),
                metrics.get('avg_empathy', 3.5),
                metrics.get('avg_technical', 3.5),
                metrics.get('avg_communication', 3.5),
                len(profile.get('weaknesses', [])),
                len(profile.get('strengths', []))
            ]

            agent_ids.append(agent_id)
            features.append(feature_vector)

        features = np.array(features)

        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Cluster
        kmeans = KMeans(n_clusters=min(n_clusters, len(agent_ids)), random_state=42)
        cluster_labels = kmeans.fit_predict(features_scaled)

        # Analyze clusters
        clusters = defaultdict(list)
        for agent_id, cluster_id in zip(agent_ids, cluster_labels):
            clusters[int(cluster_id)].append(agent_id)

        cluster_analysis = {}
        for cluster_id, agent_list in clusters.items():
            # Compute cluster characteristics
            cluster_profiles = [profiles[aid] for aid in agent_list]

            avg_quality = np.mean([p['metrics'].get('avg_quality_score', 0) for p in cluster_profiles])
            avg_csat = np.mean([p['metrics'].get('avg_csat', 0) for p in cluster_profiles if 'avg_csat' in p['metrics']])
            avg_fcr = np.mean([p['metrics'].get('fcr_rate', 0) for p in cluster_profiles if 'fcr_rate' in p['metrics']])

            cluster_analysis[f"cluster_{cluster_id}"] = {
                'cluster_id': cluster_id,
                'size': len(agent_list),
                'agents': agent_list,
                'characteristics': {
                    'avg_quality_score': round(avg_quality, 2),
                    'avg_csat': round(avg_csat, 2),
                    'avg_fcr_rate': round(avg_fcr, 2)
                },
                'cluster_name': self._name_cluster(avg_quality, avg_csat, avg_fcr)
            }

        return cluster_analysis

    def _name_cluster(self, quality: float, csat: float, fcr: float) -> str:
        """Generate descriptive name for cluster"""

        if quality >= 85 and csat >= 4.5:
            return "⭐ Top Performers"
        elif quality >= 75:
            return "✅ Solid Performers"
        elif quality >= 65:
            return "📈 Developing Performers"
        else:
            return "🎯 High-Focus Group"


def generate_agent_reports(
    profiles: Dict,
    clusters: Dict,
    output_dir: str
):
    """Generate comprehensive agent performance reports"""

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    print(f"\nGenerating agent performance reports...")

    # Overall summary
    summary = {
        'total_agents': len(profiles),
        'total_clusters': len(clusters),
        'performance_distribution': {},
        'top_performers': [],
        'agents_needing_support': []
    }

    # Performance tiers
    for tier in ['Excellent', 'Good', 'Needs Improvement', 'Critical']:
        count = sum(1 for p in profiles.values() if p.get('performance_tier') == tier)
        summary['performance_distribution'][tier] = count

    # Top performers
    sorted_agents = sorted(
        profiles.items(),
        key=lambda x: x[1]['metrics'].get('avg_quality_score', 0),
        reverse=True
    )

    summary['top_performers'] = [
        {
            'agent_id': aid,
            'quality_score': p['metrics'].get('avg_quality_score', 0),
            'csat': p['metrics'].get('avg_csat', 0)
        }
        for aid, p in sorted_agents[:10]
    ]

    summary['agents_needing_support'] = [
        {
            'agent_id': aid,
            'quality_score': p['metrics'].get('avg_quality_score', 0),
            'weaknesses': p.get('weaknesses', [])
        }
        for aid, p in sorted_agents[-10:]
    ]

    # Save summary
    with open(output_path / 'agent_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Save all profiles
    with open(output_path / 'agent_profiles.json', 'w') as f:
        json.dump(profiles, f, indent=2)

    # Save clusters
    with open(output_path / 'agent_clusters.json', 'w') as f:
        json.dump(clusters, f, indent=2)

    print(f"✅ Reports saved to: {output_dir}")
    print(f"   - agent_summary.json")
    print(f"   - agent_profiles.json")
    print(f"   - agent_clusters.json")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Create agent performance fingerprints")
    parser.add_argument('--input', type=str, default='data/train_temporal.jsonl',
                       help='Input training file')
    parser.add_argument('--output-dir', type=str, default='data/agent_profiles',
                       help='Output directory for reports')
    parser.add_argument('--n-clusters', type=int, default=5,
                       help='Number of agent clusters')

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"AGENT PERFORMANCE FINGERPRINTING")
    print(f"{'='*80}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*80}\n")

    # Load data
    print("Loading training data...")
    with open(args.input, 'r') as f:
        examples = [json.loads(line) for line in f]

    print(f"Loaded {len(examples)} training examples\n")

    # Initialize fingerprinter
    fingerprinter = AgentFingerprinter()

    # Extract metrics
    agent_data = fingerprinter.extract_agent_metrics(examples)
    print(f"\n✅ Analyzed {len(agent_data)} unique agents")

    # Compute profiles
    profiles = fingerprinter.compute_agent_profiles()

    # Cluster agents
    clusters = fingerprinter.cluster_agents(profiles, n_clusters=args.n_clusters)

    # Generate reports
    generate_agent_reports(profiles, clusters, args.output_dir)

    print(f"\n{'='*80}")
    print(f"AGENT FINGERPRINTING COMPLETE")
    print(f"{'='*80}")
    print(f"✅ Profiles created for {len(profiles)} agents")
    print(f"✅ Organized into {len(clusters)} behavioral clusters")
    print(f"💾 Reports saved to: {args.output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
