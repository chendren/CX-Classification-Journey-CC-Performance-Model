#!/usr/bin/env python3
"""
RAG Integration for Contact Center Model

Connects fine-tuned model to ChromaDB for retrieval-augmented generation.
Enhances model predictions with similar historical interactions.

Features:
- Query ChromaDB for similar interactions
- Retrieve top-k relevant examples
- Inject context into model prompts
- Hybrid: Model reasoning + Historical patterns
"""

import os
import json
import chromadb
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime

class ContactCenterRAG:
    def __init__(
        self,
        chroma_path: str = "/Users/chadhendren/contact-center-analytics/database/chroma_db",
        collection_name: str = "contact_center_transcripts"
    ):
        """Initialize RAG system with ChromaDB"""

        print(f"Initializing Contact Center RAG...")
        print(f"ChromaDB path: {chroma_path}")
        print(f"Collection: {collection_name}")

        # Connect to ChromaDB
        self.client = chromadb.PersistentClient(path=chroma_path)

        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"✅ Connected to existing collection: {collection_name}")
            print(f"   Documents: {self.collection.count()}")
        except:
            print(f"❌ Collection '{collection_name}' not found")
            print(f"   Available collections: {[c.name for c in self.client.list_collections()]}")
            raise

    def retrieve_similar(
        self,
        query: str,
        n_results: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Retrieve similar interactions from ChromaDB

        Args:
            query: Transcript or query text
            n_results: Number of similar examples to retrieve
            filters: Optional metadata filters

        Returns:
            List of similar interactions with metadata
        """

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=filters
        )

        # Format results
        similar_interactions = []

        if results['documents'] and len(results['documents']) > 0:
            for i in range(len(results['documents'][0])):
                interaction = {
                    'transcript': results['documents'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None,
                    'metadata': results['metadatas'][0][i] if 'metadatas' in results else {},
                    'id': results['ids'][0][i]
                }
                similar_interactions.append(interaction)

        return similar_interactions

    def create_rag_prompt(
        self,
        current_transcript: str,
        similar_interactions: List[Dict],
        task: str = "quality_analysis"
    ) -> str:
        """
        Create RAG-enhanced prompt with similar examples

        Args:
            current_transcript: The transcript to analyze
            similar_interactions: Retrieved similar interactions
            task: Analysis task type

        Returns:
            Enhanced prompt with historical context
        """

        # Build similar examples section
        examples_context = ""

        if similar_interactions:
            examples_context = "\n[HISTORICAL CONTEXT - Similar Interactions]\n\n"

            for i, interaction in enumerate(similar_interactions, 1):
                metadata = interaction.get('metadata', {})

                examples_context += f"Example {i}:\n"

                # Add quality score if available
                if 'quality_score' in metadata:
                    examples_context += f"Quality Score: {metadata['quality_score']}/100\n"

                # Add CSAT if available
                if 'csat_score' in metadata:
                    examples_context += f"CSAT: {metadata['csat_score']}/5\n"

                # Add resolution status
                if 'resolution_status' in metadata:
                    examples_context += f"Resolution: {metadata['resolution_status']}\n"

                # Add key issues
                if 'issues' in metadata:
                    examples_context += f"Issues: {metadata['issues']}\n"

                # Add brief transcript excerpt
                transcript = interaction.get('transcript', '')
                if len(transcript) > 500:
                    transcript = transcript[:500] + "..."

                examples_context += f"Transcript: {transcript}\n\n"

        # Create enhanced prompt
        prompt = f"""{examples_context}[CURRENT INTERACTION TO ANALYZE]

{current_transcript}

Based on the historical context above and your analysis of the current interaction, provide:
1. Quality assessment with comparison to similar interactions
2. Pattern recognition from historical data
3. Specific recommendations based on what worked/didn't work in similar cases
4. Risk assessment considering similar outcomes
"""

        return prompt

    def analyze_with_rag(
        self,
        transcript: str,
        task: str = "quality_analysis",
        n_similar: int = 5,
        filters: Optional[Dict] = None
    ) -> Dict:
        """
        Analyze transcript with RAG enhancement

        Args:
            transcript: The transcript to analyze
            task: Analysis task type
            n_similar: Number of similar examples to retrieve
            filters: Optional metadata filters

        Returns:
            Analysis results with RAG context
        """

        # Retrieve similar interactions
        similar = self.retrieve_similar(
            query=transcript,
            n_results=n_similar,
            filters=filters
        )

        # Create RAG-enhanced prompt
        rag_prompt = self.create_rag_prompt(
            current_transcript=transcript,
            similar_interactions=similar,
            task=task
        )

        # Return prompt and context
        return {
            'rag_prompt': rag_prompt,
            'similar_interactions': similar,
            'num_similar': len(similar),
            'retrieval_successful': len(similar) > 0
        }

    def get_collection_stats(self) -> Dict:
        """Get statistics about the ChromaDB collection"""

        count = self.collection.count()

        # Sample some documents to understand metadata
        sample = self.collection.get(limit=10)

        metadata_keys = set()
        if sample['metadatas']:
            for metadata in sample['metadatas']:
                if metadata:
                    metadata_keys.update(metadata.keys())

        return {
            'total_documents': count,
            'collection_name': self.collection.name,
            'available_metadata_fields': list(metadata_keys)
        }


def test_rag_system(chroma_path: str, collection_name: str):
    """Test the RAG system with sample queries"""

    print(f"\n{'='*80}")
    print(f"RAG SYSTEM TEST")
    print(f"{'='*80}\n")

    # Initialize RAG
    rag = ContactCenterRAG(chroma_path=chroma_path, collection_name=collection_name)

    # Get stats
    stats = rag.get_collection_stats()
    print(f"Collection Stats:")
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Metadata fields: {', '.join(stats['available_metadata_fields'])}\n")

    # Test query
    test_query = "Customer is frustrated about billing error and wants refund"

    print(f"Test Query: {test_query}\n")

    # Retrieve similar
    similar = rag.retrieve_similar(test_query, n_results=3)

    print(f"Retrieved {len(similar)} similar interactions:\n")

    for i, interaction in enumerate(similar, 1):
        print(f"Result {i}:")
        print(f"  Distance: {interaction['distance']:.4f}")
        print(f"  Metadata: {json.dumps(interaction['metadata'], indent=4)}")
        print(f"  Transcript preview: {interaction['transcript'][:200]}...")
        print()

    # Create RAG prompt
    print(f"\n{'='*80}")
    print(f"RAG-ENHANCED PROMPT")
    print(f"{'='*80}\n")

    rag_result = rag.analyze_with_rag(test_query, n_similar=3)
    print(rag_result['rag_prompt'][:1000] + "...\n")

    print(f"{'='*80}")
    print(f"✅ RAG System Test Complete")
    print(f"{'='*80}\n")


def create_rag_enhanced_dataset(
    input_file: str,
    output_file: str,
    chroma_path: str,
    collection_name: str,
    n_similar: int = 3
):
    """
    Create a RAG-enhanced training dataset

    Adds similar examples to each training instance for context-aware learning
    """

    print(f"\n{'='*80}")
    print(f"RAG-ENHANCED DATASET CREATION")
    print(f"{'='*80}")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Similar examples per instance: {n_similar}")
    print(f"{'='*80}\n")

    # Initialize RAG
    rag = ContactCenterRAG(chroma_path=chroma_path, collection_name=collection_name)

    # Load training data
    print("Loading training data...")
    with open(input_file, 'r') as f:
        examples = [json.loads(line) for line in f]

    print(f"Loaded {len(examples)} training examples\n")

    # Enhance each example
    print("Enhancing with RAG context...")
    from tqdm import tqdm

    enhanced_examples = []

    for example in tqdm(examples, desc="Processing"):
        # Extract transcript from user message
        messages = example.get('messages', [])
        if not messages or len(messages) < 1:
            continue

        user_message = messages[0].get('content', '')

        # Retrieve similar interactions
        try:
            similar = rag.retrieve_similar(
                query=user_message,
                n_results=n_similar
            )

            # Create RAG-enhanced example
            enhanced = example.copy()
            enhanced['rag_context'] = {
                'similar_interactions': [
                    {
                        'id': s['id'],
                        'distance': s['distance'],
                        'metadata': s['metadata']
                    }
                    for s in similar
                ],
                'num_similar': len(similar)
            }

            # Optionally add similar examples to the prompt
            if similar:
                rag_prompt = rag.create_rag_prompt(
                    current_transcript=user_message,
                    similar_interactions=similar
                )

                # Update user message with RAG context
                enhanced['messages'][0]['content'] = rag_prompt

            enhanced_examples.append(enhanced)

        except Exception as e:
            print(f"Error enhancing example: {e}")
            enhanced_examples.append(example)

    # Save enhanced dataset
    print(f"\nSaving RAG-enhanced dataset...")
    with open(output_file, 'w') as f:
        for example in enhanced_examples:
            f.write(json.dumps(example) + '\n')

    print(f"\n{'='*80}")
    print(f"RAG ENHANCEMENT COMPLETE")
    print(f"{'='*80}")
    print(f"✅ Enhanced {len(enhanced_examples)} examples")
    print(f"💾 Saved to: {output_file}")
    print(f"{'='*80}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RAG integration for contact center model")
    parser.add_argument('--chroma-path', type=str,
                       default='/Users/chadhendren/contact-center-analytics/database/chroma_db',
                       help='Path to ChromaDB')
    parser.add_argument('--collection', type=str,
                       default='contact_center_transcripts',
                       help='ChromaDB collection name')
    parser.add_argument('--test', action='store_true',
                       help='Run test queries')
    parser.add_argument('--enhance-dataset', action='store_true',
                       help='Create RAG-enhanced training dataset')
    parser.add_argument('--input', type=str,
                       default='data/train_temporal.jsonl',
                       help='Input training file')
    parser.add_argument('--output', type=str,
                       default='data/train_rag_enhanced.jsonl',
                       help='Output RAG-enhanced file')
    parser.add_argument('--n-similar', type=int, default=3,
                       help='Number of similar examples to retrieve')

    args = parser.parse_args()

    if args.test:
        test_rag_system(args.chroma_path, args.collection)

    if args.enhance_dataset:
        create_rag_enhanced_dataset(
            args.input,
            args.output,
            args.chroma_path,
            args.collection,
            args.n_similar
        )


if __name__ == "__main__":
    main()
