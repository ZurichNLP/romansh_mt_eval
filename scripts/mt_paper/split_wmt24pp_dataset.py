#!/usr/bin/env python3
"""
Split WMT24++ Romansh dataset in half by document IDs.

This script splits the dataset respecting domain and document boundaries.
Each domain is split in half, and the results are combined into a single
JSON file with ``first_half`` and ``second_half`` document ID lists (plus
``metadata``).

Output path:

    benchmarking/wmt24pp_split.json

"""

import json
from pathlib import Path
from collections import defaultdict

from datasets import load_dataset


def load_dataset_from_hf():
    """
    Load dataset from HuggingFace and extract domain/document_id information.
    
    Returns:
        Dictionary mapping domain to list of unique document IDs
    """
    domain_documents = defaultdict(set)
    
    # Load any single variety - document IDs are shared across all varieties
    dataset = load_dataset("ZurichNLP/wmt24pp-rm", "de_DE-rm-rumgr", split="test")
    
    for row in dataset:
        domain = row['domain']
        document_id = row['document_id']
        
        # Skip the canary domain (special test case)
        if domain != 'canary':
            domain_documents[domain].add(document_id)
    
    # Convert sets to sorted lists for consistent ordering
    return {domain: sorted(list(docs)) for domain, docs in domain_documents.items()}


def split_documents_by_domain(domain_documents):
    """
    Split documents in half for each domain.
    
    Args:
        domain_documents: Dictionary mapping domain to list of document IDs
        
    Returns:
        Tuple of (first_half_docs, second_half_docs, metadata)
    """
    first_half = []
    second_half = []
    metadata = {
        'domains': {}
    }
    
    for domain in sorted(domain_documents.keys()):
        documents = domain_documents[domain]
        total_count = len(documents)
        
        # Split in half (first half gets extra document if odd number)
        midpoint = (total_count + 1) // 2
        domain_first_half = documents[:midpoint]
        domain_second_half = documents[midpoint:]
        
        # Add to global lists
        first_half.extend(domain_first_half)
        second_half.extend(domain_second_half)
        
        # Record metadata
        metadata['domains'][domain] = {
            'total': total_count,
            'first_half': len(domain_first_half),
            'second_half': len(domain_second_half)
        }
    
    # Add overall metadata
    metadata['total_documents'] = len(first_half) + len(second_half)
    metadata['first_half_count'] = len(first_half)
    metadata['second_half_count'] = len(second_half)
    
    return first_half, second_half, metadata


def save_split(output_path, first_half, second_half, metadata):
    """
    Save the split to a JSON file.
    
    Args:
        output_path: Path to save the JSON file
        first_half: List of document IDs for first half
        second_half: List of document IDs for second half
        metadata: Dictionary with split statistics
    """
    output_data = {
        'first_half': first_half,
        'second_half': second_half,
        'metadata': metadata
    }
    
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(output_data, file, indent=2, ensure_ascii=False)


def print_summary(metadata):
    """
    Print summary statistics about the split.
    
    Args:
        metadata: Dictionary with split statistics
    """
    print("=" * 60)
    print("WMT24++ Dataset Split Summary")
    print("=" * 60)
    print(f"\nTotal documents: {metadata['total_documents']}")
    print(f"First half: {metadata['first_half_count']} documents")
    print(f"Second half: {metadata['second_half_count']} documents")
    print(f"\nNumber of domains processed: {len(metadata['domains'])}")
    print("\nPer-domain breakdown:")
    print("-" * 60)
    
    for domain in sorted(metadata['domains'].keys()):
        stats = metadata['domains'][domain]
        print(f"  {domain}:")
        print(f"    Total: {stats['total']}")
        print(f"    First half: {stats['first_half']}")
        print(f"    Second half: {stats['second_half']}")


def main():
    base_dir = Path(__file__).parent
    workspace_root = base_dir.parent.parent
    output_path = workspace_root / 'benchmarking' / 'wmt24pp_split.json'
    
    print("Loading dataset from HuggingFace...")
    domain_documents = load_dataset_from_hf()
    
    print("Splitting documents by domain...")
    first_half, second_half, metadata = split_documents_by_domain(domain_documents)
    
    print(f"Saving split to: {output_path}")
    save_split(output_path, first_half, second_half, metadata)
    
    print("\n✓ Split completed successfully!")
    print_summary(metadata)
    print(f"\nOutput saved to: {output_path}")


if __name__ == '__main__':
    main()















