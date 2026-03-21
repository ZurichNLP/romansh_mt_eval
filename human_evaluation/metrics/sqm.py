"""
Calculate SQM scores from human evaluation data.
"""

import numpy as np
from datasets import Dataset


def calculate_sqm_scores(dataset: Dataset) -> dict[str, float]:
    """
    Calculate average z-score (SQM) for all systems.
    
    First calculates average per language pair (lp), then averages across them.
    
    Args:
        dataset: Hugging Face dataset with 'lp', 'system1', 'system2', 'rating1', 'rating2' columns
    
    Returns:
        Dictionary mapping system names to their mean z-score across all varieties
    """
    # Group scores by variety and track unique documents
    variety_scores = {}
    variety_documents = {}
    all_systems = set()
    
    for row in dataset:
        language_pair = row.get('lp', '')
        
        if not language_pair:
            continue
        
        system1 = row.get('system1', '')
        system2 = row.get('system2', '')
        rating1 = row.get('rating1')
        rating2 = row.get('rating2')
        document_id = row.get('document_id', '')
        
        all_systems.add(system1)
        all_systems.add(system2)
        
        if language_pair not in variety_scores:
            variety_scores[language_pair] = {}
            variety_documents[language_pair] = set()
        
        # Track documents for this variety
        if document_id:
            variety_documents[language_pair].add(document_id)
        
        # Collect scores for each system
        if system1 not in variety_scores[language_pair]:
            variety_scores[language_pair][system1] = []
        if system2 not in variety_scores[language_pair]:
            variety_scores[language_pair][system2] = []
        
        if rating1 is not None:
            variety_scores[language_pair][system1].append(rating1)
        if rating2 is not None:
            variety_scores[language_pair][system2].append(rating2)
    
    if not variety_scores or not all_systems:
        return {}
    
    varieties_to_process = list(variety_documents.keys())
    
    # Calculate mean per variety for each system, then average across varieties
    system_variety_means = {system: [] for system in all_systems}
    
    for variety in varieties_to_process:
        for system in all_systems:
            scores = variety_scores[variety].get(system, [])
            if scores:
                system_variety_means[system].append(np.mean(scores).item())
    
    # Average across varieties for each system
    average_scores = {}
    for system_name, variety_means in system_variety_means.items():
        if variety_means:
            average_scores[system_name] = np.mean(variety_means).item()
    
    return average_scores
