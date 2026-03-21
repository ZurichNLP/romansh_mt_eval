"""
Bootstrap confidence intervals for human evaluation metrics.
"""

from typing import Callable, Literal

import numpy as np
from datasets import Dataset


def resample_by_segment(dataset: Dataset, rng: np.random.Generator) -> Dataset:
    """
    Resample a dataset by sampling individual rows with replacement.
    
    Args:
        dataset: Hugging Face dataset to resample
        rng: NumPy random generator
    
    Returns:
        Resampled dataset
    """
    n = len(dataset)
    indices = rng.choice(n, n, replace=True)
    return dataset.select(indices)


def resample_by_document(dataset: Dataset, rng: np.random.Generator) -> Dataset:
    """
    Resample a dataset by sampling document IDs with replacement.
    
    All rows belonging to a sampled document are included. When a document
    is sampled multiple times, its rows are included multiple times.
    
    Args:
        dataset: Hugging Face dataset with 'document_id' column
        rng: NumPy random generator
    
    Returns:
        Resampled dataset
    """
    # Get unique document IDs and build index mapping
    document_ids = dataset["document_id"]
    unique_document_ids = list(set(document_ids))
    n_documents = len(unique_document_ids)
    
    # Build mapping from document_id to row indices
    document_to_indices = {doc_id: [] for doc_id in unique_document_ids}
    for row_index, doc_id in enumerate(document_ids):
        document_to_indices[doc_id].append(row_index)
    
    # Sample document IDs with replacement
    sampled_document_ids = rng.choice(unique_document_ids, n_documents, replace=True)
    
    # Collect all row indices for sampled documents
    indices = []
    for doc_id in sampled_document_ids:
        indices.extend(document_to_indices[doc_id])
    
    return dataset.select(indices)


def bootstrap_confidence_interval(
    dataset: Dataset,
    metric_function: Callable[[Dataset], dict[str, float]],
    resampling_unit: Literal["segment", "document"],
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int | None = None,
) -> dict[str, tuple[float, float]]:
    """
    Calculate bootstrap confidence intervals for a metric.
    
    Args:
        dataset: Hugging Face dataset
        metric_function: Function that takes a dataset and returns a dict
            mapping system names to scores
        resampling_unit: "segment" to resample rows, "document" to resample
            by document_id
        n_resamples: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary mapping system names to (lower, upper) confidence bounds
    """
    if confidence_level <= 0 or confidence_level >= 1:
        raise ValueError("confidence_level must be between 0 and 1 (exclusive)")
    
    if n_resamples <= 0:
        raise ValueError("n_resamples must be a positive integer")
    
    # Select resampling function
    if resampling_unit == "segment":
        resample_function = resample_by_segment
    elif resampling_unit == "document":
        resample_function = resample_by_document
    else:
        raise ValueError(f"Unknown resampling_unit: {resampling_unit}")
    
    # Initialize random generator
    rng = np.random.default_rng(random_seed)
    
    # Collect bootstrap samples
    samples_by_system: dict[str, list[float]] = {}
    
    for _ in range(n_resamples):
        resampled_dataset = resample_function(dataset, rng)
        scores = metric_function(resampled_dataset)
        
        for system_name, score in scores.items():
            if system_name not in samples_by_system:
                samples_by_system[system_name] = []
            if score is not None and not np.isnan(score):
                samples_by_system[system_name].append(score)
    
    # Calculate confidence intervals using percentile method
    alpha = (1 - confidence_level) / 2
    lower_percentile = alpha * 100
    upper_percentile = (1 - alpha) * 100
    
    confidence_intervals = {}
    for system_name, samples in samples_by_system.items():
        if samples:
            lower = np.percentile(samples, lower_percentile)
            upper = np.percentile(samples, upper_percentile)
            confidence_intervals[system_name] = (float(lower), float(upper))
    
    return confidence_intervals

