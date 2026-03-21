#!/usr/bin/env python3
"""
Perform pairwise comparison of all systems across three metrics.

Loads JSONL data and calculates the percentage that one system beats another,
averaged across varieties. Generates three LaTeX-formatted matrices.
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import dotenv
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics.bootstrap import bootstrap_confidence_interval
from metrics.dataloader import add_pairwise_dataset_cli_arguments, load_metric_dataset


SYSTEM_ORDER = [
    "reference",
    "Gemini-3-Pro",
    "romansh-nllb-1.3b",
    "romansh-nllb-1.3b-dict-prompting",
]

# System name mappings: data name -> LaTeX display name
SYSTEM_LATEX_NAMES = {
    "reference": "Human reference",
    "Gemini-3-Pro": "Gemini 3 Pro (preview)",
    "romansh-nllb-1.3b": "LR$\\rightarrow$HR augmentation",
    "romansh-nllb-1.3b-dict-prompting": "+ dictionary prompting",
}


def determine_winner(
    system1: str,
    system2: str,
    rating1: float | int | None,
    rating2: float | int | None,
    metric_type: str,
) -> str | None:
    """
    Determine which system wins a comparison based on metric type.
    
    Args:
        system1: First system name
        system2: Second system name
        rating1: Rating for system1
        rating2: Rating for system2
        metric_type: One of "document_accuracy", "segment_accuracy", "segment_fluency"
    
    Returns:
        Winner system name, or None if tie or invalid
    """
    if rating1 is None or rating2 is None:
        return None
    
    if metric_type == "segment_accuracy":
        # Binary comparison: rating1 == 1 means system1 wins, rating2 == 1 means system2 wins
        if rating1 == 1 and rating2 == 0:
            return system1
        elif rating1 == 0 and rating2 == 1:
            return system2
        else:
            # Tie (both 0 or both 1)
            return None
    else:
        # document_accuracy and segment_fluency: higher rating wins
        if rating1 > rating2:
            return system1
        elif rating2 > rating1:
            return system2
        else:
            # Tie
            return None


def calculate_pairwise_matrix_per_variety(
    dataset: Dataset,
    metric_type: str,
) -> dict[str, dict[tuple[str, str], tuple[int, int]]]:
    """
    Calculate pairwise win counts per variety.
    
    Args:
        dataset: Hugging Face dataset with 'lp', 'system1', 'system2', 'rating1', 'rating2' columns
        metric_type: One of "document_accuracy", "segment_accuracy", "segment_fluency"
    
    Returns:
        Dictionary mapping variety (lp) to dictionary mapping (system1, system2) tuples to (wins, total) counts
    """
    variety_comparisons: dict[str, dict[tuple[str, str], tuple[int, int]]] = defaultdict(
        lambda: defaultdict(lambda: (0, 0))
    )
    
    for row in dataset:
        language_pair = row.get("lp", "")
        if not language_pair:
            continue
        
        system1 = row.get("system1", "")
        system2 = row.get("system2", "")
        rating1 = row.get("rating1")
        rating2 = row.get("rating2")
        
        if not system1 or not system2:
            continue
        
        winner = determine_winner(system1, system2, rating1, rating2, metric_type)
        
        # Track both directions: (system1, system2) and (system2, system1)
        pair = (system1, system2)
        wins, total = variety_comparisons[language_pair][pair]
        total += 1
        if winner == system1:
            wins += 1
        variety_comparisons[language_pair][pair] = (wins, total)
        
        # Also track reverse direction
        reverse_pair = (system2, system1)
        reverse_wins, reverse_total = variety_comparisons[language_pair][reverse_pair]
        reverse_total += 1
        if winner == system2:
            reverse_wins += 1
        variety_comparisons[language_pair][reverse_pair] = (reverse_wins, reverse_total)
    
    return dict(variety_comparisons)


def average_across_varieties(
    variety_matrices: dict[str, dict[tuple[str, str], tuple[int, int]]],
    systems: list[str],
) -> dict[tuple[str, str], float]:
    """
    Average pairwise win percentages across varieties.
    
    Args:
        variety_matrices: Dictionary mapping variety to pairwise win counts
        systems: List of all systems to consider
    
    Returns:
        Dictionary mapping (system1, system2) tuples to average win percentage
    """
    # Collect percentages per variety for each system pair
    pair_percentages: dict[tuple[str, str], list[float]] = defaultdict(list)
    
    for variety, matrix in variety_matrices.items():
        for (system1, system2), (wins, total) in matrix.items():
            if system1 in systems and system2 in systems and total > 0:
                percentage = (wins / total) * 100.0
                pair_percentages[(system1, system2)].append(percentage)
    
    # Average across varieties
    averaged: dict[tuple[str, str], float] = {}
    for pair, percentages in pair_percentages.items():
        if percentages:
            averaged[pair] = sum(percentages) / len(percentages)
    
    return averaged


def calculate_pairwise_win_percentages(
    dataset: Dataset,
    systems: list[str],
    metric_type: str,
) -> dict[str, float]:
    """
    Calculate pairwise win percentages averaged across varieties.
    
    This function is designed to be used with bootstrap_confidence_interval.
    It returns a dict with string keys like "system1_vs_system2" mapping to
    win percentages.
    
    Args:
        dataset: Hugging Face dataset with 'lp', 'system1', 'system2', 'rating1', 'rating2' columns
        systems: List of all systems to consider
        metric_type: One of "document_accuracy", "segment_accuracy", "segment_fluency"
    
    Returns:
        Dictionary mapping "system1_vs_system2" strings to average win percentages
    """
    variety_matrices = calculate_pairwise_matrix_per_variety(dataset, metric_type)
    averaged_matrix = average_across_varieties(variety_matrices, systems)
    
    # Convert tuple keys to string keys for bootstrap compatibility
    result: dict[str, float] = {}
    for (system1, system2), percentage in averaged_matrix.items():
        key = f"{system1}_vs_{system2}"
        result[key] = percentage
    
    return result


def write_output(content: str, output_path: Path) -> None:
    """
    Write content to output file if parent directory exists.
    
    Args:
        content: Content to write
        output_path: Path to output file
    """
    if output_path.parent.exists():
        with output_path.open("w", encoding="utf-8") as f:
            f.write(content)
        print(f"\nResults written to {output_path}")
    else:
        print(f"\nWarning: Output directory does not exist: {output_path.parent}")


def format_number_with_phantom(number: int, max_digits: int = 3) -> str:
    """
    Format number with LaTeX phantom padding for alignment.
    
    Args:
        number: Number to format
        max_digits: Maximum number of digits to pad to
    
    Returns:
        Formatted string with phantom padding
    """
    number_str = str(number)
    if len(number_str) == max_digits:
        return number_str
    elif len(number_str) == max_digits - 1:
        return f"\\phantom{{0}}{number_str}"
    elif len(number_str) == max_digits - 2:
        return f"\\phantom{{00}}{number_str}"
    else:
        # For numbers with more digits, return as is
        return number_str


def format_latex_table(
    averaged_matrix: dict[tuple[str, str], float],
    systems: list[str],
    metric_name: str,
    confidence_intervals: dict[tuple[str, str], tuple[float, float]] | None = None,
) -> str:
    """
    Format pairwise matrix as LaTeX table.
    
    Args:
        averaged_matrix: Dictionary mapping (system1, system2) tuples to win percentages
        systems: List of systems in desired order
        metric_name: Name of the metric for the header
        confidence_intervals: Optional dictionary mapping (system1, system2) tuples to (lower, upper) CI bounds
    
    Returns:
        LaTeX-formatted table string
    """
    lines = []
    lines.append(r"\footnotesize")
    
    # Build column specification: X for system names, r for each system column
    column_spec = "@{}X" + "r" * len(systems) + "@{}"
    lines.append(f"\\begin{{tabularx}}{{\\columnwidth}}{{{column_spec}}}")
    lines.append(r"\toprule")
    
    # Header row (no "System" label, no bold)
    header = ""
    for system in systems:
        latex_name = SYSTEM_LATEX_NAMES.get(system, system)
        header += f" & \\mbox{{{latex_name}}}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    
    # Data rows
    for system1 in systems:
        latex_name1 = SYSTEM_LATEX_NAMES.get(system1, system1)
        row = f"\\mbox{{{latex_name1}}}"
        for system2 in systems:
            if system1 == system2:
                row += " & -"
            else:
                percentage = averaged_matrix.get((system1, system2))
                if percentage is not None:
                    rounded_percentage = round(percentage)
                    padded_percentage = format_number_with_phantom(rounded_percentage, max_digits=3)
                    if confidence_intervals:
                        ci = confidence_intervals.get((system1, system2))
                        if ci is not None:
                            lower, upper = ci
                            half_width = round((upper - lower) / 2)
                            padded_half_width = format_number_with_phantom(half_width, max_digits=2)
                            row += f" & {padded_percentage} $\\pm$ {padded_half_width}"
                        else:
                            row += f" & {padded_percentage}"
                    else:
                        row += f" & {padded_percentage}"
                else:
                    row += " & -"
        row += r" \\"
        lines.append(row)
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    lines.append(r"\normalsize")
    
    return "\n".join(lines)


def get_systems_in_order(dataset: Dataset) -> list[str]:
    """
    Get systems present in dataset, in the specified order.
    
    Args:
        dataset: Hugging Face dataset
    
    Returns:
        List of systems in SYSTEM_ORDER that appear in the dataset
    """
    systems_in_data = set()
    for row in dataset:
        system1 = row.get("system1", "")
        system2 = row.get("system2", "")
        if system1:
            systems_in_data.add(system1)
        if system2:
            systems_in_data.add(system2)
    
    # Return systems in SYSTEM_ORDER that are present in data
    return [system for system in SYSTEM_ORDER if system in systems_in_data]


def main() -> None:
    """Main function to run pairwise comparison analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_pairwise_dataset_cli_arguments(parser)
    args = parser.parse_args()

    dotenv.load_dotenv()
    
    metrics = {
        "document_accuracy": "Document Accuracy",
        "segment_accuracy": "Segment Accuracy",
        "segment_fluency": "Segment Fluency",
    }
    
    # Map metric types to resampling units
    resampling_units = {
        "document_accuracy": "document",
        "segment_accuracy": "segment",
        "segment_fluency": "segment",
    }
    
    for metric_key, metric_name in metrics.items():
        print(f"Loading {metric_name}...")
        try:
            dataset = load_metric_dataset(
                args.dataset, metric_key, hub_revision=args.revision
            )
        except FileNotFoundError:
            print(
                f"Warning: no data for metric {metric_key!r} at {args.dataset!r}, "
                f"skipping {metric_name}"
            )
            continue
        
        # Get systems in order
        systems = get_systems_in_order(dataset)
        if not systems:
            print(f"No systems found in {metric_name}, skipping")
            continue
        
        print(f"Found systems: {systems}")
        
        # Calculate pairwise matrix per variety
        print(f"Calculating pairwise comparisons per variety for {metric_name}...")
        variety_matrices = calculate_pairwise_matrix_per_variety(dataset, metric_key)
        
        if not variety_matrices:
            print(f"No comparisons found for {metric_name}, skipping")
            continue
        
        # Average across varieties
        print(f"Averaging across {len(variety_matrices)} varieties...")
        averaged_matrix = average_across_varieties(variety_matrices, systems)
        
        # Calculate bootstrap confidence intervals
        resampling_unit = resampling_units[metric_key]
        print(f"Calculating bootstrap CIs for {metric_name} ({resampling_unit}-level)...")
        
        # Create a wrapper function that includes systems and metric_type
        def metric_function(ds: Dataset) -> dict[str, float]:
            return calculate_pairwise_win_percentages(ds, systems, metric_key)
        
        bootstrap_cis = bootstrap_confidence_interval(
            dataset,
            metric_function,
            resampling_unit=resampling_unit,
            n_resamples=1000,
            random_seed=42,
        )
        
        # Convert string keys back to tuple keys for display
        confidence_intervals: dict[tuple[str, str], tuple[float, float]] = {}
        for key, ci in bootstrap_cis.items():
            # Parse "system1_vs_system2" back to (system1, system2)
            if "_vs_" in key:
                parts = key.split("_vs_", 1)
                if len(parts) == 2:
                    system1, system2 = parts
                    confidence_intervals[(system1, system2)] = ci
        
        # Format LaTeX table
        table = format_latex_table(
            averaged_matrix,
            systems,
            metric_name,
            confidence_intervals=confidence_intervals,
        )
        print(table)
        
        # Write to output file if PAPER_DIR is set
        paper_dir = os.getenv("PAPER_DIR")
        if paper_dir is not None:
            output_path = Path(paper_dir) / "include" / f"pairwise_human_{metric_key}.tex"
            write_output(table, output_path)


if __name__ == "__main__":
    main()

