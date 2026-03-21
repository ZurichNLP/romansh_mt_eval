#!/usr/bin/env python3
"""
Create LaTeX table for human evaluation statistics.
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics.dataloader import add_pairwise_dataset_cli_arguments, load_metric_dataset

SYSTEM_LATEX_NAMES = {
    "reference": "Human reference",
    "Gemini-3-Pro": "Gemini 3 Pro (preview)",
    "romansh-nllb-1.3b": "LR$\\rightarrow$HR augmentation",
    "romansh-nllb-1.3b-dict-prompting": "+ dictionary prompting",
}

# Variety codes to full display names
VARIETY_DISPLAY_NAMES = {
    "rm-rumgr": "Rumantsch Grischun",
    "rm-sursilv": "Sursilvan",
    "rm-sutsilv": "Sutsilvan",
    "rm-surmiran": "Surmiran",
    "rm-puter": "Puter",
    "rm-vallader": "Vallader",
}

TEMPLATE = r"""
\footnotesize
\begin{tabularx}{\columnwidth}{@{}Xrrrr@{}}
\toprule
Variety & Human reference & Gemini 3 Pro (preview) & LR$\rightarrow$HR augmentation & + dictionary prompting \\
\midrule
{variety_rows}
\bottomrule
\end{tabularx}
\normalsize
"""


def extract_variety_from_language_pair(language_pair: str) -> str | None:
    """
    Extract variety code from language pair string.
    
    Args:
        language_pair: Language pair string like "de_DE-rm-rumgr"
    
    Returns:
        Variety code like "rm-rumgr" or None if not found
    """
    if not language_pair:
        return None
    
    parts = language_pair.split("-")
    if len(parts) >= 3:
        # Extract the target language part (e.g., "rm-rumgr" from "de_DE-rm-rumgr")
        variety_part = "-".join(parts[-2:])
        if variety_part in VARIETY_DISPLAY_NAMES:
            return variety_part
    
    return None


def extract_person_id(annotator: str) -> str:
    """Extract person identifier: last 2 digits of username (e.g. 03 from deurgh0503)."""
    return annotator[-2:] if len(annotator) >= 2 else annotator


def count_annotations_by_system_and_variety(dataset, count_pairs: bool = False) -> dict[str, dict[str, float]]:
    """
    Count annotation rows for each system/variety combination.
    
    Each row represents one annotator's rating, so multiple annotators
    rating the same system-segment pair are counted separately.

    Args:
        dataset: Hugging Face dataset with 'lp', 'system1', 'system2', 'rating1', 'rating2', 'document_id' columns
        count_pairs: If True, count pairs (increment by 0.5 per system) instead of individual systems (increment by 1.0)
    
    Returns:
        Dictionary mapping variety codes to dictionaries mapping system names to counts
    """
    counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    increment = 0.5 if count_pairs else 1.0
    
    for row in dataset:
        language_pair = row.get('lp', '')
        variety_code = extract_variety_from_language_pair(language_pair)
        
        if not variety_code:
            continue
        
        system1 = row.get('system1', '')
        system2 = row.get('system2', '')
        rating1 = row.get('rating1')
        rating2 = row.get('rating2')
        
        # Count system1 if it has a rating
        if system1 and rating1 is not None:
            counts[variety_code][system1] += increment
        
        # Count system2 if it has a rating
        if system2 and rating2 is not None:
            counts[variety_code][system2] += increment
    
    return dict(counts)


def count_annotations_by_system_variety_and_annotator(
    dataset, count_pairs: bool = False
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Count annotation rows for each system/variety/annotator combination.

    Same coverage as count_annotations_by_system_and_variety, but groups by annotator.

    Args:
        dataset: Hugging Face dataset with 'lp', 'system1', 'system2', 'rating1',
            'rating2', 'document_id', 'annotator' columns
        count_pairs: If True, count pairs (increment by 0.5 per system) instead of
            individual systems (increment by 1.0)

    Returns:
        Dictionary mapping variety -> annotator -> system -> count
    """
    counts: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    increment = 0.5 if count_pairs else 1.0

    for row in dataset:
        language_pair = row.get("lp", "")
        variety_code = extract_variety_from_language_pair(language_pair)

        if not variety_code:
            continue

        annotator = row.get("annotator", "")
        if not annotator:
            continue

        system1 = row.get("system1", "")
        system2 = row.get("system2", "")
        rating1 = row.get("rating1")
        rating2 = row.get("rating2")

        if system1 and rating1 is not None:
            counts[variety_code][annotator][system1] += increment
        if system2 and rating2 is not None:
            counts[variety_code][annotator][system2] += increment

    return dict(counts)


def format_with_phantom(count: float, max_digits: int = 3) -> str:
    """Format number with invisible padding using LaTeX \\phantom."""
    count_int = int(round(count))
    count_str = str(count_int)
    if max_digits == 2:
        return f"\\phantom{{0}}{count_str}" if len(count_str) == 1 else count_str
    if len(count_str) == 1:
        return f"\\phantom{{00}}{count_str}"
    if len(count_str) == 2:
        return f"\\phantom{{0}}{count_str}"
    return count_str


def format_statistics(
    fluency_counts: dict[str, dict[str, float]],
    document_accuracy_counts: dict[str, dict[str, float]],
    segment_accuracy_counts: dict[str, dict[str, float]],
    fluency_per_annotator: dict[str, dict[str, dict[str, float]]],
    document_accuracy_per_annotator: dict[str, dict[str, dict[str, float]]],
    segment_accuracy_per_annotator: dict[str, dict[str, dict[str, float]]],
) -> dict[str, str]:
    """
    Format statistics for LaTeX template.

    Varieties are rows, systems are columns. Each variety has a total row followed
    by per-annotator rows (Rater 1, Rater 2, ...), with numbers incrementing across
    varieties.
    """
    system_order = [
        "reference",
        "Gemini-3-Pro",
        "romansh-nllb-1.3b",
        "romansh-nllb-1.3b-dict-prompting",
    ]
    variety_order = ["rm-rumgr", "rm-sursilv", "rm-sutsilv", "rm-surmiran", "rm-puter", "rm-vallader"]

    def format_cells(
        fluency: dict[str, float],
        document: dict[str, float],
        segment: dict[str, float],
    ) -> list[str]:
        cells = []
        for system_data_name in system_order:
            fluency_count = fluency.get(system_data_name, 0)
            document_count = document.get(system_data_name, 0)
            segment_count = segment.get(system_data_name, 0)
            fluency_str = format_with_phantom(fluency_count, max_digits=3)
            document_str = format_with_phantom(document_count, max_digits=2)
            segment_str = format_with_phantom(segment_count, max_digits=3)
            cells.append(f"{fluency_str} / {document_str} / {segment_str}")
        return cells

    variety_rows = []
    rater_number = 1

    for variety_code in variety_order:
        display_name = VARIETY_DISPLAY_NAMES[variety_code]

        # Total row
        fluency_total = fluency_counts.get(variety_code, {})
        document_total = document_accuracy_counts.get(variety_code, {})
        segment_total = segment_accuracy_counts.get(variety_code, {})
        cells = format_cells(fluency_total, document_total, segment_total)
        bold_cells = [f"\\textbf{{{cell}}}" for cell in cells]
        variety_rows.append(f"\\textbf{{{display_name}}} & {' & '.join(bold_cells)} \\\\")

        # Collect annotators for this variety from all three datasets
        annotators = set()
        annotators.update(fluency_per_annotator.get(variety_code, {}).keys())
        annotators.update(document_accuracy_per_annotator.get(variety_code, {}).keys())
        annotators.update(segment_accuracy_per_annotator.get(variety_code, {}).keys())

        # Group annotators by person ID (last 2 digits of username) and merge counts
        person_counts: dict[str, tuple[dict[str, float], dict[str, float], dict[str, float]]] = {}
        for annotator in annotators:
            person_id = extract_person_id(annotator)
            fluency_ann = fluency_per_annotator.get(variety_code, {}).get(annotator, {})
            document_ann = document_accuracy_per_annotator.get(variety_code, {}).get(annotator, {})
            segment_ann = segment_accuracy_per_annotator.get(variety_code, {}).get(annotator, {})

            if person_id not in person_counts:
                person_counts[person_id] = (
                    defaultdict(float),
                    defaultdict(float),
                    defaultdict(float),
                )
            fluency_merged, document_merged, segment_merged = person_counts[person_id]
            for system, count in fluency_ann.items():
                fluency_merged[system] += count
            for system, count in document_ann.items():
                document_merged[system] += count
            for system, count in segment_ann.items():
                segment_merged[system] += count

        # Order person IDs by total annotation count (descending)
        def total_count(person_id: str) -> float:
            fluency, document, segment = person_counts[person_id]
            return sum(fluency.values()) + sum(document.values()) + sum(segment.values())

        sorted_person_ids = sorted(person_counts.keys(), key=total_count, reverse=True)

        # Per-person rows
        for person_id in sorted_person_ids:
            fluency_merged, document_merged, segment_merged = person_counts[person_id]
            cells = format_cells(dict(fluency_merged), dict(document_merged), dict(segment_merged))
            variety_rows.append(f"\\hspace{{1em}}Rater {rater_number} & {' & '.join(cells)} \\\\")
            rater_number += 1

        if variety_code != variety_order[-1]:
            variety_rows.append("\\midrule")

    return {"variety_rows": "\n".join(variety_rows)}


def format_template(template: str, statistics: dict[str, str]) -> str:
    """
    Format LaTeX template with statistics.
    
    Args:
        template: LaTeX template string with placeholders like {key}
        statistics: Dictionary mapping placeholder keys to formatted statistics strings
    
    Returns:
        Formatted template string
    """
    filled_template = template
    for key, value in statistics.items():
        placeholder = f"{{{key}}}"
        filled_template = filled_template.replace(placeholder, value)
    return filled_template


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_pairwise_dataset_cli_arguments(parser)
    args = parser.parse_args()

    print("Loading datasets...")
    segment_fluency = load_metric_dataset(
        args.dataset, "segment_fluency", hub_revision=args.revision
    )
    document_accuracy = load_metric_dataset(
        args.dataset, "document_accuracy", hub_revision=args.revision
    )
    segment_accuracy = load_metric_dataset(
        args.dataset, "segment_accuracy", hub_revision=args.revision
    )
    
    # Count annotations for each system/variety combination
    print("Counting fluency segment annotations...")
    fluency_counts = count_annotations_by_system_and_variety(segment_fluency)
    fluency_per_annotator = count_annotations_by_system_variety_and_annotator(
        segment_fluency
    )

    print("Counting document accuracy annotations...")
    document_accuracy_counts = count_annotations_by_system_and_variety(
        document_accuracy
    )
    document_accuracy_per_annotator = (
        count_annotations_by_system_variety_and_annotator(document_accuracy)
    )

    print("Counting segment accuracy annotations...")
    segment_accuracy_counts = count_annotations_by_system_and_variety(
        segment_accuracy, count_pairs=True
    )
    segment_accuracy_per_annotator = (
        count_annotations_by_system_variety_and_annotator(
            segment_accuracy, count_pairs=True
        )
    )

    # Format statistics for LaTeX
    statistics = format_statistics(
        fluency_counts,
        document_accuracy_counts,
        segment_accuracy_counts,
        fluency_per_annotator,
        document_accuracy_per_annotator,
        segment_accuracy_per_annotator,
    )
    
    # Format template
    filled_template = format_template(TEMPLATE, statistics)
    
    print(filled_template)
    
    # Write to output file if PAPER_DIR is set
    dotenv.load_dotenv()
    paper_dir = os.getenv("PAPER_DIR")
    if paper_dir is not None:
        output_path = Path(paper_dir) / "include" / "human_evaluation_statistics.tex"
        write_output(filled_template, output_path)


if __name__ == "__main__":
    main()

