#!/usr/bin/env python3
"""
Create an aggregate LaTeX table for human evaluation statistics.
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics.dataloader import add_pairwise_dataset_cli_arguments, load_metric_dataset
from create_human_evaluation_statistics_table import (
    VARIETY_DISPLAY_NAMES,
    extract_person_id,
    extract_variety_from_language_pair,
    format_template,
    write_output,
)
from create_extended_inter_annotator_agreement import (
    MIN_OVERLAPPING_SEGMENTS,
    _compute_accuracy_metrics_for_variety,
    _compute_fluency_metrics_for_variety,
)


VARIETY_ORDER = [
    "rm-rumgr",
    "rm-sursilv",
    "rm-sutsilv",
    "rm-surmiran",
    "rm-puter",
    "rm-vallader",
]

VARIETY_TABLE_NAMES = {
    **VARIETY_DISPLAY_NAMES,
    "rm-rumgr": "RG",
}

VARIETY_TO_IAA_KEY = {
    "rm-rumgr": "rumgr",
    "rm-sursilv": "sursilv",
    "rm-sutsilv": "sutsilv",
    "rm-surmiran": "surmiran",
    "rm-puter": "puter",
    "rm-vallader": "vallader",
}

TEMPLATE = r"""
\footnotesize
\begin{tabularx}{\textwidth}{@{}Xr@{\hspace{1.5em}}rrr@{}}
\toprule
Variety & \# Raters & \# Fluency ratings & \# Acc. ratings (document-level) & \# Acc. preferences (segment-level) \\
\midrule
{variety_rows}
\bottomrule
\end{tabularx}
\normalsize
"""


def count_ratings_by_variety(dataset) -> dict[str, int]:
    """
    Count all non-empty system ratings by variety.

    Segment fluency and document accuracy rows contain ratings for up to two
    systems, so each non-null rating is one rating.
    """
    counts: dict[str, int] = defaultdict(int)

    for row in dataset:
        variety_code = extract_variety_from_language_pair(row.get("lp", ""))
        if not variety_code:
            continue

        if row.get("rating1") is not None:
            counts[variety_code] += 1
        if row.get("rating2") is not None:
            counts[variety_code] += 1

    return dict(counts)


def count_preferences_by_variety(dataset) -> dict[str, int]:
    """
    Count pairwise preference judgements by variety.

    A segment-level accuracy row compares two systems and therefore contributes
    one preference judgement when at least one side has a rating.
    """
    counts: dict[str, int] = defaultdict(int)

    for row in dataset:
        variety_code = extract_variety_from_language_pair(row.get("lp", ""))
        if not variety_code:
            continue

        if row.get("rating1") is not None or row.get("rating2") is not None:
            counts[variety_code] += 1

    return dict(counts)


def collect_raters_by_variety(*datasets) -> dict[str, set[str]]:
    """Collect distinct rater person IDs by variety across datasets."""
    raters: dict[str, set[str]] = defaultdict(set)

    for dataset in datasets:
        for row in dataset:
            variety_code = extract_variety_from_language_pair(row.get("lp", ""))
            annotator = row.get("annotator", "")
            if not variety_code or not annotator:
                continue
            raters[variety_code].add(extract_person_id(annotator))

    return dict(raters)


def format_count(count: int) -> str:
    """Format counts with a thousands separator for LaTeX."""
    return f"{count:,}"


def format_overlap(overlap: int, max_overlap: int) -> str:
    """Format a gray overlap count with phantom padding for alignment."""
    overlap_str = format_count(overlap)
    max_overlap_str = format_count(max_overlap)
    padding = ""
    if len(overlap_str) < len(max_overlap_str):
        padding = f"\\phantom{{{max_overlap_str[: len(max_overlap_str) - len(overlap_str)]}}}"
    return f"\\textcolor{{gray}}{{{padding}({overlap_str} overlap)}}"


def format_count_with_overlap(count: int, overlap: int, max_overlap: int) -> str:
    """Format a total count followed by its overlap count in parentheses."""
    return f"{format_count(count)}~{format_overlap(overlap, max_overlap)}"


def format_bold_count_with_overlap_phantom(count: int, overlap: int) -> str:
    """Format a bold count with normal-weight invisible overlap text."""
    return f"\\textbf{{{format_count(count)}}}~\\phantom{{({format_count(overlap)} overlap)}}"


def calculate_overlap_counts_by_variety(
    segment_fluency,
    document_accuracy,
    segment_accuracy,
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Calculate IAA overlap counts by variety using the extended IAA helpers."""
    fluency_overlaps = {}
    document_accuracy_overlaps = {}
    segment_accuracy_overlaps = {}

    for variety_code in VARIETY_ORDER:
        iaa_key = VARIETY_TO_IAA_KEY[variety_code]
        fluency_overlaps[variety_code] = _compute_fluency_metrics_for_variety(
            segment_fluency, iaa_key
        )["overlapping"]
        document_accuracy_overlaps[variety_code] = _compute_accuracy_metrics_for_variety(
            document_accuracy, iaa_key
        )["overlapping"]
        segment_accuracy_overlaps[variety_code] = _compute_accuracy_metrics_for_variety(
            segment_accuracy,
            iaa_key,
            item_id_key="segment_id",
            min_overlapping=MIN_OVERLAPPING_SEGMENTS,
        )["overlapping"]

    return fluency_overlaps, document_accuracy_overlaps, segment_accuracy_overlaps


def format_statistics(
    raters_by_variety: dict[str, set[str]],
    fluency_counts: dict[str, int],
    document_accuracy_counts: dict[str, int],
    segment_accuracy_counts: dict[str, int],
    fluency_overlaps: dict[str, int],
    document_accuracy_overlaps: dict[str, int],
    segment_accuracy_overlaps: dict[str, int],
) -> dict[str, str]:
    """Format aggregate statistics for the LaTeX template."""
    rows = []

    total_raters = 0
    total_fluency = 0
    total_document_accuracy = 0
    total_segment_accuracy = 0
    total_fluency_overlap = 0
    total_document_accuracy_overlap = 0
    total_segment_accuracy_overlap = 0
    max_fluency_overlap = sum(fluency_overlaps.values())
    max_document_accuracy_overlap = sum(document_accuracy_overlaps.values())
    max_segment_accuracy_overlap = sum(segment_accuracy_overlaps.values())

    for variety_code in VARIETY_ORDER:
        display_name = VARIETY_TABLE_NAMES[variety_code]
        raters = raters_by_variety.get(variety_code, set())
        fluency = fluency_counts.get(variety_code, 0)
        document_accuracy = document_accuracy_counts.get(variety_code, 0)
        segment_accuracy = segment_accuracy_counts.get(variety_code, 0)
        fluency_overlap = fluency_overlaps.get(variety_code, 0)
        document_accuracy_overlap = document_accuracy_overlaps.get(variety_code, 0)
        segment_accuracy_overlap = segment_accuracy_overlaps.get(variety_code, 0)

        total_raters += len(raters)
        total_fluency += fluency
        total_document_accuracy += document_accuracy
        total_segment_accuracy += segment_accuracy
        total_fluency_overlap += fluency_overlap
        total_document_accuracy_overlap += document_accuracy_overlap
        total_segment_accuracy_overlap += segment_accuracy_overlap

        rows.append(
            f"{display_name} & {len(raters)} & "
            f"{format_count_with_overlap(fluency, fluency_overlap, max_fluency_overlap)} & "
            f"{format_count_with_overlap(document_accuracy, document_accuracy_overlap, max_document_accuracy_overlap)} & "
            f"{format_count_with_overlap(segment_accuracy, segment_accuracy_overlap, max_segment_accuracy_overlap)} \\\\"
        )

    rows.append("\\midrule")
    rows.append(
        f"\\textbf{{Total}} & \\textbf{{{total_raters}}} & "
        f"{format_bold_count_with_overlap_phantom(total_fluency, total_fluency_overlap)} & "
        f"{format_bold_count_with_overlap_phantom(total_document_accuracy, total_document_accuracy_overlap)} & "
        f"{format_bold_count_with_overlap_phantom(total_segment_accuracy, total_segment_accuracy_overlap)} \\\\"
    )

    return {"variety_rows": "\n".join(rows)}


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

    print("Counting raters...")
    raters_by_variety = collect_raters_by_variety(
        segment_fluency, document_accuracy, segment_accuracy
    )

    print("Counting fluency ratings...")
    fluency_counts = count_ratings_by_variety(segment_fluency)

    print("Counting document-level accuracy ratings...")
    document_accuracy_counts = count_ratings_by_variety(document_accuracy)

    print("Counting segment-level accuracy preferences...")
    segment_accuracy_counts = count_preferences_by_variety(segment_accuracy)

    print("Calculating overlap counts...")
    (
        fluency_overlaps,
        document_accuracy_overlaps,
        segment_accuracy_overlaps,
    ) = calculate_overlap_counts_by_variety(
        segment_fluency, document_accuracy, segment_accuracy
    )

    statistics = format_statistics(
        raters_by_variety,
        fluency_counts,
        document_accuracy_counts,
        segment_accuracy_counts,
        fluency_overlaps,
        document_accuracy_overlaps,
        segment_accuracy_overlaps,
    )
    filled_template = format_template(TEMPLATE, statistics)

    print(filled_template)

    dotenv.load_dotenv()
    paper_dir = os.getenv("PAPER_DIR")
    if paper_dir is not None:
        output_path = (
            Path(paper_dir) / "include" / "human_evaluation_aggregate_statistics.tex"
        )
        write_output(filled_template, output_path)


if __name__ == "__main__":
    main()
