#!/usr/bin/env python3
"""
Extended inter-annotator agreement analysis for segment-level fluency,
document-level accuracy, and segment-level accuracy.

Computes Pearson correlation, polarity agreement, and ranking Spearman for
all Romansh varieties with sufficient overlap, and writes LaTeX tables.
"""

import argparse
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import dotenv
from datasets import Dataset
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics.dataloader import add_pairwise_dataset_cli_arguments, load_metric_dataset

VARIETIES = ["rumgr", "sursilv", "sutsilv", "surmiran", "puter", "vallader"]
VARIETY_ABBREVIATIONS = {
    "rumgr": "RG",
    "sursilv": "Surs.",
    "sutsilv": "Suts.",
    "surmiran": "Surm.",
    "puter": "Puter",
    "vallader": "Vall.",
}
MIN_OVERLAPPING_SEGMENTS = 90
MIN_OVERLAPPING_DOCS = 15


def filter_by_variety(dataset: Dataset, variety: str) -> Dataset:
    """Filter dataset to rows matching the given Romansh variety."""
    return dataset.filter(lambda row: row["lp"].endswith(f"rm-{variety}"))


def select_top_two_annotators(dataset: Dataset) -> tuple[str, str]:
    """
    Select the two annotators with the most annotations.

    Raises ValueError if fewer than two annotators are present.
    """
    annotator_counts = Counter(dataset["annotator"])
    if len(annotator_counts) < 2:
        raise ValueError(
            f"Need at least 2 annotators, found {len(annotator_counts)}: "
            f"{dict(annotator_counts)}"
        )
    top_two = annotator_counts.most_common(2)
    return top_two[0][0], top_two[1][0]


def find_overlapping_pairs(
    dataset: Dataset,
    annotator_a: str,
    annotator_b: str,
    item_id_key: str = "document_id",
) -> list[tuple[dict, dict]]:
    """
    Find item/system-pair comparisons where both annotators provided ratings.

    Groups rows by (item_id, system1, system2) and returns pairs of rows
    (one per annotator) for each overlapping comparison.
    """
    rows_by_key: dict[tuple, dict[str, dict]] = defaultdict(dict)

    for row in dataset:
        key = (row[item_id_key], row["system1"], row["system2"])
        annotator = row["annotator"]
        if annotator in (annotator_a, annotator_b):
            rows_by_key[key][annotator] = row

    pairs = []
    for annotator_rows in rows_by_key.values():
        if annotator_a in annotator_rows and annotator_b in annotator_rows:
            pairs.append((annotator_rows[annotator_a], annotator_rows[annotator_b]))

    return pairs


def calculate_pearson(pairs: list[tuple[dict, dict]]) -> float | None:
    """
    Pearson correlation of normalized ratings between two annotators.

    For each overlapping document, both rating1 and rating2 contribute
    a paired observation.
    """
    scores_a = []
    scores_b = []

    for row_a, row_b in pairs:
        for rating_key in ("rating1", "rating2"):
            value_a = row_a.get(rating_key)
            value_b = row_b.get(rating_key)
            if value_a is not None and value_b is not None:
                scores_a.append(value_a)
                scores_b.append(value_b)

    if len(scores_a) < 2:
        return None

    correlation, _ = stats.pearsonr(scores_a, scores_b)
    return float(correlation)


def _polarity(row: dict) -> str | None:
    """Determine which system wins based on raw ratings, or tie."""
    raw1 = row.get("rating1_raw")
    raw2 = row.get("rating2_raw")
    if raw1 is None or raw2 is None:
        return None
    if raw1 > raw2:
        return "system1"
    if raw2 > raw1:
        return "system2"
    return "tie"


def calculate_polarity_agreement(pairs: list[tuple[dict, dict]]) -> float | None:
    """
    Percentage of overlapping documents where both annotators agree on
    which system is better (or that they are equal).
    """
    agreements = 0
    total = 0

    for row_a, row_b in pairs:
        polarity_a = _polarity(row_a)
        polarity_b = _polarity(row_b)
        if polarity_a is None or polarity_b is None:
            continue
        total += 1
        if polarity_a == polarity_b:
            agreements += 1

    if total == 0:
        return None

    return (agreements / total) * 100


@dataclass
class RankingResult:
    correlation: float
    systems: list[str]
    averages_a: list[float]
    averages_b: list[float]


def calculate_ranking_spearman_all(
    dataset: Dataset,
    annotator_a: str,
    annotator_b: str,
) -> RankingResult | None:
    """
    Spearman correlation of per-system average normalized ratings.

    Uses all document-level accuracy annotations by both annotators (not just
    overlapping documents). For each system, average its normalized ratings
    across all documents each annotator rated, then compute Spearman between
    the two annotators' vectors of system averages.
    """
    system_ratings_a: dict[str, list[float]] = defaultdict(list)
    system_ratings_b: dict[str, list[float]] = defaultdict(list)

    for row in dataset:
        annotator = row["annotator"]
        if annotator not in (annotator_a, annotator_b):
            continue
        for system_key, rating_key in (("system1", "rating1"), ("system2", "rating2")):
            system = row[system_key]
            value = row.get(rating_key)
            if value is not None:
                if annotator == annotator_a:
                    system_ratings_a[system].append(value)
                else:
                    system_ratings_b[system].append(value)

    systems = sorted(system_ratings_a.keys() & system_ratings_b.keys())

    if len(systems) < 3:
        return None

    averages_a = [sum(system_ratings_a[s]) / len(system_ratings_a[s]) for s in systems]
    averages_b = [sum(system_ratings_b[s]) / len(system_ratings_b[s]) for s in systems]

    correlation, _ = stats.spearmanr(averages_a, averages_b)
    return RankingResult(
        correlation=float(correlation),
        systems=systems,
        averages_a=averages_a,
        averages_b=averages_b,
    )


def _format_for_latex(
    value: float | None,
    *,
    is_percentage: bool = False,
    as_percent_scale: bool = False,
) -> str:
    """
    Format a metric value for LaTeX table cell.

    as_percent_scale: Format with one decimal. Values in [-1, 1] (correlations)
    are scaled by 100; values in [0, 100] (agreement rate) are used as-is.
    """
    if value is None:
        return "---"
    if is_percentage:
        return f"{value:.1f}\\%"
    if as_percent_scale:
        display = value * 100 if abs(value) <= 1 else value
        return f"{display:.1f}"
    return f"{value:.2f}"


def _compute_fluency_metrics_for_variety(
    filtered_dataset: Dataset,
    variety: str,
) -> dict:
    """
    Compute fluency metrics for a variety. Always returns a dict.
    """
    filtered_filtered = filter_by_variety(filtered_dataset, variety)
    if len(filtered_filtered) == 0:
        return {"overlapping": 0, "pearson": None, "polarity": None, "ranking": None}
    try:
        annotator_a, annotator_b = select_top_two_annotators(filtered_filtered)
    except ValueError:
        return {"overlapping": 0, "pearson": None, "polarity": None, "ranking": None}
    pairs = find_overlapping_pairs(
        filtered_filtered, annotator_a, annotator_b, item_id_key="segment_id"
    )
    overlapping = len(pairs)
    pearson = calculate_pearson(pairs)
    polarity = calculate_polarity_agreement(pairs)
    ranking_result = calculate_ranking_spearman_all(
        filtered_filtered, annotator_a, annotator_b
    )
    ranking = ranking_result.correlation if ranking_result else None
    return {
        "overlapping": overlapping,
        "pearson": pearson,
        "polarity": polarity,
        "ranking": ranking,
    }


def _compute_accuracy_metrics_for_variety(
    filtered_dataset: Dataset,
    variety: str,
    item_id_key: str = "document_id",
    min_overlapping: int = MIN_OVERLAPPING_DOCS,
) -> dict:
    """
    Compute accuracy metrics for a variety. Always returns a dict.
    """
    filtered_filtered = filter_by_variety(filtered_dataset, variety)
    if len(filtered_filtered) == 0:
        return {"overlapping": 0, "pearson": None, "polarity": None, "ranking": None}
    try:
        annotator_a, annotator_b = select_top_two_annotators(filtered_filtered)
    except ValueError:
        return {"overlapping": 0, "pearson": None, "polarity": None, "ranking": None}
    pairs = find_overlapping_pairs(
        filtered_filtered, annotator_a, annotator_b, item_id_key=item_id_key
    )
    overlapping = len(pairs)
    if overlapping < min_overlapping:
        return {
            "overlapping": overlapping,
            "pearson": None,
            "polarity": None,
            "ranking": None,
        }
    pearson = calculate_pearson(pairs)
    polarity = calculate_polarity_agreement(pairs)
    ranking_result = calculate_ranking_spearman_all(
        filtered_filtered, annotator_a, annotator_b
    )
    ranking = ranking_result.correlation if ranking_result else None
    return {
        "overlapping": overlapping,
        "pearson": pearson,
        "polarity": polarity,
        "ranking": ranking,
    }


def _build_latex_table(
    metrics_by_variety: dict[str, dict],
    overlapping_column_header: str,
) -> str:
    """Build a LaTeX table with metrics as rows and varieties as columns."""
    column_headers = " & ".join(
        f"\\textbf{{{VARIETY_ABBREVIATIONS[v]}}}" for v in VARIETIES
    )
    table_rows = []

    percent_format = {"as_percent_scale": True}
    for metric_key, metric_label, format_args in [
        ("overlapping", overlapping_column_header, {}),
        ("pearson", "Item-level Pearson correlation", percent_format),
        ("polarity", "Pairwise comparison agreement rate", percent_format),
        ("ranking", "System-level Spearman correlation", percent_format),
    ]:
        cells = [metric_label]
        for variety in VARIETIES:
            metrics = metrics_by_variety.get(variety, {})
            above_threshold = metrics.get("pearson") is not None

            if metric_key == "overlapping":
                if above_threshold:
                    cells.append(str(metrics.get("overlapping", 0)))
                else:
                    cells.append(f"({metrics.get('overlapping', 0)})")
            else:
                value = metrics.get(metric_key) if above_threshold else None
                cells.append(_format_for_latex(value, **format_args))
        table_rows.append(" & ".join(cells) + " \\\\")

    body = table_rows[0] + "\n\\midrule\n" + "\n".join(table_rows[1:])
    return rf"""
\footnotesize
\begin{{tabularx}}{{0.8\columnwidth}}{{@{{}}Xrrrrrr@{{}}}}
\toprule
\textbf{{Metric}} & {column_headers} \\
\midrule
{body}
\bottomrule
\end{{tabularx}}
\normalsize
"""


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

    fluency_by_variety: dict[str, dict] = {}
    accuracy_by_variety: dict[str, dict] = {}
    segment_accuracy_by_variety: dict[str, dict] = {}

    for variety in VARIETIES:
        fluency_by_variety[variety] = _compute_fluency_metrics_for_variety(
            segment_fluency, variety
        )
        accuracy_by_variety[variety] = _compute_accuracy_metrics_for_variety(
            document_accuracy, variety
        )
        segment_accuracy_by_variety[variety] = _compute_accuracy_metrics_for_variety(
            segment_accuracy,
            variety,
            item_id_key="segment_id",
            min_overlapping=MIN_OVERLAPPING_SEGMENTS,
        )

    fluency_table = _build_latex_table(
        fluency_by_variety,
        overlapping_column_header="Number of overlapping segments",
    )
    accuracy_table = _build_latex_table(
        accuracy_by_variety,
        overlapping_column_header="Number of overlapping documents",
    )
    segment_accuracy_table = _build_latex_table(
        segment_accuracy_by_variety,
        overlapping_column_header="Number of overlapping segments",
    )

    print("\n=== Segment-level fluency ===")
    print(fluency_table)
    print("\n=== Document-level accuracy ===")
    print(accuracy_table)
    print("\n=== Segment-level accuracy ===")
    print(segment_accuracy_table)

    dotenv.load_dotenv()
    paper_dir = os.getenv("PAPER_DIR")
    if paper_dir is not None:
        output_dir = Path(paper_dir) / "include"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "results_extended_iaa_fluency.tex").write_text(
            fluency_table.strip(), encoding="utf-8"
        )
        (output_dir / "results_extended_iaa_accuracy.tex").write_text(
            accuracy_table.strip(), encoding="utf-8"
        )
        (output_dir / "results_extended_iaa_segment_accuracy.tex").write_text(
            segment_accuracy_table.strip(), encoding="utf-8"
        )
        print(f"\nResults written to {output_dir}/results_extended_iaa_fluency.tex")
        print(f"Results written to {output_dir}/results_extended_iaa_accuracy.tex")
        print(f"Results written to {output_dir}/results_extended_iaa_segment_accuracy.tex")


if __name__ == "__main__":
    main()
