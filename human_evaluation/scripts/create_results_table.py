#!/usr/bin/env python3
"""
Create LaTeX table for human evaluation results.
"""

import argparse
import os
import sys
from pathlib import Path

import dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics.bootstrap import bootstrap_confidence_interval
from metrics.dataloader import add_pairwise_dataset_cli_arguments, load_metric_dataset
from metrics.sqm import calculate_sqm_scores


TEMPLATE = r"""
\footnotesize
\begin{tabularx}{\columnwidth}{@{}Xrr@{}}
\toprule
\textbf{System} & \textbf{DE$\rightarrow$RM~} \textbf{Fluency} & \textbf{Accuracy} \\
\midrule
Human reference & {reference_fluency_sqm} & {reference_accuracy_sqm} \\
\midrule
\mbox{Gemini 3 Pro (preview)} & {gemini_3_pro_fluency_sqm} & {gemini_3_pro_accuracy_sqm} \\
\midrule
\mbox{\textit{Fine-tuned NLLB}} & & \\[0.2em]
\mbox{LR$\rightarrow$HR augmentation} & {back_translation_fluency_sqm} & {back_translation_accuracy_sqm} \\
\mbox{\quad + dictionary prompting} & {dict_prompting_fluency_sqm} & {dict_prompting_accuracy_sqm} \\
\bottomrule
\end{tabularx}
\normalsize
"""


def format_template(
    template: str,
    scores: dict[str, float | None],
    confidence_intervals: dict[str, tuple[float, float] | None],
) -> str:
    """
    Format LaTeX template with scores and confidence intervals.
    
    Args:
        template: LaTeX template string with placeholders like {key}
        scores: Dictionary mapping placeholder keys to score values (or None)
        confidence_intervals: Dictionary mapping placeholder keys to (lower, upper) bounds
    
    Returns:
        Formatted template string
    """
    filled_template = template
    for key, score in scores.items():
        placeholder = f"{{{key}}}"
        confidence_interval = confidence_intervals.get(key)
        
        if score is None:
            formatted_score = "tba $\\pm$ tba"
        elif confidence_interval is None:
            formatted_score = f"{score:.2f} $\\pm$ tba"
        else:
            lower, upper = confidence_interval
            half_width = (upper - lower) / 2
            formatted_score = f"{score:.2f} $\\pm$ {half_width:.2f}"
        
        filled_template = filled_template.replace(placeholder, formatted_score)
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
    
    # System name mappings: data name -> template key prefix
    system_mappings = {
        "reference": "reference",
        "Gemini-3-Pro": "gemini_3_pro",
        "romansh-nllb-1.3b": "back_translation",
        "romansh-nllb-1.3b-dict-prompting": "dict_prompting",
    }
    
    # Calculate scores for each system
    print("Calculating fluency SQM scores...")
    fluency_sqm_scores = calculate_sqm_scores(segment_fluency)
    
    print("Calculating accuracy SQM scores...")
    accuracy_sqm_scores = calculate_sqm_scores(document_accuracy)
    
    # Calculate bootstrap confidence intervals
    print("Calculating bootstrap CIs for fluency SQM (segment-level)...")
    fluency_sqm_confidence_intervals = bootstrap_confidence_interval(
        segment_fluency,
        calculate_sqm_scores,
        resampling_unit="segment",
        n_resamples=1000,
        random_seed=42,
    )
    
    print("Calculating bootstrap CIs for accuracy SQM (document-level)...")
    accuracy_sqm_confidence_intervals = bootstrap_confidence_interval(
        document_accuracy,
        calculate_sqm_scores,
        resampling_unit="document",
        n_resamples=1000,
        random_seed=42,
    )
    
    # Map scores and CIs to template keys
    scores = {}
    confidence_intervals = {}
    
    for system_name, key_prefix in system_mappings.items():
        scores[f"{key_prefix}_fluency_sqm"] = fluency_sqm_scores.get(system_name)
        scores[f"{key_prefix}_accuracy_sqm"] = accuracy_sqm_scores.get(system_name)
        
        confidence_intervals[f"{key_prefix}_fluency_sqm"] = fluency_sqm_confidence_intervals.get(system_name)
        confidence_intervals[f"{key_prefix}_accuracy_sqm"] = accuracy_sqm_confidence_intervals.get(system_name)
    
    # Format scores for LaTeX
    filled_template = format_template(TEMPLATE, scores, confidence_intervals)
    
    print(filled_template)
    
    # Write to output file if PAPER_DIR is set
    dotenv.load_dotenv()
    paper_dir = os.getenv("PAPER_DIR")
    if paper_dir is not None:
        output_path = Path(paper_dir) / "include" / "results_human.tex"
        write_output(filled_template, output_path)


if __name__ == "__main__":
    main()
