#!/usr/bin/env python3
"""
Create LaTeX tables for human evaluation results per variety.
"""

import argparse
import os
import sys
from pathlib import Path

import dotenv
import numpy as np
from datasets import Dataset

# Add parent directory to path to import metrics
sys.path.insert(0, str(Path(__file__).parent.parent))
from metrics.bootstrap import bootstrap_confidence_interval
from metrics.dataloader import add_pairwise_dataset_cli_arguments, load_metric_dataset


ACCURACY_TEMPLATE = r"""
\footnotesize
\begin{tabularx}{\columnwidth}{@{}Xrrrrrr@{}}
\toprule
\textbf{System} & \textbf{RG} & \textbf{Surs.} & \textbf{Suts.} & \textbf{Surm.} & \textbf{Puter} & \textbf{Vall.} \\
\midrule
Human reference & {reference_RG} & {reference_Surs} & {reference_Suts} & {reference_Surm} & {reference_Puter} & {reference_Vall} \\
\midrule
Gemini 3 Pro (preview) & {gemini_3_pro_RG} & {gemini_3_pro_Surs} & {gemini_3_pro_Suts} & {gemini_3_pro_Surm} & {gemini_3_pro_Puter} & {gemini_3_pro_Vall} \\
\mbox{LR$\rightarrow$HR NLLB} & {back_translation_RG} & {back_translation_Surs} & {back_translation_Suts} & {back_translation_Surm} & {back_translation_Puter} & {back_translation_Vall} \\
\quad + dict prompting & {dict_prompting_RG} & {dict_prompting_Surs} & {dict_prompting_Suts} & {dict_prompting_Surm} & {dict_prompting_Puter} & {dict_prompting_Vall} \\
\bottomrule
\end{tabularx}
\normalsize
"""

FLUENCY_TEMPLATE = r"""
\footnotesize
\begin{tabularx}{\columnwidth}{@{}Xrrrrrr@{}}
\toprule
\textbf{System} & \textbf{RG} & \textbf{Surs.} & \textbf{Suts.} & \textbf{Surm.} & \textbf{Puter} & \textbf{Vall.} \\
\midrule
Human reference & {reference_RG} & {reference_Surs} & {reference_Suts} & {reference_Surm} & {reference_Puter} & {reference_Vall} \\
\midrule
Gemini 3 Pro (preview) & {gemini_3_pro_RG} & {gemini_3_pro_Surs} & {gemini_3_pro_Suts} & {gemini_3_pro_Surm} & {gemini_3_pro_Puter} & {gemini_3_pro_Vall} \\
\mbox{LR$\rightarrow$HR NLLB} & {back_translation_RG} & {back_translation_Surs} & {back_translation_Suts} & {back_translation_Surm} & {back_translation_Puter} & {back_translation_Vall} \\
\quad + dict prompting & {dict_prompting_RG} & {dict_prompting_Surs} & {dict_prompting_Suts} & {dict_prompting_Surm} & {dict_prompting_Puter} & {dict_prompting_Vall} \\
\bottomrule
\end{tabularx} 
\normalsize
"""


# Map language pair codes to LaTeX column names
LANGUAGE_PAIR_TO_VARIETY = {
    "rm-rumgr": "RG",
    "rm-sursilv": "Surs",
    "rm-sutsilv": "Suts",
    "rm-surmiran": "Surm",
    "rm-puter": "Puter",
    "rm-vallader": "Vall",
}


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
        if variety_part in LANGUAGE_PAIR_TO_VARIETY:
            return variety_part
    
    return None


def calculate_sqm_scores_by_variety(
    dataset: Dataset,
    count_documents: bool = False,
) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    """
    Calculate SQM scores per variety for all systems.

    Args:
        dataset: Hugging Face dataset with 'lp', 'system1', 'system2', 'rating1', 'rating2' columns
        count_documents: If True, count unique documents per variety. If False, count segments (rows) per variety.
    
    Returns:
        Tuple of:
        - Dictionary mapping variety codes to dictionaries mapping system names to scores
        - Dictionary mapping variety codes to total counts (segments or documents) across all system pairs
    """
    # Group scores by variety and track unique documents
    variety_scores = {}
    variety_documents = {}
    variety_total_counts = {}  # Track total counts per variety (across all system pairs)
    variety_documents_per_system = {}  # Track unique documents per system/variety (for document counting)
    all_systems = set()
    
    for row in dataset:
        language_pair = row.get('lp', '')
        variety_code = extract_variety_from_language_pair(language_pair)
        
        if not variety_code:
            continue
        
        system1 = row.get('system1', '')
        system2 = row.get('system2', '')
        rating1 = row.get('rating1')
        rating2 = row.get('rating2')
        document_id = row.get('document_id', '')
        
        all_systems.add(system1)
        all_systems.add(system2)
        
        if variety_code not in variety_scores:
            variety_scores[variety_code] = {}
            variety_documents[variety_code] = set()
            variety_total_counts[variety_code] = 0
            variety_documents_per_system[variety_code] = {}
        
        # Track documents for filtering (needed for both segment and document counting)
        if document_id:
            variety_documents[variety_code].add(document_id)
        
        # Count total segments/documents per variety (across all system pairs)
        if count_documents:
            # Count unique documents per variety (already tracked above)
            pass
        else:
            # Count total segments (rows) per variety
            variety_total_counts[variety_code] += 1
        
        # Collect scores for each system
        if system1 not in variety_scores[variety_code]:
            variety_scores[variety_code][system1] = []
        if system2 not in variety_scores[variety_code]:
            variety_scores[variety_code][system2] = []
        
        # Collect ratings for score calculation
        if rating1 is not None:
            variety_scores[variety_code][system1].append(rating1)
        if rating2 is not None:
            variety_scores[variety_code][system2].append(rating2)
    
    if not variety_scores or not all_systems:
        return {}, {}
    
    varieties_to_process = list(variety_documents.keys())
    
    if not varieties_to_process:
        return {}, {}
    
    # Calculate mean per variety for each system
    scores_by_variety = {}
    for variety in varieties_to_process:
        scores_by_variety[variety] = {}
        for system in all_systems:
            scores = variety_scores[variety].get(system, [])
            if scores:
                scores_by_variety[variety][system] = np.mean(scores).item()
    
    # Set total counts per variety
    counts_by_variety = {}
    for variety in varieties_to_process:
        if count_documents:
            # Count unique documents per variety
            counts_by_variety[variety] = len(variety_documents[variety])
        else:
            # Use segment count
            counts_by_variety[variety] = variety_total_counts[variety]
    
    return scores_by_variety, counts_by_variety


def calculate_sqm_scores_single_variety(dataset: Dataset) -> dict[str, float]:
    """
    Calculate SQM scores for a dataset filtered to a single variety.
    
    Used for bootstrap resamples that may contain fewer rows than the full
    exported dataset for that variety.
    
    Args:
        dataset: Hugging Face dataset filtered to a single variety
    
    Returns:
        Dictionary mapping system names to scores
    """
    scores_by_variety, _ = calculate_sqm_scores_by_variety(
        dataset, count_documents=False
    )
    
    # Extract scores for the single variety (should only be one)
    if not scores_by_variety:
        return {}
    
    # Return scores for the first (and only) variety
    variety_code = next(iter(scores_by_variety))
    return scores_by_variety[variety_code]


def bootstrap_confidence_interval_by_variety(
    dataset: Dataset,
    resampling_unit: str,
    n_resamples: int = 1000,
    random_seed: int | None = None,
) -> dict[str, dict[str, tuple[float, float]]]:
    """
    Calculate bootstrap confidence intervals per variety.
    
    Args:
        dataset: Hugging Face dataset
        resampling_unit: "segment" or "document"
        n_resamples: Number of bootstrap samples
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary mapping variety codes to dictionaries mapping system names to (lower, upper) bounds
    """
    # Get unique varieties
    language_pairs = set(dataset["lp"])
    varieties = set()
    for lp in language_pairs:
        variety = extract_variety_from_language_pair(lp)
        if variety:
            varieties.add(variety)
    
    cis_by_variety = {}
    
    for variety_code in varieties:
        # Filter dataset for this variety
        variety_indices = [
            i for i, lp in enumerate(dataset["lp"])
            if extract_variety_from_language_pair(lp) == variety_code
        ]
        
        if not variety_indices:
            continue
        
        variety_dataset = dataset.select(variety_indices)
        
        # Calculate bootstrap CI for this variety
        cis = bootstrap_confidence_interval(
            variety_dataset,
            calculate_sqm_scores_single_variety,
            resampling_unit=resampling_unit,
            n_resamples=n_resamples,
            random_seed=random_seed,
        )
        
        cis_by_variety[variety_code] = cis
    
    return cis_by_variety


def format_detailed_template(
    template: str,
    scores_by_variety: dict[str, dict[str, float]],
    confidence_intervals_by_variety: dict[str, dict[str, tuple[float, float]]],
    counts_by_variety: dict[str, int],
    system_mappings: dict[str, str],
    threshold: int,
) -> str:
    """
    Format LaTeX template with scores and confidence intervals per variety.
    
    Args:
        template: LaTeX template string with placeholders like {system_key_variety_key}
        scores_by_variety: Dictionary mapping variety codes to dictionaries mapping system names to scores
        confidence_intervals_by_variety: Dictionary mapping variety codes to dictionaries mapping system names to (lower, upper) bounds
        counts_by_variety: Dictionary mapping variety codes to total counts (across all system pairs)
        system_mappings: Dictionary mapping data system names to template key prefixes
        threshold: Minimum count threshold for non-italicized numbers (>= threshold means no italics)
    
    Returns:
        Formatted template string
    """
    filled_template = template
    
    # Map variety codes to LaTeX column names
    variety_to_column = {code: LANGUAGE_PAIR_TO_VARIETY[code] for code in LANGUAGE_PAIR_TO_VARIETY}
    
    for system_name, key_prefix in system_mappings.items():
        for variety_code, column_name in variety_to_column.items():
            placeholder = f"{{{key_prefix}_{column_name}}}"
            
            # Get score for this system/variety combination
            score = scores_by_variety.get(variety_code, {}).get(system_name)
            
            # Get confidence interval for this system/variety combination
            confidence_interval = confidence_intervals_by_variety.get(variety_code, {}).get(system_name)
            
            # Get total count for this variety (across all system pairs)
            count = counts_by_variety.get(variety_code, 0)
            
            # Determine if number should be italicized (only if count < threshold)
            use_italics = count < threshold
            
            if score is None:
                formatted_score = r"\phantom{0.00 $\pm$ 0.00}-"
            elif confidence_interval is None:
                if use_italics:
                    formatted_score = f"{score:.2f} $\\pm$ tba"
                else:
                    formatted_score = f"{score:.2f} $\\pm$ tba"
            else:
                lower, upper = confidence_interval
                half_width = (upper - lower) / 2
                if use_italics:
                    formatted_score = f"{score:.2f} $\\pm$ {half_width:.2f}"
                else:
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
    
    # Calculate fluency scores per variety (count segments)
    print("Calculating fluency SQM scores per variety...")
    fluency_scores_by_variety, fluency_counts_by_variety = calculate_sqm_scores_by_variety(
        segment_fluency, count_documents=False
    )
    
    # Calculate accuracy scores per variety (count documents)
    print("Calculating accuracy SQM scores per variety...")
    accuracy_scores_by_variety, accuracy_counts_by_variety = calculate_sqm_scores_by_variety(
        document_accuracy, count_documents=True
    )
    
    # Calculate bootstrap confidence intervals per variety for fluency
    print("Calculating bootstrap CIs for fluency SQM per variety (segment-level)...")
    fluency_cis_by_variety = bootstrap_confidence_interval_by_variety(
        segment_fluency,
        resampling_unit="segment",
        n_resamples=1000,
        random_seed=42,
    )
    
    # Calculate bootstrap confidence intervals per variety for accuracy
    print("Calculating bootstrap CIs for accuracy SQM per variety (document-level)...")
    accuracy_cis_by_variety = bootstrap_confidence_interval_by_variety(
        document_accuracy,
        resampling_unit="document",
        n_resamples=1000,
        random_seed=42,
    )
    
    # Format fluency template (threshold: >= 511 segments)
    print("\nFormatting fluency table...")
    fluency_table = format_detailed_template(
        FLUENCY_TEMPLATE,
        fluency_scores_by_variety,
        fluency_cis_by_variety,
        fluency_counts_by_variety,
        system_mappings,
        threshold=511,
    )
    
    # Format accuracy template (threshold: >= 84 documents)
    print("Formatting accuracy table...")
    accuracy_table = format_detailed_template(
        ACCURACY_TEMPLATE,
        accuracy_scores_by_variety,
        accuracy_cis_by_variety,
        accuracy_counts_by_variety,
        system_mappings,
        threshold=84,
    )
    
    print("\n" + "="*80)
    print("FLUENCY TABLE:")
    print("="*80)
    print(fluency_table)
    
    print("\n" + "="*80)
    print("ACCURACY TABLE:")
    print("="*80)
    print(accuracy_table)
    
    # Write to output files if PAPER_DIR is set
    dotenv.load_dotenv()
    paper_dir = os.getenv("PAPER_DIR")
    if paper_dir is not None:
        fluency_output_path = Path(paper_dir) / "include" / "results_detailed_human_fluency.tex"
        accuracy_output_path = Path(paper_dir) / "include" / "results_detailed_human_accuracy.tex"
        
        write_output(fluency_table, fluency_output_path)
        write_output(accuracy_table, accuracy_output_path)
    else:
        print("\nWarning: PAPER_DIR environment variable not set. Tables not written to files.")


if __name__ == "__main__":
    main()

