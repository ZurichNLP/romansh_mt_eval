import json
import os
from pathlib import Path

import dotenv
import numpy as np

from romansh_mt_eval.benchmarking.constants import VARIETIES
from romansh_mt_eval.benchmarking.evaluation import RomanshWMT24Evaluation
from romansh_mt_eval.benchmarking.system_translations import load_llm_translations

TEMPLATE = r"""\begin{tabularx}{\columnwidth}{@{}Xrr@{}}
\toprule
& \multicolumn{2}{c}{\textbf{RM$\rightarrow$DE}} \\
\cmidrule(lr){2-3}
\textbf{Approach} & \textbf{BLEU} & \textbf{COMET} \\
\midrule
\mbox{Baseline prompt (\ref{app:forward-backtranslation})} & {baseline_rm_to_de_bleu} & {baseline_rm_to_de_comet} \\
\mbox{\quad + reasoning} & {reasoning_rm_to_de_bleu} & {reasoning_rm_to_de_comet} \\
\mbox{\quad – without few-shot examples} & {zeroshot_rm_to_de_bleu} & {zeroshot_rm_to_de_comet} \\[0.2em]
\midrule
\mbox{Dictionary prompting (\ref{app:backtranslation-dict})} & {dictionary_rm_to_de_bleu} & {dictionary_rm_to_de_comet} \\
\mbox{\quad + reasoning} & {dictionary_reasoning_rm_to_de_bleu} & {dictionary_reasoning_rm_to_de_comet} \\
\bottomrule
\end{tabularx}
"""


def load_split_document_ids(split_path: Path, half: str) -> set[str] | None:
    """
    Load document IDs from the split JSON file for the specified half.
    
    Args:
        split_path: Path to wmt24pp_split.json
        half: One of 'first', 'second', or 'both'
        
    Returns:
        Set of document IDs to filter by, or None if half is 'both'
    """
    if half == "both":
        return None
    
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    
    with open(split_path, 'r', encoding='utf-8') as f:
        split_data = json.load(f)
    
    if half == "first":
        return set(split_data["first_half"])
    elif half == "second":
        return set(split_data["second_half"])
    else:
        raise ValueError(f"Invalid half value: {half}. Must be 'first', 'second', or 'both'.")


def evaluate_system(
    system_path: str,
    document_ids_filter: set[str] | None,
    evaluation: RomanshWMT24Evaluation
) -> dict[str, float]:
    """
    Evaluate a system and return aggregated scores across all varieties.
    Only evaluates RM→DE metrics.
    
    Returns:
        Dictionary with keys: 'rm_to_de_bleu', 'rm_to_de_comet'
    """
    llm_translations = load_llm_translations(
        system_path,
        de_to_rm=False,
        rm_to_de=True,
        document_ids_filter=document_ids_filter
    )
    
    # Evaluate for BLEU
    results_bleu = {}
    for variety, system_translations in llm_translations.items():
        results_bleu[variety] = evaluation.evaluate(
            system_translations,
            "bleu",
            document_ids_filter=document_ids_filter
        )
    
    # Evaluate for COMET (RM→DE only)
    results_comet = {}
    for variety, system_translations in llm_translations.items():
        results_comet[variety] = evaluation.evaluate(
            system_translations,
            "xcomet-xl",
            document_ids_filter=document_ids_filter
        )
    
    # Aggregate scores across varieties
    rm_to_de_bleu_scores = []
    rm_to_de_comet_scores = []
    
    for variety in VARIETIES.keys():
        result_bleu = results_bleu.get(variety)
        result_comet = results_comet.get(variety)
        
        if result_bleu is not None:
            rm_to_de_score = result_bleu.scores_rm_to_de.micro_avg
            if rm_to_de_score is not None:
                rm_to_de_bleu_scores.append(rm_to_de_score)
        
        if result_comet is not None:
            rm_to_de_comet_score = result_comet.scores_rm_to_de.macro_avg
            if rm_to_de_comet_score is not None:
                rm_to_de_comet_scores.append(rm_to_de_comet_score)
    
    # Average across varieties
    aggregated = {
        'rm_to_de_bleu': np.mean(rm_to_de_bleu_scores).item() if rm_to_de_bleu_scores else None,
        'rm_to_de_comet': np.mean(rm_to_de_comet_scores).item() if rm_to_de_comet_scores else None,
    }
    
    return aggregated


def main():
    # System mappings: display_name -> system_path
    system_mappings = {
        "baseline": "system_translations/mt_paper/first_half/config_wmt24pp_validation_gemini_2_5_flash_baseline_collect_monolingual_build_wmt24pp_wmt24pp_dataset_first_half_jsonl_5f95b4547d06.output.stage1",
        "reasoning": "system_translations/mt_paper/first_half/config_wmt24pp_validation_gemini_2_5_flash_reasoning_collect_monolingual_build_wmt24pp_wmt24pp_dataset_first_half_jsonl_86cb7f9dd57c.output.stage1",
        "zeroshot": "system_translations/mt_paper/first_half/config_wmt24pp_validation_gemini_2_5_flash_zeroshot_collect_monolingual_build_wmt24pp_wmt24pp_dataset_first_half_jsonl_33be843bde96.output.stage1",
        "dictionary": "system_translations/mt_paper/first_half/config_wmt24pp_validation_gemini_2_5_flash_dictionary_collect_monolingual_build_wmt24pp_wmt24pp_dataset_first_half_jsonl_ea752607b08d.output.stage1",
        "dictionary_reasoning": "system_translations/mt_paper/first_half/config_wmt24pp_validation_gemini_2_5_flash_dictionary_reasoning_collect_monolingual_build_wmt24pp_wmt24pp_dataset_first_half_jsonl_aec0ff3a0dca.output.stage1",
    }
    
    # Load split file from benchmarking directory
    workspace_root = Path(__file__).parent.parent.parent
    split_path = workspace_root / "benchmarking" / "wmt24pp_split.json"
    document_ids_filter = load_split_document_ids(split_path, "first")
    
    # Evaluate systems
    evaluation = RomanshWMT24Evaluation()
    scores = {}
    
    for system_key, system_path in system_mappings.items():
        print(f"Evaluating {system_key}...")
        scores[system_key] = evaluate_system(
            system_path,
            document_ids_filter,
            evaluation
        )
    
    # Format scores for LaTeX
    filled_template = TEMPLATE
    for system_key in system_mappings.keys():
        system_scores = scores[system_key]
        for metric_key in ['rm_to_de_bleu', 'rm_to_de_comet']:
            placeholder = f"{{{system_key}_{metric_key}}}"
            score = system_scores[metric_key]
            if isinstance(score, float):
                formatted_score = f"{score:.1f}"
            else:
                formatted_score = str(score)
            filled_template = filled_template.replace(placeholder, formatted_score)
    
    print(filled_template)
    
    # Write to output file
    dotenv.load_dotenv()
    if os.getenv("PAPER_DIR") is not None:
        output_path = Path(os.getenv("PAPER_DIR")) / "include/results_validation_backtranslation.tex"
    
        if output_path.parent.exists():
            with output_path.open("w", encoding="utf-8") as f:
                f.write(filled_template)
            print(f"\nResults written to {output_path}")
        else:
            print(f"\nWarning: Output directory does not exist: {output_path.parent}")


if __name__ == "__main__":
    main()

