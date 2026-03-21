import json
import os
from pathlib import Path

import dotenv
import numpy as np

from romansh_mt_eval.benchmarking.constants import VARIETIES
from romansh_mt_eval.benchmarking.evaluation import RomanshWMT24Evaluation
from romansh_mt_eval.benchmarking.system_translations import load_llm_translations

TEMPLATE = r"""
{\footnotesize
\begin{tabularx}{\columnwidth}{@{}Xcrr@{}}
\toprule
& \textbf{DE$\rightarrow$RM} & \multicolumn{2}{c}{\textbf{RM$\rightarrow$DE}} \\
\cmidrule(lr){2-2} \cmidrule(lr){3-4}
\textbf{System} & \textbf{BLEU} & \textbf{BLEU} & \textbf{COMET} \\
\midrule
\mbox{Gemini 2.5 Flash\textdagger} & {gemini_25_flash_de_to_rm_bleu} & {gemini_25_flash_rm_to_de_bleu} & {gemini_25_flash_rm_to_de_comet} \\
\mbox{Gemini 3 Flash (preview)} & {gemini_3_flash_de_to_rm_bleu} & {gemini_3_flash_rm_to_de_bleu} & {gemini_3_flash_rm_to_de_comet} \\
\mbox{Gemini 3 Pro (preview)} & {gemini_3_pro_de_to_rm_bleu} & {gemini_3_pro_rm_to_de_bleu} & {gemini_3_pro_rm_to_de_comet} \\
\midrule
\mbox{\textit{Fine-tuned NLLB}} & & & \\[0.2em]
\mbox{No data augmentation} & {no_data_aug_de_to_rm_bleu} & {no_data_aug_rm_to_de_bleu} & {no_data_aug_rm_to_de_comet} \\
\mbox{HR$\rightarrow$LR augmentation} & {forward_translation_de_to_rm_bleu} & {forward_translation_rm_to_de_bleu} & {forward_translation_rm_to_de_comet} \\
\mbox{LR$\rightarrow$HR augmentation} & {back_translation_de_to_rm_bleu} & {back_translation_rm_to_de_bleu} & {back_translation_rm_to_de_comet} \\
\mbox{+ dictionary prompting} & {dict_prompting_de_to_rm_bleu} & {dict_prompting_rm_to_de_bleu} & {dict_prompting_rm_to_de_comet} \\
\bottomrule
\end{tabularx}
}
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
    
    Returns:
        Dictionary with keys: 'de_to_rm_bleu', 'rm_to_de_bleu', 'rm_to_de_comet'
    """
    llm_translations = load_llm_translations(
        system_path,
        de_to_rm=True,
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
    de_to_rm_bleu_scores = []
    rm_to_de_bleu_scores = []
    rm_to_de_comet_scores = []
    
    for variety in VARIETIES.keys():
        result_bleu = results_bleu.get(variety)
        result_comet = results_comet.get(variety)
        
        if result_bleu is not None:
            de_to_rm_score = result_bleu.scores_de_to_rm.micro_avg
            rm_to_de_score = result_bleu.scores_rm_to_de.micro_avg
            if de_to_rm_score is not None:
                de_to_rm_bleu_scores.append(de_to_rm_score)
            if rm_to_de_score is not None:
                rm_to_de_bleu_scores.append(rm_to_de_score)
        
        if result_comet is not None:
            rm_to_de_comet_score = result_comet.scores_rm_to_de.macro_avg
            if rm_to_de_comet_score is not None:
                rm_to_de_comet_scores.append(rm_to_de_comet_score)
    
    # Average across varieties
    aggregated = {
        'de_to_rm_bleu': np.mean(de_to_rm_bleu_scores).item() if de_to_rm_bleu_scores else None,
        'rm_to_de_bleu': np.mean(rm_to_de_bleu_scores).item() if rm_to_de_bleu_scores else None,
        'rm_to_de_comet': np.mean(rm_to_de_comet_scores).item() if rm_to_de_comet_scores else None,
    }
    
    return aggregated


def main():
    # System mappings: display_name -> system_path (None means output "tba")
    system_mappings = {
        "gemini_25_flash": "system_translations/mt_paper/second_half/Gemini-2.5-Flash",
        "gemini_3_flash": "system_translations/mt_paper/second_half/Gemini-3-Flash",
        "gemini_3_pro": "system_translations/mt_paper/second_half/Gemini-3-Pro",
        "no_data_aug": "system_translations/mt_paper/second_half/ctranslate2_fairseq_nllb-200-distilled-1.3B.norm.temp1.5.10e.noback.withdict_ct2",
        "forward_translation": "system_translations/mt_paper/second_half/ctranslate2_fairseq_nllb-200-distilled-1.3B.norm.temp1.5.10e.forward_override.withdict_ct2",
        "back_translation": "system_translations/mt_paper/second_half/ctranslate2_fairseq_nllb-200-distilled-1.3B.norm.temp1.5.10e.withdict_ct2",
        "dict_prompting": "system_translations/mt_paper/second_half/ctranslate2_fairseq_nllb-200-distilled-1.3B.norm.temp1.5.10e.withdict.dict_prompting_ct2",
    }
    
    # Load split file from benchmarking directory
    workspace_root = Path(__file__).parent.parent.parent
    split_path = workspace_root / "benchmarking" / "wmt24pp_split.json"
    document_ids_filter = load_split_document_ids(split_path, "second")
    
    # Evaluate systems
    evaluation = RomanshWMT24Evaluation()
    scores = {}
    
    for system_key, system_path in system_mappings.items():
        if system_path is None:
            # Systems without paths (forward translation) - output "tba"
            scores[system_key] = {
                'de_to_rm_bleu': 'tba',
                'rm_to_de_bleu': 'tba',
                'rm_to_de_comet': 'tba',
            }
        else:
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
        for metric_key in ['de_to_rm_bleu', 'rm_to_de_bleu', 'rm_to_de_comet']:
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
        output_path = Path(os.getenv("PAPER_DIR")) / "include/results_automatic.tex"
    
        if output_path.parent.exists():
            with output_path.open("w", encoding="utf-8") as f:
                f.write(filled_template)
            print(f"\nResults written to {output_path}")
        else:
            print(f"\nWarning: Output directory does not exist: {output_path.parent}")


if __name__ == "__main__":
    main()

