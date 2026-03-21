import json
import os
from pathlib import Path

import dotenv

from romansh_mt_eval.benchmarking.constants import VARIETIES
from romansh_mt_eval.benchmarking.evaluation import RomanshWMT24Evaluation
from romansh_mt_eval.benchmarking.system_translations import load_llm_translations

# LaTeX table templates for each metric
TEMPLATE_DE_TO_RM_BLEU = r"""
{\footnotesize
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
\textbf{System} & \textbf{RG} & \textbf{Surs.} & \textbf{Suts.} & \textbf{Surm.} & \textbf{Puter} & \textbf{Vall.} \\
\midrule
\mbox{Gemini 2.5 Flash} & {gemini_25_flash_rm-rumgr} & {gemini_25_flash_rm-sursilv} & {gemini_25_flash_rm-sutsilv} & {gemini_25_flash_rm-surmiran} & {gemini_25_flash_rm-puter} & {gemini_25_flash_rm-vallader} \\
\mbox{Gemini 3 Flash (preview)} & {gemini_3_flash_rm-rumgr} & {gemini_3_flash_rm-sursilv} & {gemini_3_flash_rm-sutsilv} & {gemini_3_flash_rm-surmiran} & {gemini_3_flash_rm-puter} & {gemini_3_flash_rm-vallader} \\
\mbox{Gemini 3 Pro (preview)} & {gemini_3_pro_rm-rumgr} & {gemini_3_pro_rm-sursilv} & {gemini_3_pro_rm-sutsilv} & {gemini_3_pro_rm-surmiran} & {gemini_3_pro_rm-puter} & {gemini_3_pro_rm-vallader} \\
\midrule
\mbox{\textit{Fine-tuned NLLB}} & & & & & & \\[0.2em]
\mbox{No data augmentation} & {no_data_aug_rm-rumgr} & {no_data_aug_rm-sursilv} & {no_data_aug_rm-sutsilv} & {no_data_aug_rm-surmiran} & {no_data_aug_rm-puter} & {no_data_aug_rm-vallader} \\[0.2em]
\mbox{HR$\rightarrow$LR augmentation} & {forward_translation_rm-rumgr} & {forward_translation_rm-sursilv} & {forward_translation_rm-sutsilv} & {forward_translation_rm-surmiran} & {forward_translation_rm-puter} & {forward_translation_rm-vallader} \\[0.2em]
\mbox{LR$\rightarrow$HR augmentation} & {back_translation_rm-rumgr} & {back_translation_rm-sursilv} & {back_translation_rm-sutsilv} & {back_translation_rm-surmiran} & {back_translation_rm-puter} & {back_translation_rm-vallader} \\
\mbox{+ dictionary prompting} & {dict_prompting_rm-rumgr} & {dict_prompting_rm-sursilv} & {dict_prompting_rm-sutsilv} & {dict_prompting_rm-surmiran} & {dict_prompting_rm-puter} & {dict_prompting_rm-vallader} \\
\bottomrule
\end{tabular}
}
"""

TEMPLATE_RM_TO_DE_BLEU = r"""
{\footnotesize
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
\textbf{System} & \textbf{RG} & \textbf{Surs.} & \textbf{Suts.} & \textbf{Surm.} & \textbf{Puter} & \textbf{Vall.} \\
\midrule
\mbox{Gemini 2.5 Flash} & {gemini_25_flash_rm-rumgr} & {gemini_25_flash_rm-sursilv} & {gemini_25_flash_rm-sutsilv} & {gemini_25_flash_rm-surmiran} & {gemini_25_flash_rm-puter} & {gemini_25_flash_rm-vallader} \\
\mbox{Gemini 3 Flash (preview)} & {gemini_3_flash_rm-rumgr} & {gemini_3_flash_rm-sursilv} & {gemini_3_flash_rm-sutsilv} & {gemini_3_flash_rm-surmiran} & {gemini_3_flash_rm-puter} & {gemini_3_flash_rm-vallader} \\
\mbox{Gemini 3 Pro (preview)} & {gemini_3_pro_rm-rumgr} & {gemini_3_pro_rm-sursilv} & {gemini_3_pro_rm-sutsilv} & {gemini_3_pro_rm-surmiran} & {gemini_3_pro_rm-puter} & {gemini_3_pro_rm-vallader} \\
\midrule
\mbox{\textit{Fine-tuned NLLB}} & & & & & & \\[0.2em]
\mbox{No data augmentation} & {no_data_aug_rm-rumgr} & {no_data_aug_rm-sursilv} & {no_data_aug_rm-sutsilv} & {no_data_aug_rm-surmiran} & {no_data_aug_rm-puter} & {no_data_aug_rm-vallader} \\[0.2em]
\mbox{HR$\rightarrow$LR augmentation} & {forward_translation_rm-rumgr} & {forward_translation_rm-sursilv} & {forward_translation_rm-sutsilv} & {forward_translation_rm-surmiran} & {forward_translation_rm-puter} & {forward_translation_rm-vallader} \\[0.2em]
\mbox{LR$\rightarrow$HR augmentation} & {back_translation_rm-rumgr} & {back_translation_rm-sursilv} & {back_translation_rm-sutsilv} & {back_translation_rm-surmiran} & {back_translation_rm-puter} & {back_translation_rm-vallader} \\
\mbox{+ dictionary prompting} & {dict_prompting_rm-rumgr} & {dict_prompting_rm-sursilv} & {dict_prompting_rm-sutsilv} & {dict_prompting_rm-surmiran} & {dict_prompting_rm-puter} & {dict_prompting_rm-vallader} \\
\bottomrule
\end{tabular}
}
"""

TEMPLATE_RM_TO_DE_COMET = r"""
{\footnotesize
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
\textbf{System} & \textbf{RG} & \textbf{Surs.} & \textbf{Suts.} & \textbf{Surm.} & \textbf{Puter} & \textbf{Vall.} \\
\midrule
\mbox{Gemini 2.5 Flash} & {gemini_25_flash_rm-rumgr} & {gemini_25_flash_rm-sursilv} & {gemini_25_flash_rm-sutsilv} & {gemini_25_flash_rm-surmiran} & {gemini_25_flash_rm-puter} & {gemini_25_flash_rm-vallader} \\
\mbox{Gemini 3 Flash (preview)} & {gemini_3_flash_rm-rumgr} & {gemini_3_flash_rm-sursilv} & {gemini_3_flash_rm-sutsilv} & {gemini_3_flash_rm-surmiran} & {gemini_3_flash_rm-puter} & {gemini_3_flash_rm-vallader} \\
\mbox{Gemini 3 Pro (preview)} & {gemini_3_pro_rm-rumgr} & {gemini_3_pro_rm-sursilv} & {gemini_3_pro_rm-sutsilv} & {gemini_3_pro_rm-surmiran} & {gemini_3_pro_rm-puter} & {gemini_3_pro_rm-vallader} \\
\midrule
\mbox{\textit{Fine-tuned NLLB}} & & & & & & \\[0.2em]
\mbox{No data augmentation} & {no_data_aug_rm-rumgr} & {no_data_aug_rm-sursilv} & {no_data_aug_rm-sutsilv} & {no_data_aug_rm-surmiran} & {no_data_aug_rm-puter} & {no_data_aug_rm-vallader} \\[0.2em]
\mbox{HR$\rightarrow$LR augmentation} & {forward_translation_rm-rumgr} & {forward_translation_rm-sursilv} & {forward_translation_rm-sutsilv} & {forward_translation_rm-surmiran} & {forward_translation_rm-puter} & {forward_translation_rm-vallader} \\[0.2em]
\mbox{LR$\rightarrow$HR augmentation} & {back_translation_rm-rumgr} & {back_translation_rm-sursilv} & {back_translation_rm-sutsilv} & {back_translation_rm-surmiran} & {back_translation_rm-puter} & {back_translation_rm-vallader} \\
\mbox{+ dictionary prompting} & {dict_prompting_rm-rumgr} & {dict_prompting_rm-sursilv} & {dict_prompting_rm-sutsilv} & {dict_prompting_rm-surmiran} & {dict_prompting_rm-puter} & {dict_prompting_rm-vallader} \\
\bottomrule
\end{tabular}
}
"""

# System display name mapping
SYSTEM_DISPLAY_NAMES = {
    "gemini_25_flash": r"\mbox{Gemini 2.5 Flash}",
    "gemini_3_flash": r"\mbox{Gemini 3 Flash (preview)}",
    "gemini_3_pro": r"\mbox{Gemini 3 Pro (preview)}",
    "no_data_aug": r"\mbox{No data augmentation}",
    "forward_translation": r"\mbox{HR$\rightarrow$LR augmentation}",
    "back_translation": r"\mbox{LR$\rightarrow$HR augmentation}",
    "dict_prompting": r"\mbox{+ dictionary prompting}",
}


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


def evaluate_system_detailed(
    system_path: str,
    document_ids_filter: set[str] | None,
    evaluation: RomanshWMT24Evaluation
) -> dict[str, dict[str, float | None]]:
    """
    Evaluate a system and return per-variety scores.
    
    Returns:
        Dictionary with keys: 'de_to_rm_bleu', 'rm_to_de_bleu', 'rm_to_de_comet'
        Each value is a dict mapping variety codes to scores (or None if not available)
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
    
    # Extract per-variety scores
    de_to_rm_bleu_scores = {}
    rm_to_de_bleu_scores = {}
    rm_to_de_comet_scores = {}
    
    for variety in VARIETIES.keys():
        result_bleu = results_bleu.get(variety)
        result_comet = results_comet.get(variety)
        
        if result_bleu is not None:
            de_to_rm_score = result_bleu.scores_de_to_rm.micro_avg
            rm_to_de_score = result_bleu.scores_rm_to_de.micro_avg
            de_to_rm_bleu_scores[variety] = de_to_rm_score
            rm_to_de_bleu_scores[variety] = rm_to_de_score
        else:
            de_to_rm_bleu_scores[variety] = None
            rm_to_de_bleu_scores[variety] = None
        
        if result_comet is not None:
            rm_to_de_comet_score = result_comet.scores_rm_to_de.macro_avg
            rm_to_de_comet_scores[variety] = rm_to_de_comet_score
        else:
            rm_to_de_comet_scores[variety] = None
    
    return {
        'de_to_rm_bleu': de_to_rm_bleu_scores,
        'rm_to_de_bleu': rm_to_de_bleu_scores,
        'rm_to_de_comet': rm_to_de_comet_scores,
    }


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
        print(f"Evaluating {system_key}...")
        scores[system_key] = evaluate_system_detailed(
            system_path,
            document_ids_filter,
            evaluation
        )

    # Generate three tables
    templates = {
        'de_to_rm_bleu': TEMPLATE_DE_TO_RM_BLEU,
        'rm_to_de_bleu': TEMPLATE_RM_TO_DE_BLEU,
        'rm_to_de_comet': TEMPLATE_RM_TO_DE_COMET,
    }
    
    output_files = {
        'de_to_rm_bleu': 'results_detailed_automatic_de_to_rm_bleu.tex',
        'rm_to_de_bleu': 'results_detailed_automatic_rm_to_de_bleu.tex',
        'rm_to_de_comet': 'results_detailed_automatic_rm_to_de_comet.tex',
    }
    
    for metric_key, template in templates.items():
        filled_template = template
        for system_key in system_mappings.keys():
            system_scores = scores[system_key]
            metric_scores = system_scores[metric_key]
            
            for variety in VARIETIES.keys():
                placeholder = f"{{{system_key}_{variety}}}"
                score = metric_scores.get(variety)
                if isinstance(score, float):
                    formatted_score = f"{score:.1f}"
                elif score is None:
                    formatted_score = "--"
                else:
                    formatted_score = str(score)
                filled_template = filled_template.replace(placeholder, formatted_score)
        
        print(f"\n{metric_key} table:")
        print(filled_template)
        
        # Write to output file
        dotenv.load_dotenv()
        if os.getenv("PAPER_DIR") is not None:
            output_path = Path(os.getenv("PAPER_DIR")) / "include" / output_files[metric_key]
            
            if output_path.parent.exists():
                with output_path.open("w", encoding="utf-8") as f:
                    f.write(filled_template)
                print(f"\nResults written to {output_path}")
            else:
                print(f"\nWarning: Output directory does not exist: {output_path.parent}")


if __name__ == "__main__":
    main()

