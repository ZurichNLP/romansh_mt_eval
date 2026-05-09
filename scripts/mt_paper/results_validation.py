import json
import os
from pathlib import Path

import dotenv
import numpy as np

from romansh_mt_eval.benchmarking.constants import VARIETIES
from romansh_mt_eval.benchmarking.evaluation import RomanshWMT24Evaluation, SystemResult
from romansh_mt_eval.benchmarking.system_translations import load_llm_translations

TEMPLATE = r"""\footnotesize
\begin{tabularx}{\columnwidth}{@{}Xrr@{}}
\toprule
& \textbf{DE$\rightarrow$RM} & \textbf{RM$\rightarrow$DE} \\
\cmidrule(lr){2-2} \cmidrule(lr){3-3}
\textbf{Approach} & \textbf{BLEU} & \textbf{BLEU / COMET} \\
\midrule
\mbox{\textit{Baseline LLMs}} & & \\[0.2em]
\mbox{Llama 3.3 (70B)} & {llama_70b_de_to_rm} & {llama_70b_rm_to_de} \\
\mbox{GPT-4o} & {gpt_4o_de_to_rm} & {gpt_4o_rm_to_de} \\
\midrule
\mbox{\textit{Gemini 2.5 Flash}} & & \\[0.2em]
\mbox{Baseline prompt (\ref{app:forward-backtranslation})} & {baseline_de_to_rm} & {baseline_rm_to_de} \\
\mbox{\quad – with reasoning} & {reasoning_de_to_rm} & {reasoning_rm_to_de} \\
\mbox{\quad -- without few-shot examples} & {zeroshot_de_to_rm} & {zeroshot_rm_to_de} \\[0.2em]
\mbox{Dictionary prompting (\ref{app:backtranslation-dict})} & - & {dictionary_rm_to_de} \\
\mbox{\quad – with reasoning} & - & {dictionary_reasoning_rm_to_de} \\
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


def evaluate_validation_system(
    system_path: str,
    document_ids_filter: set[str] | None,
    evaluation: RomanshWMT24Evaluation,
    *,
    load_de_to_rm: bool,
    load_rm_to_de: bool = True,
) -> dict[str, float | None]:
    """
    Evaluate a system and return scores averaged across varieties.

    Either direction can be skipped if its translation files are absent in this
    folder (e.g. the Gemini RM→DE validation runs only have RM→DE files, while the
    new forward-translation runs only have DE→RM files). Skipped directions return
    None for their score keys.

    Returns:
        Keys: de_to_rm_bleu, rm_to_de_bleu, rm_to_de_comet
    """
    llm_translations = load_llm_translations(
        system_path,
        de_to_rm=load_de_to_rm,
        rm_to_de=load_rm_to_de,
        document_ids_filter=document_ids_filter,
    )

    results_bleu: dict[str, SystemResult] = {}
    for variety, system_translations in llm_translations.items():
        results_bleu[variety] = evaluation.evaluate(
            system_translations,
            "bleu",
            document_ids_filter=document_ids_filter,
        )

    results_comet: dict[str, SystemResult] = {}
    if load_rm_to_de:
        for variety, system_translations in llm_translations.items():
            results_comet[variety] = evaluation.evaluate(
                system_translations,
                "xcomet-xl",
                document_ids_filter=document_ids_filter,
            )

    de_to_rm_bleu_scores: list[float] = []
    rm_to_de_bleu_scores: list[float] = []
    rm_to_de_comet_scores: list[float] = []

    for variety in VARIETIES.keys():
        result_bleu = results_bleu.get(variety)
        result_comet = results_comet.get(variety)

        if result_bleu is not None:
            if load_de_to_rm:
                de_to_rm_score = result_bleu.scores_de_to_rm.micro_avg
                if de_to_rm_score is not None:
                    de_to_rm_bleu_scores.append(de_to_rm_score)
            if load_rm_to_de:
                rm_to_de_score = result_bleu.scores_rm_to_de.micro_avg
                if rm_to_de_score is not None:
                    rm_to_de_bleu_scores.append(rm_to_de_score)

        if result_comet is not None:
            rm_to_de_comet_score = result_comet.scores_rm_to_de.macro_avg
            if rm_to_de_comet_score is not None:
                rm_to_de_comet_scores.append(rm_to_de_comet_score)

    return {
        "de_to_rm_bleu": np.mean(de_to_rm_bleu_scores).item() if de_to_rm_bleu_scores else None,
        "rm_to_de_bleu": np.mean(rm_to_de_bleu_scores).item() if rm_to_de_bleu_scores else None,
        "rm_to_de_comet": np.mean(rm_to_de_comet_scores).item() if rm_to_de_comet_scores else None,
    }


def format_de_to_rm_bleu(system_scores: dict[str, float | None]) -> str:
    bleu_score = system_scores["de_to_rm_bleu"]
    if isinstance(bleu_score, float):
        return f"{bleu_score:.1f}"
    return "tba"


def format_rm_to_de_bleu_comet(system_scores: dict[str, float | None]) -> str:
    bleu_score = system_scores["rm_to_de_bleu"]
    comet_score = system_scores["rm_to_de_comet"]
    if isinstance(bleu_score, float) and isinstance(comet_score, float):
        return f"{bleu_score:.1f} / {comet_score:.1f}"
    return "tba / tba"


def main():
    baseline_llm_systems = {
        "llama_70b": "system_translations/mt_paper/first_half/Llama-3.3-70b",
        "gpt_4o": "system_translations/mt_paper/first_half/GPT-4o",
    }
    gemini_system_mappings = {
        "baseline": "system_translations/mt_paper/first_half/config_wmt24pp_validation_gemini_2_5_flash_baseline_collect_monolingual_build_wmt24pp_wmt24pp_dataset_first_half_jsonl_5f95b4547d06.output.stage1",
        "reasoning": "system_translations/mt_paper/first_half/config_wmt24pp_validation_gemini_2_5_flash_reasoning_collect_monolingual_build_wmt24pp_wmt24pp_dataset_first_half_jsonl_86cb7f9dd57c.output.stage1",
        "zeroshot": "system_translations/mt_paper/first_half/config_wmt24pp_validation_gemini_2_5_flash_zeroshot_collect_monolingual_build_wmt24pp_wmt24pp_dataset_first_half_jsonl_33be843bde96.output.stage1",
        "dictionary": "system_translations/mt_paper/first_half/config_wmt24pp_validation_gemini_2_5_flash_dictionary_collect_monolingual_build_wmt24pp_wmt24pp_dataset_first_half_jsonl_ea752607b08d.output.stage1",
        "dictionary_reasoning": "system_translations/mt_paper/first_half/config_wmt24pp_validation_gemini_2_5_flash_dictionary_reasoning_collect_monolingual_build_wmt24pp_wmt24pp_dataset_first_half_jsonl_aec0ff3a0dca.output.stage1",
    }
    # Forward (DE→RM) translation folders, produced by the forward_translation pipeline
    # and exported with a "forward." prefix to keep them distinct from the RM→DE folders
    # above that share the same parent-naming pattern.
    gemini_forward_system_mappings = {
        "baseline": "system_translations/mt_paper/first_half/forward.config_wmt24pp_validation_gemini_2_5_flash_baseline_build_wmt24pp_wmt24pp_dataset_first_half_jsonl_861f22cd4c4f.output.stage1",
        "reasoning": "system_translations/mt_paper/first_half/forward.config_wmt24pp_validation_gemini_2_5_flash_reasoning_build_wmt24pp_wmt24pp_dataset_first_half_jsonl_21e6aaf18f2a.output.stage1",
        "zeroshot": "system_translations/mt_paper/first_half/forward.config_wmt24pp_validation_gemini_2_5_flash_zeroshot_build_wmt24pp_wmt24pp_dataset_first_half_jsonl_85a713973acc.output.stage1",
    }

    # Load split file from benchmarking directory
    workspace_root = Path(__file__).parent.parent.parent
    split_path = workspace_root / "benchmarking" / "wmt24pp_split.json"
    document_ids_filter = load_split_document_ids(split_path, "first")

    evaluation = RomanshWMT24Evaluation()
    scores: dict[str, dict[str, float | None]] = {}

    for system_key, system_path in baseline_llm_systems.items():
        print(f"Evaluating {system_key}...")
        scores[system_key] = evaluate_validation_system(
            system_path,
            document_ids_filter,
            evaluation,
            load_de_to_rm=True,
        )

    for system_key, system_path in gemini_system_mappings.items():
        print(f"Evaluating {system_key}...")
        scores[system_key] = evaluate_validation_system(
            system_path,
            document_ids_filter,
            evaluation,
            load_de_to_rm=False,
        )

    for system_key, system_path in gemini_forward_system_mappings.items():
        print(f"Evaluating {system_key} (forward, DE→RM)...")
        forward_scores = evaluate_validation_system(
            system_path,
            document_ids_filter,
            evaluation,
            load_de_to_rm=True,
            load_rm_to_de=False,
        )
        scores[system_key]["de_to_rm_bleu"] = forward_scores["de_to_rm_bleu"]

    # Format scores for LaTeX
    filled_template = TEMPLATE
    for system_key in baseline_llm_systems.keys():
        filled_template = filled_template.replace(
            f"{{{system_key}_de_to_rm}}",
            format_de_to_rm_bleu(scores[system_key]),
        )
        filled_template = filled_template.replace(
            f"{{{system_key}_rm_to_de}}",
            format_rm_to_de_bleu_comet(scores[system_key]),
        )
    for system_key in gemini_system_mappings.keys():
        placeholder = f"{{{system_key}_rm_to_de}}"
        filled_template = filled_template.replace(
            placeholder, format_rm_to_de_bleu_comet(scores[system_key])
        )
    for system_key in gemini_forward_system_mappings.keys():
        placeholder = f"{{{system_key}_de_to_rm}}"
        filled_template = filled_template.replace(
            placeholder, format_de_to_rm_bleu(scores[system_key])
        )
    
    print(filled_template)

    # Write to output file
    dotenv.load_dotenv()
    paper_directory = os.getenv("PAPER_DIR")
    if paper_directory is not None:
        output_path = Path(paper_directory) / "include/results_validation_data_augmentation.tex"

        if output_path.parent.exists():
            with output_path.open("w", encoding="utf-8") as f:
                f.write(filled_template)
            print(f"\nResults written to {output_path}")
        else:
            print(f"\nWarning: Output directory does not exist: {output_path.parent}")


if __name__ == "__main__":
    main()

