import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import dotenv

# Add workspace root to Python path
script_dir = Path(__file__).resolve().parent
workspace_root = script_dir.parent.parent
parent_dir = workspace_root.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from romansh_mt_eval.benchmarking.constants import VARIETIES
from romansh_mt_eval.benchmarking.evaluation import RomanshWMT24Evaluation
from romansh_mt_eval.benchmarking.system_translations import load_llm_translations

METRIC = "bleu"

# System mappings: system_key -> (system_path, display_name)
# None path means show "tba" for all cells
SYSTEM_MAPPINGS = {
    "gemini_25_flash": ("system_translations/mt_paper/second_half/Gemini-2.5-Flash", r"\mbox{Gemini 2.5 Flash}"),
    "gemini_3_flash": ("system_translations/mt_paper/second_half/Gemini-3-Flash", r"\mbox{Gemini 3 Flash}"),
    "gemini_3_pro": ("system_translations/mt_paper/second_half/Gemini-3-Pro", r"\mbox{Gemini 3 Pro}"),
    "no_data_aug": ("system_translations/mt_paper/second_half/ctranslate2_fairseq_nllb-200-distilled-1.3B.norm.temp1.5.10e.noback.withdict_ct2", r"\mbox{No data augmentation}"),
    "forward_translation": ("system_translations/mt_paper/second_half/ctranslate2_fairseq_nllb-200-distilled-1.3B.norm.temp1.5.10e.forward_override.withdict_ct2", r"\mbox{High-to-Low synthetization}"),
    "back_translation": ("system_translations/mt_paper/second_half/ctranslate2_fairseq_nllb-200-distilled-1.3B.norm.temp1.5.10e.withdict_ct2", r"\mbox{Low-to-High synthetization}"),
    "dict_prompting": ("system_translations/mt_paper/second_half/ctranslate2_fairseq_nllb-200-distilled-1.3B.norm.temp1.5.10e.withdict.dict_prompting_ct2", r"\mbox{Low-to-High synthetization with dictionary prompting}"),
}

# System order for grid layout (2 columns)
SYSTEM_ORDER = [
    ["gemini_25_flash", "gemini_3_flash"],
    ["gemini_3_pro", "no_data_aug"],
    ["forward_translation", "back_translation"],
    ["dict_prompting", None],  # None means empty spot
]

benchmark = RomanshWMT24Evaluation()


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


def modify_benchmark_ref(ref_variety):
    """
    Returns a copy of the benchmark where ref_variety is used a reference for every variety
    """
    modified_benchmark = deepcopy(benchmark)
    for variety in VARIETIES:
        modified_benchmark.dataset[variety] = deepcopy(benchmark.dataset[ref_variety])
    return modified_benchmark


def compute_scores(system_path: str | None, document_ids_filter: set[str] | None):
    """
    Compute scores for a given system path.
    Returns a dictionary of scores: {tgt: {ref: score}}
    If system_path is None, returns all "tba" values.
    """
    if system_path is None:
        # Return "tba" for all cells
        return {tgt: {ref: "tba" for ref in VARIETIES} for tgt in VARIETIES}
    
    scores = {tgt: {ref: 0. for ref in VARIETIES} for tgt in VARIETIES}
    
    # Prepend mt_paper second_half if evaluating second half and not already prefixed
    if document_ids_filter is not None:
        prefix = "system_translations/mt_paper/second_half/"
        if not system_path.startswith(prefix) and not system_path.startswith("system_translations/"):
            system_path = f"{prefix}{system_path}"
    
    for ref in VARIETIES:
        modified_benchmark = modify_benchmark_ref(ref)
        system_translations = load_llm_translations(system_path, document_ids_filter=document_ids_filter)
        for tgt in VARIETIES:
            result = modified_benchmark.evaluate(system_translations[tgt], METRIC, document_ids_filter=document_ids_filter)
            score = result.scores_de_to_rm.micro_avg
            scores[tgt][ref] = score
    
    return scores


# LaTeX template with y-axis labels (for first column of each row)
TABLE_WITH_LABELS = r"""
\begin{tabularx}{8cm}{@{}p{0.9cm}|*{6}{>{\raggedleft\arraybackslash}p{0.65cm}}}
\multirow[b]{2}{*}{$\downarrow$~\textbf{tgt}}
& \multicolumn{6}{l}{$\textbf{ref}$~$\rightarrow$} \\
& RG & Surs.\ & Suts.\ & Surm.\ & Puter & Vall. \\
RG     & {rm-rumgr_rm-rumgr}   & {rm-rumgr_rm-sursilv}   & {rm-rumgr_rm-sutsilv}   & {rm-rumgr_rm-surmiran}   & {rm-rumgr_rm-puter}   & {rm-rumgr_rm-vallader}   \\
Surs.  & {rm-sursilv_rm-rumgr} & {rm-sursilv_rm-sursilv} & {rm-sursilv_rm-sutsilv} & {rm-sursilv_rm-surmiran} & {rm-sursilv_rm-puter} & {rm-sursilv_rm-vallader} \\
Suts.  & {rm-sutsilv_rm-rumgr} & {rm-sutsilv_rm-sursilv} & {rm-sutsilv_rm-sutsilv} & {rm-sutsilv_rm-surmiran} & {rm-sutsilv_rm-puter} & {rm-sutsilv_rm-vallader} \\
Surm.  & {rm-surmiran_rm-rumgr} & {rm-surmiran_rm-sursilv} & {rm-surmiran_rm-sutsilv} & {rm-surmiran_rm-surmiran} & {rm-surmiran_rm-puter} & {rm-surmiran_rm-vallader} \\
Puter  & {rm-puter_rm-rumgr}   & {rm-puter_rm-sursilv}   & {rm-puter_rm-sutsilv}   & {rm-puter_rm-surmiran}   & {rm-puter_rm-puter}   & {rm-puter_rm-vallader}   \\
Vall.  & {rm-vallader_rm-rumgr} & {rm-vallader_rm-sursilv} & {rm-vallader_rm-sutsilv} & {rm-vallader_rm-surmiran} & {rm-vallader_rm-puter} & {rm-vallader_rm-vallader} \\
\end{tabularx}
"""

# LaTeX template without y-axis labels (for other columns)
TABLE_WITHOUT_LABELS = r"""
\begin{tabularx}{6cm}{@{}*{6}{>{\raggedleft\arraybackslash}p{0.65cm}}}
\multicolumn{6}{l}{$\textbf{ref}$~$\rightarrow$} \\
RG & Surs.\ & Suts.\ & Surm.\ & Puter & Vall. \\
{rm-rumgr_rm-rumgr}   & {rm-rumgr_rm-sursilv}   & {rm-rumgr_rm-sutsilv}   & {rm-rumgr_rm-surmiran}   & {rm-rumgr_rm-puter}   & {rm-rumgr_rm-vallader}   \\
{rm-sursilv_rm-rumgr} & {rm-sursilv_rm-sursilv} & {rm-sursilv_rm-sutsilv} & {rm-sursilv_rm-surmiran} & {rm-sursilv_rm-puter} & {rm-sursilv_rm-vallader} \\
{rm-sutsilv_rm-rumgr} & {rm-sutsilv_rm-sursilv} & {rm-sutsilv_rm-sutsilv} & {rm-sutsilv_rm-surmiran} & {rm-sutsilv_rm-puter} & {rm-sutsilv_rm-vallader} \\
{rm-surmiran_rm-rumgr} & {rm-surmiran_rm-sursilv} & {rm-surmiran_rm-sutsilv} & {rm-surmiran_rm-surmiran} & {rm-surmiran_rm-puter} & {rm-surmiran_rm-vallader} \\
{rm-puter_rm-rumgr}   & {rm-puter_rm-sursilv}   & {rm-puter_rm-sutsilv}   & {rm-puter_rm-surmiran}   & {rm-puter_rm-puter}   & {rm-puter_rm-vallader}   \\
{rm-vallader_rm-rumgr} & {rm-vallader_rm-sursilv} & {rm-vallader_rm-sutsilv} & {rm-vallader_rm-surmiran} & {rm-vallader_rm-puter} & {rm-vallader_rm-vallader} \\
\end{tabularx}
"""


def render_table(scores, with_labels: bool, pct_func):
    """Render table with or without y-axis labels"""
    template = TABLE_WITH_LABELS if with_labels else TABLE_WITHOUT_LABELS
    out = template[:]
    for tgt in VARIETIES:
        for ref in VARIETIES:
            val = scores[tgt][ref]
            if isinstance(val, float):
                p = 0.5 * pct_func(val)
                cell = rf"\cellcolor{{uzhblue!{p:.0f}}}{val:.1f}"
            else:
                # Handle "tba" or other non-float values
                cell = str(val)
            out = out.replace(f"{{{tgt}_{ref}}}", cell)
    return out


def main():
    # Load split file from benchmarking directory
    split_path = workspace_root / "benchmarking" / "wmt24pp_split.json"
    document_ids_filter = load_split_document_ids(split_path, "second")
    
    # Compute scores for all systems
    all_scores = {}
    for system_key, (system_path, _) in SYSTEM_MAPPINGS.items():
        if system_path is None:
            print(f"Processing {system_key} (tba)...")
            all_scores[system_key] = compute_scores(None, document_ids_filter)
        else:
            print(f"Processing {system_key}...")
            all_scores[system_key] = compute_scores(system_path, document_ids_filter)
    
    # Compute global min & max across systems with actual scores (exclude "tba" systems)
    all_vals = []
    for scores in all_scores.values():
        for tgt in VARIETIES:
            for ref in VARIETIES:
                val = scores[tgt][ref]
                if isinstance(val, float):
                    all_vals.append(val)
    
    if not all_vals:
        raise ValueError("No valid scores found for normalization")
    
    min_s, max_s = min(all_vals), max(all_vals)
    
    # Normalize to 0–100%
    def pct(val):
        return 0 if not isinstance(val, float) or max_s <= min_s else (val - min_s) / (max_s - min_s) * 100
    
    # Generate LaTeX grid layout
    grid_output = []
    grid_output.append(r"\begin{figure}[H]")
    grid_output.append(r"\centering")
    
    for row_idx, row in enumerate(SYSTEM_ORDER):
        row_subfigures = []
        for col_idx, system_key in enumerate(row):
            if system_key is None:
                # Empty spot - skip
                continue
            
            if system_key not in SYSTEM_MAPPINGS:
                raise ValueError(f"System key {system_key} not found in SYSTEM_MAPPINGS")
            
            system_path, display_name = SYSTEM_MAPPINGS[system_key]
            scores = all_scores[system_key]
            
            # Use labels for first column (col_idx == 0)
            table_content = render_table(scores, with_labels=(col_idx == 0), pct_func=pct)
            
            # Create subfigure
            subfigure_label = f"subfig:confusion_{system_key}"
            subfigure_parts = [
                rf"\begin{{subfigure}}[b]{{0.48\textwidth}}",
                r"\centering",
                table_content,
                rf"\caption{{{display_name}}}",
                rf"\label{{{subfigure_label}}}",
                r"\end{subfigure}"
            ]
            row_subfigures.append("\n".join(subfigure_parts))
        
        # Join row subfigures with \hfill
        if row_subfigures:
            grid_output.append(r"\hfill".join(row_subfigures))
        
        # Add vertical spacing between rows (except after last)
        if row_idx < len(SYSTEM_ORDER) - 1:
            grid_output.append(r"\vspace{0.5cm}")
            grid_output.append("")
    
    grid_output.append(r"\caption{Confusion matrices similar to Figure~\\ref{fig:figure2} illustrating the target variety adherence in German$\rightarrow$Romansh translation. Results are based on BLEU.}")
    grid_output.append(r"\label{fig:confusion_matrices_all_systems}")
    grid_output.append(r"\end{figure}")
    
    # Combine all output
    final_output = "\n".join(grid_output)
    
    # Print to console
    print("\n" + "="*80)
    print("Generated LaTeX output:")
    print("="*80)
    print(final_output)
    
    # Write to output file
    dotenv.load_dotenv()
    if os.getenv("PAPER_DIR") is not None:
        output_path = Path(os.getenv("PAPER_DIR")) / "include" / "target_variety_adherence_all_systems.tex"
        
        if output_path.parent.exists():
            with output_path.open("w", encoding="utf-8") as f:
                f.write(final_output)
            print(f"\nResults written to {output_path}")
        else:
            print(f"\nWarning: Output directory does not exist: {output_path.parent}")
    else:
        print("\nWarning: PAPER_DIR environment variable not set")


if __name__ == "__main__":
    main()

