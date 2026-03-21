import json
from copy import deepcopy
from pathlib import Path

from romansh_mt_eval.benchmarking.constants import VARIETIES
from romansh_mt_eval.benchmarking.evaluation import RomanshWMT24Evaluation
from romansh_mt_eval.benchmarking.system_translations import load_llm_translations

METRIC = "bleu"
MODEL_KEY_1 = "Gemini-3-Pro"
MODEL_KEY_2 = "ctranslate2_fairseq_nllb-200-distilled-1.3B.norm.temp1.5.10e.withdict.dict_prompting_ct2"

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


# Load document IDs filter for second half
script_dir = Path(__file__).resolve().parent
workspace_root = script_dir.parent.parent
split_path = workspace_root / "benchmarking" / "wmt24pp_split.json"
document_ids_filter = load_split_document_ids(split_path, "second")

def modify_benchmark_ref(ref_variety):
    """
    Returns a copy of the benchmark where ref_variety is used a reference for every variety
    """
    modified_benchmark = deepcopy(benchmark)
    for variety in VARIETIES:
        modified_benchmark.dataset[variety] = deepcopy(benchmark.dataset[ref_variety])
    return modified_benchmark

def compute_scores(model_key):
    """
    Compute scores for a given model key.
    Returns a dictionary of scores: {tgt: {ref: score}}
    """
    scores = {tgt: {ref: 0. for ref in VARIETIES} for tgt in VARIETIES}
    
    # Prepend mt_paper second_half if evaluating second half and not already prefixed
    if document_ids_filter is not None:
        prefix = "system_translations/mt_paper/second_half/"
        if not model_key.startswith(prefix) and not model_key.startswith("system_translations/"):
            model_key = f"{prefix}{model_key}"
    
    for ref in VARIETIES:
        modified_benchmark = modify_benchmark_ref(ref)
        system_translations = load_llm_translations(model_key, document_ids_filter=document_ids_filter)
        for tgt in VARIETIES:
            result = modified_benchmark.evaluate(system_translations[tgt], METRIC, document_ids_filter=document_ids_filter)
            score = result.scores_de_to_rm.micro_avg
            scores[tgt][ref] = score
    
    return scores

# Compute scores for both models
scores_model1 = compute_scores(MODEL_KEY_1)
scores_model2 = compute_scores(MODEL_KEY_2)

# Compute global min & max across both models for consistent color scale
all_vals = []
for scores in [scores_model1, scores_model2]:
    for tgt in VARIETIES:
        for ref in VARIETIES:
            val = scores[tgt][ref]
            if isinstance(val, float):
                all_vals.append(val)

min_s, max_s = min(all_vals), max(all_vals)

# Normalize to 0–100%
def pct(val):
    return 0 if not isinstance(val, float) or max_s <= min_s else (val - min_s) / (max_s - min_s) * 100

# LaTeX template with y-axis labels (for first model)
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

# LaTeX template without y-axis labels (for second model)
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

def render_table_with_labels(scores):
    """Render table with y-axis labels for first model"""
    out = TABLE_WITH_LABELS[:]
    for tgt in VARIETIES:
        for ref in VARIETIES:
            val = scores[tgt][ref]
            if isinstance(val, float):
                p = 0.5 * pct(val)
                cell = rf"\cellcolor{{uzhblue!{p:.0f}}}{val:.1f}"
            else:
                cell = val
            out = out.replace(f"{{{tgt}_{ref}}}", cell)
    return out

def render_table_without_labels(scores):
    """Render table without y-axis labels for second model"""
    out = TABLE_WITHOUT_LABELS[:]
    for tgt in VARIETIES:
        for ref in VARIETIES:
            val = scores[tgt][ref]
            if isinstance(val, float):
                p = 0.5 * pct(val)
                cell = rf"\cellcolor{{uzhblue!{p:.0f}}}{val:.1f}"
            else:
                cell = val
            out = out.replace(f"{{{tgt}_{ref}}}", cell)
    return out

# Generate output for both models
output_model1 = render_table_with_labels(scores_model1)
output_model2 = render_table_without_labels(scores_model2)

# Print to console
print("Model 1 output:")
print(output_model1)
print("\nModel 2 output:")
print(output_model2)

# Save to paper project
paper_include_dir = script_dir.parent.parent.parent / "rm-mt-paper" / "include"

if paper_include_dir.exists():
    output_path_1 = paper_include_dir / f"llm_confusion_matrix_{MODEL_KEY_1}.tex"
    output_path_2 = paper_include_dir / f"llm_confusion_matrix_{MODEL_KEY_2}.tex"
    
    with output_path_1.open("w", encoding="utf-8") as f:
        f.write(output_model1)
    
    with output_path_2.open("w", encoding="utf-8") as f:
        f.write(output_model2)
    
    print(f"\nSaved to:")
    print(f"  {output_path_1}")
    print(f"  {output_path_2}")
else:
    print(f"\nWarning: Paper include directory not found at {paper_include_dir}")
