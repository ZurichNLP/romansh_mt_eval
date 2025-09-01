from copy import deepcopy
import os

import dotenv

from romansh_mt_eval.benchmarking.constants import VARIETIES
from romansh_mt_eval.benchmarking.evaluation import RomanshWMT24Evaluation
from romansh_mt_eval.benchmarking.system_translations import load_llm_translations

METRIC = "chrf"
SYSTEM_KEY = "Gemini-2.5-Flash"
SYSTEM_DISPLAY = "Gemini 2.5 Flash"

benchmark = RomanshWMT24Evaluation()

def modify_benchmark_ref(ref_variety):
    """
    Returns a copy of the benchmark where ref_variety is used a reference for every variety
    """
    modified_benchmark = deepcopy(benchmark)
    for variety in VARIETIES:
        modified_benchmark.dataset[variety] = deepcopy(benchmark.dataset[ref_variety])
    return modified_benchmark

# 3) Collect every numeric score
scores = {tgt: {ref: 0. for ref in VARIETIES} for tgt in VARIETIES}

for ref in VARIETIES:
    modified_benchmark = modify_benchmark_ref(ref)
    system_translations = load_llm_translations(SYSTEM_KEY)
    for tgt in VARIETIES:
        result = modified_benchmark.evaluate(system_translations[tgt], METRIC)
        score = result.scores_de_to_rm.micro_avg
        scores[tgt][ref] = score

# 4) Compute global min & max
all_vals = []
for tgt in VARIETIES:
    for ref in VARIETIES:
        val = scores[tgt][ref]
        if isinstance(val, float):
            all_vals.append(val)
min_s, max_s = min(all_vals), max(all_vals)

# 5) Normalize to 0–100%
def pct(val):
    return 0 if not isinstance(val, float) or max_s <= min_s else (val - min_s) / (max_s - min_s) * 100

# 6) LaTeX template
TABLE = r"""
\begin{tabularx}{\columnwidth}{@{}X|*{6}{>{\raggedleft\arraybackslash}p{0.65cm}}}
\multirow[b]{2}{*}{$\downarrow$~\textbf{tgt}}
& \multicolumn{6}{l}{$\textbf{ref}$~$\rightarrow$} \\
& RG & Surs.\ & Suts.\ & Surm.\ & Puter & Vall. \\
\midrule
RG     & {rm-rumgr_rm-rumgr}   & {rm-rumgr_rm-sursilv}   & {rm-rumgr_rm-sutsilv}   & {rm-rumgr_rm-surmiran}   & {rm-rumgr_rm-puter}   & {rm-rumgr_rm-vallader}   \\
Surs.  & {rm-sursilv_rm-rumgr} & {rm-sursilv_rm-sursilv} & {rm-sursilv_rm-sutsilv} & {rm-sursilv_rm-surmiran} & {rm-sursilv_rm-puter} & {rm-sursilv_rm-vallader} \\
Suts.  & {rm-sutsilv_rm-rumgr} & {rm-sutsilv_rm-sursilv} & {rm-sutsilv_rm-sutsilv} & {rm-sutsilv_rm-surmiran} & {rm-sutsilv_rm-puter} & {rm-sutsilv_rm-vallader} \\
Surm.  & {rm-surmiran_rm-rumgr} & {rm-surmiran_rm-sursilv} & {rm-surmiran_rm-sutsilv} & {rm-surmiran_rm-surmiran} & {rm-surmiran_rm-puter} & {rm-surmiran_rm-vallader} \\
Puter  & {rm-puter_rm-rumgr}   & {rm-puter_rm-sursilv}   & {rm-puter_rm-sutsilv}   & {rm-puter_rm-surmiran}   & {rm-puter_rm-puter}   & {rm-puter_rm-vallader}   \\
Vall.  & {rm-vallader_rm-rumgr} & {rm-vallader_rm-sursilv} & {rm-vallader_rm-sutsilv} & {rm-vallader_rm-surmiran} & {rm-vallader_rm-puter} & {rm-vallader_rm-vallader} \\
\bottomrule
\end{tabularx}
"""

def render_table():
    out = TABLE[:]
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

from pathlib import Path

output_lines = []
output_lines.append(render_table())

# Print to console
for line in output_lines:
    print(line)

dotenv.load_dotenv()
if os.getenv("PAPER_DIR") is not None:
    output_path = Path(os.getenv("PAPER_DIR")) / "latex/include/llm_confusion_matrix.tex"
    if output_path.exists():
        with output_path.open("w", encoding="utf-8") as f:
            for line in output_lines:
                f.write(line + "\n")
