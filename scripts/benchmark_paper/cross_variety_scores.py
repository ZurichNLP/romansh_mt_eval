import os
from pathlib import Path

import dotenv
from sacrebleu import corpus_chrf

from romansh_mt_eval.benchmarking.constants import VARIETIES
from romansh_mt_eval.benchmarking.evaluation import RomanshWMT24Evaluation

benchmark = RomanshWMT24Evaluation()

# Filter out bad sources
for variety, subset in benchmark.dataset.items():
    benchmark.dataset[variety] = subset.filter(lambda x: not x.get("is_bad_source", False))


# LaTeX table with placeholders {sys_ref}
table = r"""
\begin{tabularx}{\columnwidth}{@{}X|*{6}{>{\raggedleft\arraybackslash}p{0.65cm}}}
\multirow[b]{2}{*}{$\downarrow$~\textbf{sys}}
& \multicolumn{6}{l}{$\textbf{ref}$~$\rightarrow$} \\
& RG & Surs.\ & Suts.\ & Surm.\ & Puter & Vall. \\
\midrule
RG    & {rm-rumgr_rm-rumgr}   & {rm-rumgr_rm-sursilv}   & {rm-rumgr_rm-sutsilv}   & {rm-rumgr_rm-surmiran}   & {rm-rumgr_rm-puter}   & {rm-rumgr_rm-vallader}   \\
Surs.  & {rm-sursilv_rm-rumgr} & {rm-sursilv_rm-sursilv} & {rm-sursilv_rm-sutsilv} & {rm-sursilv_rm-surmiran} & {rm-sursilv_rm-puter} & {rm-sursilv_rm-vallader} \\
Suts.  & {rm-sutsilv_rm-rumgr} & {rm-sutsilv_rm-sursilv} & {rm-sutsilv_rm-sutsilv} & {rm-sutsilv_rm-surmiran} & {rm-sutsilv_rm-puter} & {rm-sutsilv_rm-vallader} \\
Surm. & {rm-surmiran_rm-rumgr} & {rm-surmiran_rm-sursilv} & {rm-surmiran_rm-sutsilv} & {rm-surmiran_rm-surmiran} & {rm-surmiran_rm-puter} & {rm-surmiran_rm-vallader} \\
Puter    & {rm-puter_rm-rumgr}   & {rm-puter_rm-sursilv}   & {rm-puter_rm-sutsilv}   & {rm-puter_rm-surmiran}   & {rm-puter_rm-puter}   & {rm-puter_rm-vallader}   \\
Vall.  & {rm-vallader_rm-rumgr} & {rm-vallader_rm-sursilv} & {rm-vallader_rm-sutsilv} & {rm-vallader_rm-surmiran} & {rm-vallader_rm-puter} & {rm-vallader_rm-vallader} \\
\bottomrule
\end{tabularx}
"""

# Calculate chrF scores between all variety pairs
chrf_scores = {sys: {ref: 0 for ref in VARIETIES} for sys in VARIETIES}

for sys_variety in VARIETIES:
    for ref_variety in VARIETIES:
        if sys_variety == ref_variety:
            # Skip diagonal (same variety)
            continue
        
        score = corpus_chrf(
            benchmark.dataset[sys_variety]["test"]["target"],
            [benchmark.dataset[ref_variety]["test"]["target"]]
        ).score
        chrf_scores[sys_variety][ref_variety] = score

# Determine min and max scores for normalization
min_score = min(
    chrf_scores[sys][ref]
    for sys in VARIETIES
    for ref in VARIETIES
    if sys != ref
)
max_score = 100.0  # Perfect match on diagonal

# Replace each placeholder with a \cellcolor shade + value
for sys_variety in VARIETIES:
    for ref_variety in VARIETIES:
        placeholder = f"{{{sys_variety}_{ref_variety}}}"
        
        if sys_variety == ref_variety:
            # Diagonal: full color, no value
            shade = 75  # 0.75 * 100
            cell = f"\\cellcolor{{uzhapple!{shade}}}"
        else:
            score = chrf_scores[sys_variety][ref_variety]
            # Normalize to [0-100] using min_score and max_score
            if max_score > min_score:
                shade = int(((score - min_score) / (max_score - min_score)) * 100)
            else:
                shade = 0
            shade *= 0.75
            cell = f"\\cellcolor{{uzhapple!{shade}}}{score:.1f}"
        
        table = table.replace(placeholder, cell)

print(table)

dotenv.load_dotenv()
if os.getenv("PAPER_DIR") is not None:
    output_path = Path(os.getenv("PAPER_DIR")) / "latex/include/cross_variety_scores.tex"
    if output_path.exists():
        with output_path.open("w", encoding="utf-8") as f:
            f.write(table)
