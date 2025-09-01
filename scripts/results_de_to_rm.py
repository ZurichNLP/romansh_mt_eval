import os
from pathlib import Path

import dotenv

from romansh_mt_eval.benchmarking.constants import VARIETIES
from romansh_mt_eval.benchmarking.system_results import get_all_system_results

METRIC = "chrf"

TEMPLATE = r"""
\begin{tabularx}{\textwidth}{@{}Xrrrrrr@{}}
\toprule
\textbf{System} & \textbf{Rumantsch Grischun} & \textbf{Sursilvan} & \textbf{Sutsilvan} & \textbf{Surmiran} & \textbf{Puter} & \textbf{Vallader} \\
\midrule
\mbox{MADLAD-400 (10.7B)}      & & & & & & \\
\mbox{– direct} & {madlad400-10b-mt_direct_rm-rumgr} & \textcolor{graycolor}{{madlad400-10b-mt_direct_rm-sursilv}} & \textcolor{graycolor}{{madlad400-10b-mt_direct_rm-sutsilv}} & \textcolor{graycolor}{{madlad400-10b-mt_direct_rm-surmiran}} & \textcolor{graycolor}{{madlad400-10b-mt_direct_rm-puter}} & \textcolor{graycolor}{{madlad400-10b-mt_direct_rm-vallader}} \\
\mbox{– pivoting via English} & {madlad400-10b-mt_pivot_en_rm-rumgr} & \textcolor{graycolor}{{madlad400-10b-mt_pivot_en_rm-sursilv}} & \textcolor{graycolor}{{madlad400-10b-mt_pivot_en_rm-sutsilv}} & \textcolor{graycolor}{{madlad400-10b-mt_pivot_en_rm-surmiran}} & \textcolor{graycolor}{{madlad400-10b-mt_pivot_en_rm-puter}} & \textcolor{graycolor}{{madlad400-10b-mt_pivot_en_rm-vallader}} \\
\midrule
\mbox{Translatur-ia}     & {translaturia_rm-rumgr} & \textcolor{graycolor}{{translaturia_rm-sursilv}} & \textcolor{graycolor}{{translaturia_rm-sutsilv}} & \textcolor{graycolor}{{translaturia_rm-surmiran}} & \textcolor{graycolor}{{translaturia_rm-puter}} & \textcolor{graycolor}{{translaturia_rm-vallader}} \\
Supertext     & {supertext_rm-rumgr} & \textcolor{graycolor}{{supertext_rm-sursilv}} & \textcolor{graycolor}{{supertext_rm-sutsilv}} & \textcolor{graycolor}{{supertext_rm-surmiran}} & \textcolor{graycolor}{{supertext_rm-puter}} & \textcolor{graycolor}{{supertext_rm-vallader}} \\
\midrule
\mbox{Llama 3.3 (70B)}        & {Llama-3.3-70b_rm-rumgr} & {Llama-3.3-70b_rm-sursilv} & {Llama-3.3-70b_rm-sutsilv} & {Llama-3.3-70b_rm-surmiran} & {Llama-3.3-70b_rm-puter} & {Llama-3.3-70b_rm-vallader} \\
GPT-4o        & {GPT-4o_rm-rumgr} & {GPT-4o_rm-sursilv} & {GPT-4o_rm-sutsilv} & {GPT-4o_rm-surmiran} & {GPT-4o_rm-puter} & {GPT-4o_rm-vallader} \\
\mbox{Gemini 2.5 Flash}        & {Gemini-2.5-Flash_rm-rumgr} & {Gemini-2.5-Flash_rm-sursilv} & {Gemini-2.5-Flash_rm-sutsilv} & {Gemini-2.5-Flash_rm-surmiran} & {Gemini-2.5-Flash_rm-puter} & {Gemini-2.5-Flash_rm-vallader} \\
\bottomrule
\end{tabularx}
"""

results = get_all_system_results(METRIC)

system_names = [
    "madlad400-10b-mt_direct",
    "madlad400-10b-mt_pivot_en",
    "translaturia",
    "supertext",
    "GPT-4o",
    "Llama-3.3-70b",
    "Gemini-2.5-Flash",
]

# Build a dict: scores[variety][sys_name] = score (as float or string)
varieties = list(VARIETIES.keys())
scores = {variety: {} for variety in varieties}
for sys_name, varieties_results in results.items():
    for variety in varieties:
        result = varieties_results[variety]
        score = result.scores_de_to_rm.micro_avg
        scores[variety][sys_name] = score

# For each variety, find the highest rounded float score (ignore "tbd" and "-")
# and collect all sys_names that reach this rounded value
max_scores_rounded = {}
for variety in varieties:
    rounded_vals = []
    for sys_name in scores[variety]:
        val = scores[variety][sys_name]
        if isinstance(val, float):
            rounded_vals.append(round(val, 1))
    max_val = max(rounded_vals) if rounded_vals else None
    max_scores_rounded[variety] = max_val

# Now, format all scores as strings, bolding all that match the max rounded value
formatted_scores = {variety: {} for variety in varieties}
for variety in varieties:
    max_val = max_scores_rounded[variety]
    for sys_name in scores[variety]:
        val = scores[variety][sys_name]
        if isinstance(val, float):
            sval = f"{val:.1f}"
            if max_val is not None and round(val, 1) == max_val:
                sval = r"\textbf{" + sval + "}"
        else:
            sval = val
        formatted_scores[variety][sys_name] = sval

# Replace in template
filled_template = TEMPLATE
for variety in varieties:
    for sys_name in system_names:
        key = f"{sys_name}_{variety}"
        # Some sys_names in template may not be present in results, so fallback to "-"
        sval = formatted_scores[variety].get(sys_name, "-")
        filled_template = filled_template.replace(f"{{{key}}}", r"\phantom{0.00 / }" + sval)

print(filled_template)

dotenv.load_dotenv()
if os.getenv("PAPER_DIR") is not None:
    output_path = Path(os.getenv("PAPER_DIR")) / "latex/include/results_de_to_rm.tex"
    if output_path.exists():
        with output_path.open("w", encoding="utf-8") as f:
            f.write(filled_template)
