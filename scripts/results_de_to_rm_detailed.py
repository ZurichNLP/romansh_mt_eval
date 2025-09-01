import os
from pathlib import Path

import dotenv

from romansh_mt_eval.benchmarking.constants import VARIETIES, DOMAINS
from romansh_mt_eval.benchmarking.system_results import get_all_system_results

METRIC = "chrf"

TEMPLATE_RG_ONLY = r"""
\mbox{MADLAD-400 10.7B}      & & & & & \\
\mbox{– direct} & {madlad400-10b-mt_direct_literary_{variety}} & {madlad400-10b-mt_direct_news_{variety}} & {madlad400-10b-mt_direct_social_{variety}} & {madlad400-10b-mt_direct_speech_{variety}} & {madlad400-10b-mt_direct_macro_{variety}} \\
\mbox{– pivoting via English} & {madlad400-10b-mt_pivot_en_literary_{variety}} & {madlad400-10b-mt_pivot_en_news_{variety}} & {madlad400-10b-mt_pivot_en_social_{variety}} & {madlad400-10b-mt_pivot_en_speech_{variety}} & {madlad400-10b-mt_pivot_en_macro_{variety}} \\
\midrule
Translatur-ia     & {translaturia_literary_{variety}} & {translaturia_news_{variety}} & {translaturia_social_{variety}} & {translaturia_speech_{variety}} & {translaturia_macro_{variety}} \\
Supertext     & {supertext_literary_{variety}} & {supertext_news_{variety}} & {supertext_social_{variety}} & {supertext_speech_{variety}} & {supertext_macro_{variety}} \\
\midrule
"""

TEMPLATE = r"""\subsection{German to {variety_display}}

\begin{table}[H]
\centering
\begin{tabularx}{\textwidth}{@{}Xrrrrr@{}}
\toprule
\textbf{System} & \textbf{Literary} & \textbf{News} & \textbf{Social} & \textbf{Speech} & \textbf{Macro-Average} \\
\midrule
{rg_only}
Llama 3.3 (70B)        & {Llama-3.3-70b_literary_{variety}} & {Llama-3.3-70b_news_{variety}} & {Llama-3.3-70b_social_{variety}} & {Llama-3.3-70b_speech_{variety}} & {Llama-3.3-70b_macro_{variety}} \\
GPT-4o        & {GPT-4o_literary_{variety}} & {GPT-4o_news_{variety}} & {GPT-4o_social_{variety}} & {GPT-4o_speech_{variety}} & {GPT-4o_macro_{variety}} \\
Gemini 2.5 Flash        & {Gemini-2.5-Flash_literary_{variety}} & {Gemini-2.5-Flash_news_{variety}} & {Gemini-2.5-Flash_social_{variety}} & {Gemini-2.5-Flash_speech_{variety}} & {Gemini-2.5-Flash_macro_{variety}} \\
\bottomrule
\end{tabularx}
\caption{ChrF scores for translation from German into {variety_display}.}
\end{table}
"""

results = get_all_system_results(METRIC)

rg_only_system_names = [
    "madlad400-10b-mt_direct",
    "madlad400-10b-mt_pivot_en",
    "translaturia",
    "supertext",
]
all_system_names = [
    "GPT-4o",
    "Llama-3.3-70b",
    "Gemini-2.5-Flash",
]
system_names = rg_only_system_names + all_system_names

all_output = []

for variety, variety_display in VARIETIES.items():
    # Build a dict: scores[domain][sys_name] = score (float or string)
    scores = {domain: {} for domain in DOMAINS}
    for sys_name in system_names:
        for domain in DOMAINS:
            result = results[sys_name][variety]
            score = result.scores_de_to_rm.domain_results[domain]
            scores[domain][sys_name] = score
    scores["macro"] = {}
    for sys_name in system_names:
        vals = [scores[domain][sys_name] for domain in DOMAINS if isinstance(scores[domain][sys_name], float)]
        if vals:
            scores["macro"][sys_name] = sum(vals) / len(vals)
        else:
            scores["macro"][sys_name] = "-"


    # For each domain, find the highest rounded float score (ignore "tbd" and "-")
    # First, collect all rounded float values per domain
    rounded_scores = {domain: {} for domain in scores}
    for domain in scores:
        for sys_name in system_names:
            val = scores[domain][sys_name]
            if isinstance(val, float):
                sval = float(f"{val:.1f}")
                rounded_scores[domain][sys_name] = sval
            else:
                rounded_scores[domain][sys_name] = None

    # For each domain, find the max rounded value
    max_scores = {}
    for domain in scores:
        max_val = None
        for sys_name in system_names:
            sval = rounded_scores[domain][sys_name]
            if sval is not None:
                if max_val is None or sval > max_val:
                    max_val = sval
        max_scores[domain] = max_val

    # For each domain, collect all sys_names with the max rounded value
    max_sysnames = {domain: set() for domain in scores}
    for domain in scores:
        for sys_name in system_names:
            sval = rounded_scores[domain][sys_name]
            if sval is not None and max_scores[domain] is not None and sval == max_scores[domain]:
                max_sysnames[domain].add(sys_name)

    # Format all scores as strings, bolding all with the max rounded value per column
    formatted_scores = {domain: {} for domain in scores}
    for domain in scores:
        for sys_name in system_names:
            val = scores[domain][sys_name]
            if isinstance(val, float):
                sval = f"{val:.1f}"
                if sys_name in max_sysnames[domain]:
                    sval = r"\textbf{" + sval + "}"
            else:
                sval = val
            formatted_scores[domain][sys_name] = sval

    # Prepare the open source section only for Rumantsch Grischun
    if variety == "rm-rumgr":
        rg_only = TEMPLATE_RG_ONLY
    else:
        rg_only = ""

    # Fill in the template
    template = TEMPLATE.replace("{rg_only}", rg_only).replace("{variety}", variety).replace("{variety_display}", variety_display)
    for domain in scores:
        for sys_name in system_names:
            template = template.replace(
                f"{{{sys_name}_{domain}_{variety}}}",
                r"\phantom{0.00 / }" + formatted_scores[domain][sys_name]
            )
    print(template)
    all_output.append(template)

dotenv.load_dotenv()
if os.getenv("PAPER_DIR") is not None:
    output_path = Path(os.getenv("PAPER_DIR")) / "latex/include/results_de_to_rm_detailed.tex"
    if output_path.exists():
        with output_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(all_output))
