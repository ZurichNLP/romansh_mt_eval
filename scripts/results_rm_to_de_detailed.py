import os
from pathlib import Path

import dotenv

from romansh_mt_eval.benchmarking.constants import VARIETIES, DOMAINS
from romansh_mt_eval.benchmarking.system_results import get_all_system_results

METRIC1 = "chrf"
METRIC2 = "xcomet-xl"

TEMPLATE = r"""\subsection{{variety_display} to German}

\begin{table}[H]
\centering
\begin{tabularx}{\textwidth}{@{}Xrrrrr@{}}
\toprule
\textbf{System} & \textbf{Literary} & \textbf{News} & \textbf{Social} & \textbf{Speech} & \textbf{Macro-Average} \\
\midrule
\mbox{MADLAD-400 10.7B}      & & & & & \\
\mbox{– direct} & {madlad400-10b-mt_direct_literary_{variety}} & {madlad400-10b-mt_direct_news_{variety}} & {madlad400-10b-mt_direct_social_{variety}} & {madlad400-10b-mt_direct_speech_{variety}} & {madlad400-10b-mt_direct_macro_{variety}} \\
\mbox{– pivoting via English} & {madlad400-10b-mt_pivot_en_literary_{variety}} & {madlad400-10b-mt_pivot_en_news_{variety}} & {madlad400-10b-mt_pivot_en_social_{variety}} & {madlad400-10b-mt_pivot_en_speech_{variety}} & {madlad400-10b-mt_pivot_en_macro_{variety}} \\
\midrule
Supertext     & {supertext_literary_{variety}} & {supertext_news_{variety}} & {supertext_social_{variety}} & {supertext_speech_{variety}} & {supertext_macro_{variety}} \\
\midrule
Llama 3.3 (70B)        & {Llama-3.3-70b_literary_{variety}} & {Llama-3.3-70b_news_{variety}} & {Llama-3.3-70b_social_{variety}} & {Llama-3.3-70b_speech_{variety}} & {Llama-3.3-70b_macro_{variety}} \\
GPT-4o        & {GPT-4o_literary_{variety}} & {GPT-4o_news_{variety}} & {GPT-4o_social_{variety}} & {GPT-4o_speech_{variety}} & {GPT-4o_macro_{variety}} \\
Gemini 2.5 Flash        & {Gemini-2.5-Flash_literary_{variety}} & {Gemini-2.5-Flash_news_{variety}} & {Gemini-2.5-Flash_social_{variety}} & {Gemini-2.5-Flash_speech_{variety}} & {Gemini-2.5-Flash_macro_{variety}} \\
\bottomrule
\end{tabularx}
\caption{ChrF / xCOMET scores for translation from {variety_display} into German.}
\end{table}
"""

results_metric1 = get_all_system_results(METRIC1)
results_metric2 = get_all_system_results(METRIC2)

system_names = [
    "madlad400-10b-mt_direct",
    "madlad400-10b-mt_pivot_en",
    "supertext",
    "GPT-4o",
    "Llama-3.3-70b",
    "Gemini-2.5-Flash",
]

all_output = []

for i, (variety, variety_display) in enumerate(list(VARIETIES.items())):
    # Build a dict: scores[domain][sys_name] = score (float or string)
    scores_metric1 = {domain: {} for domain in DOMAINS + ["macro"]}
    for sys_name in system_names:
        for domain in DOMAINS + ["macro"]:
            result = results_metric1[sys_name][variety]
            if domain == "macro":
                score = result.scores_rm_to_de.macro_avg
            else:
                score = result.scores_rm_to_de.domain_results[domain]
            scores_metric1[domain][sys_name] = score
    scores_metric2 = {domain: {} for domain in DOMAINS + ["macro"]}
    for sys_name in system_names:
        for domain in DOMAINS + ["macro"]:
            result = results_metric2[sys_name][variety]
            if domain == "macro":
                score = result.scores_rm_to_de.macro_avg
            else:
                score = result.scores_rm_to_de.domain_results[domain]
            scores_metric2[domain][sys_name] = score


    # For each domain, find the highest rounded float score (ignore "tbd" and "-")
    # First, collect all rounded float values per domain (for metric1)
    rounded_scores_metric1 = {domain: {} for domain in scores_metric1}
    for domain in scores_metric1:
        for sys_name in system_names:
            val = scores_metric1[domain][sys_name]
            if isinstance(val, float):
                sval = float(f"{val:.1f}")
                rounded_scores_metric1[domain][sys_name] = sval
            else:
                rounded_scores_metric1[domain][sys_name] = None

    # For each domain, find the max rounded value (for metric1)
    max_scores_metric1 = {}
    for domain in scores_metric1:
        max_val = None
        for sys_name in system_names:
            sval = rounded_scores_metric1[domain][sys_name]
            if sval is not None:
                if max_val is None or sval > max_val:
                    max_val = sval
        max_scores_metric1[domain] = max_val

    # For each domain, collect all sys_names with the max rounded value (for metric1)
    max_sysnames_metric1 = {domain: set() for domain in scores_metric1}
    for domain in scores_metric1:
        for sys_name in system_names:
            sval = rounded_scores_metric1[domain][sys_name]
            if sval is not None and max_scores_metric1[domain] is not None and sval == max_scores_metric1[domain]:
                max_sysnames_metric1[domain].add(sys_name)

    # Format all scores as strings, bolding all with the max rounded value per column (for metric1)
    formatted_scores_metric1 = {domain: {} for domain in scores_metric1}
    for domain in scores_metric1:
        for sys_name in system_names:
            val = scores_metric1[domain][sys_name]
            if isinstance(val, float):
                sval = f"{val:.1f}"
                if sys_name in max_sysnames_metric1[domain]:
                    sval = r"\textbf{" + sval + "}"
            else:
                sval = val
            formatted_scores_metric1[domain][sys_name] = sval

    # For each domain, find the highest rounded float score (ignore "tbd" and "-") (for metric2)
    # First, collect all rounded float values per domain
    rounded_scores_metric2 = {domain: {} for domain in scores_metric2}
    for domain in scores_metric2:
        for sys_name in system_names:
            val = scores_metric2[domain][sys_name]
            if isinstance(val, float):
                sval = float(f"{val:.1f}")
                rounded_scores_metric2[domain][sys_name] = sval
            else:
                rounded_scores_metric2[domain][sys_name] = None

    # For each domain, find the max rounded value (for metric2)
    max_scores_metric2 = {}
    for domain in scores_metric2:
        max_val = None
        for sys_name in system_names:
            sval = rounded_scores_metric2[domain][sys_name]
            if sval is not None:
                if max_val is None or sval > max_val:
                    max_val = sval
        max_scores_metric2[domain] = max_val

    # For each domain, collect all sys_names with the max rounded value (for metric2)
    max_sysnames_metric2 = {domain: set() for domain in scores_metric2}
    for domain in scores_metric2:
        for sys_name in system_names:
            sval = rounded_scores_metric2[domain][sys_name]
            if sval is not None and max_scores_metric2[domain] is not None and sval == max_scores_metric2[domain]:
                max_sysnames_metric2[domain].add(sys_name)

    # Format all scores as strings, bolding all with the max rounded value per column (for metric2)
    formatted_scores_metric2 = {domain: {} for domain in scores_metric2}
    for domain in scores_metric2:
        for sys_name in system_names:
            val = scores_metric2[domain][sys_name]
            if isinstance(val, float):
                sval = f"{val:.1f}"
                if sys_name in max_sysnames_metric2[domain]:
                    sval = r"\textbf{" + sval + "}"
            else:
                sval = val
            formatted_scores_metric2[domain][sys_name] = sval

    # Fill in the template with both metric results, separated by a slash
    template = TEMPLATE.replace("{variety}", variety).replace("{variety_display}", variety_display)
    for domain in scores_metric1:
        for sys_name in system_names:
            sval1 = formatted_scores_metric1[domain][sys_name]
            sval2 = formatted_scores_metric2[domain][sys_name]
            sval = f"{sval1} / {sval2}"
            template = template.replace(
                f"{{{sys_name}_{domain}_{variety}}}",
                sval
            )
    print(template)
    all_output.append(template)
    if i % 3 == 2:
        print(r"\vfill")
        print(r"\clearpage")
        all_output.append(r"\vfill")
        all_output.append(r"\clearpage")

dotenv.load_dotenv()
if os.getenv("PAPER_DIR") is not None:
    output_path = Path(os.getenv("PAPER_DIR")) / "latex/include/results_rm_to_de_detailed.tex"
    if output_path.exists():
        with output_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(all_output))

