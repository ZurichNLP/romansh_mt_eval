import os
import sys
from pathlib import Path

import dotenv
import fasttext

from romansh_mt_eval.benchmarking.constants import VARIETIES
from romansh_mt_eval.benchmarking.evaluation import RomanshWMT24Evaluation

if len(sys.argv) != 2:
    raise ValueError("Usage: python language_classifier_confusion_matrix.py /path/to/model.bin")
classifier_path = Path(sys.argv[1])
assert classifier_path.exists(), f"Classifier not found at {classifier_path}"

classifier = fasttext.load_model(str(classifier_path))

benchmark = RomanshWMT24Evaluation()

for variety, subset in benchmark.dataset.items():
    benchmark.dataset[variety] = subset.filter(lambda x: not x.get("is_bad_source", False))

# LaTeX table with placeholders {pred_gold}
table = r"""
\begin{tabularx}{\columnwidth}{@{}X|*{6}{>{\raggedleft\arraybackslash}p{0.65cm}}}
\multirow[b]{2}{*}{$\downarrow$~\textbf{pred}}
& \multicolumn{6}{l}{$\textbf{gold}$~$\rightarrow$} \\
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

# Compute confusion matrix: confusion[gold][pred] = count
confusion = {gold: {pred: 0 for pred in VARIETIES} for gold in VARIETIES}
for gold_variety in VARIETIES:
    segments = [item["target"] for item in benchmark.dataset[gold_variety]["test"]]
    if not segments:
        continue
    for seg in segments:
        seg = seg.strip().replace("“", '"').replace("”", '"').replace("«", '"').replace("»", '"').replace("\n", " ")
        pred_label = classifier.predict(seg)[0][0]
        if pred_label.startswith("__label__"):
            pred_variety = pred_label[len("__label__"):]
        else:
            pred_variety = pred_label
        pred_variety = pred_variety.replace("_", "-")
        if pred_variety in VARIETIES:
            confusion[gold_variety][pred_variety] += 1

# Determine maximum count for normalization
max_count = max(
    confusion[gold][pred]
    for gold in VARIETIES
    for pred in VARIETIES
)

# Replace each placeholder with a \cellcolor shade + value
for gold_variety in VARIETIES:
    for pred_variety in VARIETIES:
        value = confusion[gold_variety][pred_variety]
        shade = int((value / max_count) * 100) if max_count else 0
        shade *= 0.75
        cell = f"\\cellcolor{{uzhapple!{shade}}}{value}"
        table = table.replace(f"{{{pred_variety}_{gold_variety}}}", cell)

print(table)

dotenv.load_dotenv()
if os.getenv("PAPER_DIR") is not None:
    output_path = Path(os.getenv("PAPER_DIR")) / "latex/include/language_classifier_confusion_matrix.tex"
    if output_path.exists():
        with output_path.open("w", encoding="utf-8") as f:
            f.write(table)
