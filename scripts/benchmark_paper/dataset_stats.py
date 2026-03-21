import os
from pathlib import Path
import re

from datasets import load_dataset
import dotenv

from romansh_mt_eval.benchmarking.constants import DOMAINS

variety_names = {
    "rm-rumgr": "RG",
    "rm-sursilv": "Sursilvan",
    "rm-sutsilv": "Sutsilvan",
    "rm-surmiran": "Surmiran",
    "rm-puter": "Puter",
    "rm-vallader": "Vallader",
}

domains = DOMAINS

def tokenize(text):
    # Tokenize based on whitespace and punctuation
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

# Load the original dataset to access the German versions
wmt24pp = load_dataset("google/wmt24pp", "en-de_DE", split="train")

# Count German segments and tokens per domain and total
def count_segments_tokens(rows, text_key, domains=None):
    stats = {}
    if domains is not None:
        for domain in domains:
            filtered = [row for row in rows if row.get("domain") == domain and row[text_key].strip()]
            n_segments = len(filtered)
            n_tokens = sum(len(tokenize(row[text_key])) for row in filtered)
            stats[domain] = (n_segments, n_tokens)
    # Total (all)
    filtered = [row for row in rows if row[text_key].strip()]
    n_segments = len(filtered)
    n_tokens = sum(len(tokenize(row[text_key])) for row in filtered)
    stats["all"] = (n_segments, n_tokens)
    return stats

# German stats
german_stats = count_segments_tokens(wmt24pp, "source", domains=domains)

# Variety stats: for each variety, per domain and total
variety_stats = {}
for variety in variety_names.keys():
    name = variety_names.get(variety, variety)
    ds = load_dataset("ZurichNLP/wmt24pp-rm", f"de_DE-{variety}", split="test")
    if ds is not None:
        # Per domain
        stats = count_segments_tokens(ds, "target", domains=domains)
        variety_stats[name] = stats
    else:
        # tba for all domains and total
        variety_stats[name] = {domain: ("tba", "tba") for domain in domains + ["all"]}

def format_number(n):
    if isinstance(n, int):
        return f"{n:,}".replace(",", "\,")
    return n  # for "tba"

def format_italic(n):
    if isinstance(n, int):
        return r"\textit{" + format_number(n) + "}"
    return n

colspec = "X" + "r" * (len(domains) + 1) + "r" * (len(domains) + 1)  # X + 5 Seg + 5 Tok
lines = []
lines.append(r"\begin{tabularx}{\textwidth}{@{}X" + "r" * (len(domains) + 1) + "r" * (len(domains) + 1) + "@{}}")
lines.append(r"\toprule")

# First header row
header1 = [r"\textbf{Variety}"]
header1.append(r"\multicolumn{" + str(len(domains) + 1) + r"}{c}{\textbf{Segments}}")
header1.append(r"\multicolumn{" + str(len(domains) + 1) + r"}{c}{\textbf{Tokens}}")
lines.append(" & ".join(header1) + r" \\")
lines.append(
    r"\cmidrule(lr){2-" + str(1 + len(domains) + 1) + r"}\cmidrule(lr){" + str(2 + len(domains) + 1) + "-" + str(1 + 2 * (len(domains) + 1)) + r"}"
)

# Second header row
header2 = [""]
for _ in range(2):  # Segments, Tokens
    for i, domain in enumerate(domains):
        if domain == "literary":
            header2.append(r"\textbf{Lit.}")
        elif domain == "social":
            header2.append(r"\textbf{Soc.}")
        else:
            header2.append(r"\textbf{" + domain.capitalize() + "}")
    header2.append(r"\textbf{Total}")
lines.append(" & ".join(header2) + r" \\")
lines.append(r"\midrule")

# German row
german_row = [r"German \mbox{\cite{deutsch2025wmt24expandinglanguagecoverage}}"]
# Segments for each domain
for domain in domains:
    segs, _ = german_stats.get(domain, ("tba", "tba"))
    german_row.append(format_number(segs))
# Segments total
segs, _ = german_stats.get("all", ("tba", "tba"))
german_row.append(format_number(segs))
# Tokens for each domain
for domain in domains:
    _, toks = german_stats.get(domain, ("tba", "tba"))
    german_row.append(format_number(toks))
# Tokens total
_, toks = german_stats.get("all", ("tba", "tba"))
german_row.append(format_number(toks))
lines.append(" & ".join(german_row) + r" \\")
lines.append(r"\midrule")

# Rows for varieties
for name, stats in variety_stats.items():
    row = [name]
    all_segs, _ = stats.get("all", ("tba", "tba"))
    # Segments for each domain
    for domain in domains:
        segs, _ = stats.get(domain, ("tba", "tba"))
        if isinstance(all_segs, int) and all_segs < 998:
            seg_str = format_italic(segs)
        else:
            seg_str = format_number(segs)
        row.append(seg_str)
    # Segments total
    segs, _ = stats.get("all", ("tba", "tba"))
    if isinstance(all_segs, int) and all_segs < 998:
        seg_str = format_italic(segs)
    else:
        seg_str = format_number(segs)
    row.append(seg_str)
    # Tokens for each domain
    for domain in domains:
        _, toks = stats.get(domain, ("tba", "tba"))
        if isinstance(all_segs, int) and all_segs < 998:
            tok_str = format_italic(toks)
        else:
            tok_str = format_number(toks)
        row.append(tok_str)
    # Tokens total
    _, toks = stats.get("all", ("tba", "tba"))
    if isinstance(all_segs, int) and all_segs < 998:
        tok_str = format_italic(toks)
    else:
        tok_str = format_number(toks)
    row.append(tok_str)
    lines.append(" & ".join(row) + r" \\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabularx}")

table = "\n".join(lines)

print(table)

dotenv.load_dotenv()
if os.getenv("PAPER_DIR") is not None:
    output_path = Path(os.getenv("PAPER_DIR")) / "latex/include/dataset_stats.tex"
    if output_path.exists():
        with output_path.open("w", encoding="utf-8") as f:
            f.write(table)
