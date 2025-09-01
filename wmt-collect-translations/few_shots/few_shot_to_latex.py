# Format few-shot examples for paper appendix

import csv
import os

tsv_path = os.path.join(os.path.dirname(__file__), "romansh_few_shots.tsv")

# Read the TSV file
with open(tsv_path, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    rows = list(reader)

for language in rows[0].keys():
    s = rf"""
\item[{language}:]
\item {rows[0][language].strip()}
\item {rows[1][language].strip()}
\item {rows[2][language].strip()}"""
    print(s)
