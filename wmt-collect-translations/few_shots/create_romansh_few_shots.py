import csv
import json
import os

romansh_column_to_code = {
    "Rumantsch Grischun": "rm_rumgr",
    "Sursilvan": "rm_sursilv",
    "Vallader": "rm_vallader",
    "Puter": "rm_puter",
    "Surmiran": "rm_surmiran",
    "Sutsilvan": "rm_sutsilv",
}

# Path to the TSV file
tsv_path = os.path.join(os.path.dirname(__file__), "romansh_few_shots.tsv")

# Read the TSV file
with open(tsv_path, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    rows = list(reader)

# For each variety, create JSON files in both directions
for column, code in romansh_column_to_code.items():
    # German to Romansh
    data_de_to_rm = []
    # Romansh to German
    data_rm_to_de = []
    for row in rows:
        source_de = row["German"].strip()
        target_rm = row[column].strip()
        if source_de and target_rm:
            data_de_to_rm.append({"source": source_de, "target": target_rm})
            data_rm_to_de.append({"source": target_rm, "target": source_de})

    # Write German to Romansh
    json_filename_de_to_rm = f"shots.de-{code}.json"
    json_path_de_to_rm = os.path.join(os.path.dirname(__file__), json_filename_de_to_rm)
    with open(json_path_de_to_rm, "w", encoding="utf-8") as jf:
        json.dump(data_de_to_rm, jf, ensure_ascii=False, indent=2)

    # Write Romansh to German
    json_filename_rm_to_de = f"shots.{code}-de.json"
    json_path_rm_to_de = os.path.join(os.path.dirname(__file__), json_filename_rm_to_de)
    with open(json_path_rm_to_de, "w", encoding="utf-8") as jf:
        json.dump(data_rm_to_de, jf, ensure_ascii=False, indent=2)
