import os

from dataset_parser import DatasetParser

from romansh_mt_eval.benchmarking.constants import VARIETIES
from worksheet_writer import WorksheetWriter

COLUMNS = [
    "source",
    "document_id",
    "segment_id",
    "url",
    "target",
    "translation",
    "comment",
]

parser = DatasetParser("google/wmt24pp", "google/wmt24pp-images")
merged_df = parser.load_datasets()

dir_name = "blank_xlsx"
os.makedirs(dir_name, exist_ok=True)

writer = WorksheetWriter(merged_df, COLUMNS, dir_name)
for variety in VARIETIES:
    writer.create_worksheet(variety)

print(f"Worksheets created in directory: {dir_name}")
