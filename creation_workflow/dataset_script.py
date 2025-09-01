import pandas as pd
import datasets
import os


class HFDatasetConfig(datasets.BuilderConfig):
    def __init__(self, name, file_name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.file_name = file_name

class HFDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        HFDatasetConfig(name="en-de_DE", file_name="rm-rumgr.xlsx", version=VERSION, description="basic dataset"),
        HFDatasetConfig(name="rm-rumgr", file_name="rm-rumgr.xlsx", version=VERSION, description="Rumantsch Grischun split"),
        HFDatasetConfig(name="rm-sursilv", file_name="rm-sursilv.xlsx", version=VERSION, description="Rumantsch Sursilv split"),
        HFDatasetConfig(name="rm-vallader", file_name="rm-vallader.xlsx", version=VERSION, description="Rumantsch Vallader split"),
        HFDatasetConfig(name="rm-puter", file_name="rm-puter.xlsx", version=VERSION, description="Rumantsch Puter split"),
        HFDatasetConfig(name="rm-surmiran", file_name="rm-surmiran.xlsx", version=VERSION, description="Rumantsch Surmiran split"),
        HFDatasetConfig(name="rm-sutsilv", file_name="rm-sutsilv.xlsx", version=VERSION, description="Rumantsch Sutsilv split"),
    ]

    def _info(self):
        features={
                "lp": datasets.Value("string"),
                "domain": datasets.Value("string"),
                "document_id": datasets.Value("string"),
                "segment_id": datasets.Value("int32"),
                "is_bad_source": datasets.Value("bool"),
                "source": datasets.Value("string"),
                "target": datasets.Value("string"),
        }
        if self.config.name != "en-de_DE":
            features.update({
                "comment": datasets.Value("string"),
            })

        return datasets.DatasetInfo(
            features=datasets.Features(features),
            supervised_keys=None,
        )


    def _split_generators(self, dl_manager):
        if self.config.name != "en-de_DE":
            excel_path = os.path.join(self.config.data_dir, self.config.file_name)
            if not os.path.exists(excel_path):
                print(f"Warning: Excel file not found at {excel_path}. Skipping dataset config '{self.config.name}'.")
                return []  # <- No split generators if file missing

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN)
        ]

    def _generate_examples(self):
        # Load base dataset from Hugging Face hub
        base_dataset = datasets.load_dataset("google/wmt24pp", "en-de_DE", split="train")

        base_data = {}
        for example in base_dataset:
            segment_id = example.get("segment_id", "")
            if segment_id is not None:
                base_data[str(segment_id)] = {  
                    "lp": example.get("lp", {}),
                    "domain": example.get("domain", ""),
                    "document_id": example.get("document_id", ""),
                    "segment_id": segment_id,
                    "is_bad_source": example.get("is_bad_source", False),
                    "source": example.get("source", ""),
                    "target": example.get("target", ""),
                }
        
        if self.config.name == "en-de_DE":
            for i, (seg_id, row) in enumerate(base_data.items()):
                yield i, row
        else:
            excel_path = os.path.join(self.config.data_dir, self.config.file_name)
            excel_data = {}
            try:
                sheets = pd.ExcelFile(excel_path).sheet_names
                df = pd.concat([
                    pd.read_excel(excel_path, sheet_name=sheet).dropna(subset=["English", "German"], how="any").fillna("")
                    for sheet in ["news", "social", "speech", "literary"] # order like in original HF dataset
                    if sheet in sheets
                ])
            except Exception as e:
                print(f"Warning: Failed to parse Excel file '{excel_path}' for config '{self.config.name}': {e}")
                yield 0, {}
                return

            id_counter = 0
            missing_segments = []
            excel_order = []
            for _, row in df.iterrows():
                seg_id_raw = row.get("segment_id", "")
                seg_id = str(int(seg_id_raw)) if isinstance(seg_id_raw, (int, float)) else str(seg_id_raw)
                romansh_target = row.get("translation", "")
                comment = row.get("comment", "")
                excel_data[seg_id] = {
                    "translation": str(romansh_target),
                    "comment": str(comment),
                }
        
            for base_example in base_dataset:
                segment_id = str(base_example.get("segment_id", ""))
                if segment_id is None:
                    continue
                
                if segment_id in excel_data:
                    target = excel_data[segment_id]["translation"]
                    comment = excel_data[segment_id]["comment"]
                else:
                    target = str(base_example.get("source",""))
                    comment = ""
                    if base_example.get("is_bad_source") is False: # Some row was skipped from Excel although it shouldn't have been
                        missing_segments.append(segment_id)
                excel_order.append(segment_id)
                yield id_counter, {
                            "lp": "de_DE-" + self.config.name,
                            "domain": base_example.get("domain", ""),
                            "document_id": base_example.get("document_id", ""),
                            "segment_id": segment_id,
                            "is_bad_source": base_example.get("is_bad_source", False),
                            "source": base_example.get("target", ""), # German source (from base dataset's target)
                            "target": target,
                            "comment": comment,
                }
                id_counter += 1

            # Additional checks:
            if missing_segments:
                print(f"Warning: {len(missing_segments)} segments in '{self.config.file_name}' were not found in base dataset.")

            expected_order = [str(example.get("segment_id")) for example in base_dataset]
            if excel_order != expected_order:
                print(f"Warning: Row order in Excel file '{self.config.file_name}' does not match the order in the base dataset.")