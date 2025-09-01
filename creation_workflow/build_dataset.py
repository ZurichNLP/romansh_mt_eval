from pathlib import Path

from datasets import load_dataset

from romansh_mt_eval.benchmarking.constants import VARIETIES


def main():
    out_dir = Path(__file__).parent / "dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    for config_name in ["en-de_DE"] + list(VARIETIES):
        dataset_script_path = Path(__file__).parent / "dataset_script.py"
        assert dataset_script_path.exists(), f"Dataset script not found at {dataset_script_path}"
        dataset_dir = Path(__file__).parent / "completed_xlsx"
        dataset = load_dataset(
            str(dataset_script_path),
            name=config_name,
            data_dir=str(dataset_dir),
            trust_remote_code=True,
            download_mode="force_redownload",
        )
        if config_name == "en-de_DE":
            split_save_path = out_dir / f"{config_name}.jsonl"
        else:
            split_save_path = out_dir / f"de_DE-{config_name}.jsonl"
        dataset["train"].to_json(split_save_path)

if __name__ == "__main__":
    main()
