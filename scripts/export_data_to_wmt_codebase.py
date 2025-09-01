from pathlib import Path

from datasets import load_dataset

from romansh_mt_eval.benchmarking.constants import VARIETIES


wmt_dir = Path(__file__).parent.parent / "wmt-collect-translations"
assert wmt_dir.exists(), f"WMT directory not found: {wmt_dir}"

out_dir = wmt_dir / "wmt_testset_romansh"
if not out_dir.exists():
    out_dir.mkdir(parents=True)

for variety in VARIETIES:
    dataset = load_dataset("ZurichNLP/wmt24pp-rm", f"de_DE-{variety}")["test"]
    de_path = out_dir / f"wmttest2024.src.de-{variety.replace('-', '_')}.xml.no-testsuites.de"
    with open(de_path, "w", encoding="utf-8") as f_de:
        for row in dataset:
            f_de.write(row["source"].replace("\n", " ").strip() + "\n")
    rm_path = out_dir / f"wmttest2024.src.{variety.replace('-', '_')}-de.xml.no-testsuites.{variety}"
    with open(rm_path, "w", encoding="utf-8") as f_rm:
        for row in dataset:
            f_rm.write(row["target"].replace("\n", " ").strip() + "\n")
