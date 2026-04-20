import json
from pathlib import Path

from datasets import load_dataset
from sacrebleu import BLEU

from romansh_mt_eval.benchmarking.constants import VARIETIES
from romansh_mt_eval.benchmarking.evaluation import RomanshWMT24Evaluation


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
SPLIT_PATH = WORKSPACE_ROOT / "benchmarking" / "wmt24pp_split.json"
SECOND_HALF_ROOT = WORKSPACE_ROOT / "system_translations" / "mt_paper" / "second_half"

SYSTEM_MAPPINGS = {
    "dict_prompting": (
        "dictionary prompting",
        "ctranslate2_fairseq_nllb-200-distilled-1.3B.norm.temp1.5.10e.withdict.dict_prompting_ct2_idiom_to_idiom",
    ),
    "dict_prompting_v2": (
        "dictionary prompting (v2)",
        "ctranslate2_fairseq_nllb-200-distilled-1.3B.norm.temp1.5.20e.withdict.dict_prompting.v2_ct2_idiom_to_idiom",
    ),
}

VARIETY_ABBREVIATIONS = {
    "rm-rumgr": "RG",
    "rm-sursilv": "Surs.",
    "rm-sutsilv": "Suts.",
    "rm-surmiran": "Surm.",
    "rm-puter": "Puter",
    "rm-vallader": "Vall.",
}


def load_second_half_document_ids() -> set[str]:
    split_data = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    return set(split_data["second_half"])


def load_references(variety: str, document_ids_filter: set[str]) -> list[str]:
    dataset = load_dataset("ZurichNLP/wmt24pp-rm", f"de_DE-{variety}", split="test")
    dataset = dataset.filter(lambda example: example.get("document_id") in document_ids_filter)
    dataset = dataset.filter(lambda example: not example["is_bad_source"])
    return [RomanshWMT24Evaluation.postprocess(example["target"]) for example in dataset]


def load_hypotheses(system_dir: Path, source_variety: str, target_variety: str) -> list[str] | None:
    source_slug = source_variety.replace("-", "_")
    target_slug = target_variety.replace("-", "_")
    hypothesis_path = (
        system_dir / f"wmttest2024.src.{source_slug}-{target_slug}.xml.no-testsuites.{source_variety}"
    )
    if not hypothesis_path.exists():
        return None

    lines = hypothesis_path.read_text(encoding="utf-8").splitlines()
    return [RomanshWMT24Evaluation.postprocess(line) for line in lines]


def main() -> None:
    document_ids_filter = load_second_half_document_ids()

    print("Loading references per variety...")
    references_per_variety: dict[str, list[str]] = {
        variety: load_references(variety, document_ids_filter) for variety in VARIETIES
    }

    variety_pairs = [
        (source_variety, target_variety)
        for source_variety in VARIETIES
        for target_variety in VARIETIES
        if source_variety != target_variety
    ]

    rows: list[list[str]] = []
    for system_key, (display_name, system_dir_name) in SYSTEM_MAPPINGS.items():
        system_dir = SECOND_HALF_ROOT / system_dir_name
        print(f"\nEvaluating {system_key} ({system_dir_name})...")

        row = [display_name]
        for source_variety, target_variety in variety_pairs:
            hypotheses = load_hypotheses(system_dir, source_variety, target_variety)
            references = references_per_variety[target_variety]
            if hypotheses is None:
                row.append("--")
                continue

            assert len(hypotheses) == len(references), (
                f"Length mismatch for {system_key} {source_variety}->{target_variety}: "
                f"{len(hypotheses)} hypotheses vs {len(references)} references."
            )
            score = BLEU().corpus_score(hypotheses, [references]).score
            row.append(f"{score:.1f}")
        rows.append(row)

    header = ["system"] + [
        f"{VARIETY_ABBREVIATIONS[source]}->{VARIETY_ABBREVIATIONS[target]}"
        for source, target in variety_pairs
    ]

    print("\n" + "=" * 80)
    print("TSV output:")
    print("=" * 80)
    print("\t".join(header))
    for row in rows:
        print("\t".join(row))


if __name__ == "__main__":
    main()
