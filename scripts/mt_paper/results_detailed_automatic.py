from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import dotenv
from datasets import load_dataset
from sacrebleu import BLEU

from romansh_mt_eval.benchmarking.comet_client import Comet
from romansh_mt_eval.benchmarking.constants import VARIETIES
from romansh_mt_eval.benchmarking.evaluation import RomanshWMT24Evaluation
from romansh_mt_eval.benchmarking.system_translations import load_llm_translations

WMT_VARIETY_ORDER = [
    "rm-rumgr",
    "rm-sursilv",
    "rm-sutsilv",
    "rm-surmiran",
    "rm-puter",
    "rm-vallader",
]

SYSTEMS: list[tuple[str, str]] = [
    ("gemini_25_flash", r"\mbox{Gemini 2.5 Flash}"),
    ("gemini_3_flash", r"\mbox{Gemini 3 Flash (preview)}"),
    ("gemini_3_pro", r"\mbox{Gemini 3 Pro (preview)}"),
    ("no_data_aug", r"\mbox{No data augmentation}"),
    ("forward_translation", r"\mbox{HR$\rightarrow$LR augmentation}"),
    ("back_translation", r"\mbox{LR$\rightarrow$HR augmentation}"),
    ("dict_prompting", r"\mbox{+ dictionary prompting}"),
]

BOUQUET_REPO_ID = "facebook/bouquet"
BOUQUET_ROMANSH_CONFIG = "roh_Latn"
BOUQUET_GERMAN_CONFIG = "deu_Latn"
BOUQUET_SPLIT = "test"
BOUQUET_EXPORT_BASENAME = "bouquet.test"
BOUQUET_VARIETY = "rm-rumgr"

WMT_SYSTEM_MAPPINGS: dict[str, str] = {
    "gemini_25_flash": "system_translations/mt_paper/second_half/Gemini-2.5-Flash",
    "gemini_3_flash": "system_translations/mt_paper/second_half/Gemini-3-Flash",
    "gemini_3_pro": "system_translations/mt_paper/second_half/Gemini-3-Pro",
    "no_data_aug": "system_translations/mt_paper/second_half/ctranslate2_fairseq_nllb-200-distilled-1.3B.norm.temp1.5.10e.noback.withdict_ct2",
    "forward_translation": "system_translations/mt_paper/second_half/ctranslate2_fairseq_nllb-200-distilled-1.3B.norm.temp1.5.10e.forward_override.withdict_ct2",
    "back_translation": "system_translations/mt_paper/second_half/ctranslate2_fairseq_nllb-200-distilled-1.3B.norm.temp1.5.10e.withdict_ct2",
    "dict_prompting": "system_translations/mt_paper/second_half/ctranslate2_fairseq_nllb-200-distilled-1.3B.norm.temp1.5.10e.withdict.dict_prompting_ct2",
}

BOUQUET_SYSTEM_MAPPINGS: dict[str, str | None] = {
    "gemini_25_flash": "systems_bouquet/gemini-2-5-flash",
    "gemini_3_flash": "systems_bouquet/gemini-3-flash-preview",
    "gemini_3_pro": "systems_bouquet/gemini-3-pro-preview",
    "no_data_aug": "systems_bouquet/ctranslate2_fairseq_nllb-200-distilled-1.3B.norm.temp1.5.10e.noback.withdict_ct2",
    "forward_translation": "systems_bouquet/ctranslate2_fairseq_nllb-200-distilled-1.3B.norm.temp1.5.10e.forward_override.withdict_ct2",
    "back_translation": "systems_bouquet/ctranslate2_fairseq_nllb-200-distilled-1.3B.norm.temp1.5.10e.withdict_ct2",
    "dict_prompting": "systems_bouquet/ctranslate2_fairseq_nllb-200-distilled-1.3B.norm.temp1.5.10e.withdict.dict_prompting_ct2",
}


def parse_args() -> argparse.Namespace:
    workspace_root = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(
        description="Detailed automatic WMT24++ and BOUQuET tables for the MT paper."
    )
    parser.add_argument(
        "--bouquet-jsonl",
        type=Path,
        default=None,
        help=f"Merged BOUQuET JSONL (default: {default_bouquet_jsonl_path(workspace_root)})",
    )
    parser.add_argument(
        "--refresh-bouquet-cache",
        action="store_true",
        help="Re-download and rebuild merged BOUQuET JSONL",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading facebook/bouquet from the Hub",
    )
    return parser.parse_args()


def _row_matches_sentence_test(row: dict[str, Any]) -> bool:
    return row.get("level") == "sentence_level" and row.get("split") == BOUQUET_SPLIT


def merge_bouquet_rumgr_rows(
    roh_rows: list[dict[str, Any]],
    deu_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    deu_by_uniq_id: dict[str, str] = {}
    for row in deu_rows:
        if not _row_matches_sentence_test(row):
            continue
        uniq_id = row.get("uniq_id")
        source_text = row.get("src_text")
        if uniq_id is None or source_text is None:
            continue
        deu_by_uniq_id[str(uniq_id)] = str(source_text)

    merged: list[dict[str, Any]] = []
    for row in roh_rows:
        if not _row_matches_sentence_test(row):
            continue
        uniq_id = row.get("uniq_id")
        if uniq_id is None:
            continue
        uniq_id_str = str(uniq_id)
        if uniq_id_str not in deu_by_uniq_id:
            continue
        romansh_text = row.get("src_text")
        if romansh_text is None:
            continue
        merged.append({
            "document_id": uniq_id_str,
            "source": deu_by_uniq_id[uniq_id_str],
            "target": str(romansh_text),
            "lp": "de_DE-rm-rumgr",
        })

    merged.sort(key=lambda entry: entry["document_id"])
    return merged


def _count_jsonl_lines(path: Path) -> int:
    count = 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def default_bouquet_jsonl_path(workspace_root: Path) -> Path:
    return workspace_root / "benchmarking" / "data" / "bouquet" / "de_DE-rm-rumgr.jsonl"


def materialize_bouquet_rumgr_jsonl(
    output_path: Path,
    *,
    refresh: bool = False,
    trust_remote_code: bool = False,
) -> int:
    output_path = output_path.resolve()
    if output_path.exists() and not refresh:
        return _count_jsonl_lines(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    roh_dataset = load_dataset(
        BOUQUET_REPO_ID,
        BOUQUET_ROMANSH_CONFIG,
        split=BOUQUET_SPLIT,
        trust_remote_code=trust_remote_code,
    )
    deu_dataset = load_dataset(
        BOUQUET_REPO_ID,
        BOUQUET_GERMAN_CONFIG,
        split=BOUQUET_SPLIT,
        trust_remote_code=trust_remote_code,
    )

    roh_rows = [dict(row) for row in roh_dataset]
    deu_rows = [dict(row) for row in deu_dataset]
    merged = merge_bouquet_rumgr_rows(roh_rows, deu_rows)

    with open(output_path, "w", encoding="utf-8") as handle:
        for entry in merged:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return len(merged)


def load_bouquet_reference_lists(jsonl_path: Path) -> tuple[list[str], list[str]]:
    german_references: list[str] = []
    romansh_references: list[str] = []
    with open(jsonl_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            german_references.append(RomanshWMT24Evaluation.postprocess(entry["source"]))
            romansh_references.append(RomanshWMT24Evaluation.postprocess(entry["target"]))
    return german_references, romansh_references


def bouquet_system_output_paths(system_directory: Path) -> tuple[Path, Path]:
    variety_file = BOUQUET_VARIETY.replace("-", "_")
    de_to_rm_path = (
        system_directory
        / f"{BOUQUET_EXPORT_BASENAME}.src.de-{variety_file}.xml.no-testsuites.{BOUQUET_VARIETY}"
    )
    rm_to_de_path = (
        system_directory
        / f"{BOUQUET_EXPORT_BASENAME}.src.{variety_file}-de.xml.no-testsuites.{BOUQUET_VARIETY}"
    )
    return de_to_rm_path, rm_to_de_path


def load_bouquet_hypotheses(
    system_directory: Path,
    expected_segment_count: int,
) -> tuple[list[str], list[str]]:
    de_to_rm_path, rm_to_de_path = bouquet_system_output_paths(system_directory)
    de_to_rm_text = de_to_rm_path.read_text(encoding="utf-8")
    rm_to_de_text = rm_to_de_path.read_text(encoding="utf-8")
    de_to_rm_hypotheses = [
        RomanshWMT24Evaluation.postprocess(line) for line in de_to_rm_text.splitlines()
    ]
    rm_to_de_hypotheses = [
        RomanshWMT24Evaluation.postprocess(line) for line in rm_to_de_text.splitlines()
    ]
    if len(de_to_rm_hypotheses) != expected_segment_count:
        raise ValueError(
            f"{de_to_rm_path}: expected {expected_segment_count} lines, "
            f"got {len(de_to_rm_hypotheses)}"
        )
    if len(rm_to_de_hypotheses) != expected_segment_count:
        raise ValueError(
            f"{rm_to_de_path}: expected {expected_segment_count} lines, "
            f"got {len(rm_to_de_hypotheses)}"
        )
    return de_to_rm_hypotheses, rm_to_de_hypotheses


def score_bouquet_system(
    german_references: list[str],
    romansh_references: list[str],
    de_to_rm_hypotheses: list[str],
    rm_to_de_hypotheses: list[str],
    comet: Comet,
) -> dict[str, float]:
    bleu = BLEU()
    de_to_rm_bleu = bleu.corpus_score(de_to_rm_hypotheses, [romansh_references]).score
    rm_to_de_bleu = bleu.corpus_score(rm_to_de_hypotheses, [german_references]).score
    rm_to_de_comet = 100.0 * comet.corpus_score(
        [None] * len(rm_to_de_hypotheses),
        rm_to_de_hypotheses,
        german_references,
    )
    return {
        "de_to_rm_bleu": float(de_to_rm_bleu),
        "rm_to_de_bleu": float(rm_to_de_bleu),
        "rm_to_de_comet": float(rm_to_de_comet),
    }


def load_split_document_ids(split_path: Path, half: str) -> set[str] | None:
    if half == "both":
        return None

    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    with open(split_path, encoding="utf-8") as handle:
        split_data = json.load(handle)

    if half == "first":
        return set(split_data["first_half"])
    if half == "second":
        return set(split_data["second_half"])
    raise ValueError(f"Invalid half value: {half}. Must be 'first', 'second', or 'both'.")


def evaluate_system_detailed(
    system_path: str,
    document_ids_filter: set[str] | None,
    evaluation: RomanshWMT24Evaluation,
) -> dict[str, dict[str, float | None]]:
    llm_translations = load_llm_translations(
        system_path,
        de_to_rm=True,
        rm_to_de=True,
        document_ids_filter=document_ids_filter,
    )

    results_bleu = {}
    for variety, system_translations in llm_translations.items():
        results_bleu[variety] = evaluation.evaluate(
            system_translations,
            "bleu",
            document_ids_filter=document_ids_filter,
        )

    results_comet = {}
    for variety, system_translations in llm_translations.items():
        results_comet[variety] = evaluation.evaluate(
            system_translations,
            "xcomet-xl",
            document_ids_filter=document_ids_filter,
        )

    de_to_rm_bleu_scores: dict[str, float | None] = {}
    rm_to_de_bleu_scores: dict[str, float | None] = {}
    rm_to_de_comet_scores: dict[str, float | None] = {}

    for variety in VARIETIES.keys():
        result_bleu = results_bleu.get(variety)
        result_comet = results_comet.get(variety)

        if result_bleu is not None:
            de_to_rm_bleu_scores[variety] = result_bleu.scores_de_to_rm.micro_avg
            rm_to_de_bleu_scores[variety] = result_bleu.scores_rm_to_de.micro_avg
        else:
            de_to_rm_bleu_scores[variety] = None
            rm_to_de_bleu_scores[variety] = None

        if result_comet is not None:
            rm_to_de_comet_scores[variety] = result_comet.scores_rm_to_de.macro_avg
        else:
            rm_to_de_comet_scores[variety] = None

    return {
        "de_to_rm_bleu": de_to_rm_bleu_scores,
        "rm_to_de_bleu": rm_to_de_bleu_scores,
        "rm_to_de_comet": rm_to_de_comet_scores,
    }


def evaluate_bouquet_systems(
    workspace_root: Path,
    *,
    bouquet_jsonl_path: Path,
    refresh_bouquet_cache: bool,
    trust_remote_code: bool,
) -> dict[str, dict[str, float | str]]:
    line_count = materialize_bouquet_rumgr_jsonl(
        bouquet_jsonl_path,
        refresh=refresh_bouquet_cache,
        trust_remote_code=trust_remote_code,
    )
    print(f"BOUQuET reference JSONL: {bouquet_jsonl_path} ({line_count} segments)")

    german_references, romansh_references = load_bouquet_reference_lists(bouquet_jsonl_path)
    expected_segment_count = len(german_references)

    comet = Comet()
    if comet.client is None:
        print(
            "Warning: COMET client is not configured; RM→DE COMET scores will be 0.0. "
            "BLEU does not use COMET.\n"
        )

    scores: dict[str, dict[str, float | str]] = {}
    for system_key, relative_path in BOUQUET_SYSTEM_MAPPINGS.items():
        if relative_path is None:
            scores[system_key] = {
                "de_to_rm_bleu": "tba",
                "rm_to_de_bleu": "tba",
                "rm_to_de_comet": "tba",
            }
            continue

        system_directory = workspace_root / relative_path
        de_to_rm_path, rm_to_de_path = bouquet_system_output_paths(system_directory)
        if not system_directory.is_dir() or not de_to_rm_path.is_file() or not rm_to_de_path.is_file():
            scores[system_key] = {
                "de_to_rm_bleu": "tba",
                "rm_to_de_bleu": "tba",
                "rm_to_de_comet": "tba",
            }
            continue

        print(f"Evaluating BOUQuET {system_key}...")
        de_to_rm_hypotheses, rm_to_de_hypotheses = load_bouquet_hypotheses(
            system_directory,
            expected_segment_count,
        )
        scores[system_key] = score_bouquet_system(
            german_references,
            romansh_references,
            de_to_rm_hypotheses,
            rm_to_de_hypotheses,
            comet,
        )

    return scores


def format_score(score: float | str | None) -> str:
    if isinstance(score, float):
        return f"{score:.1f}"
    if score is None:
        return "--"
    return str(score)


def format_de_to_rm_cell(bleu: float | str | None, comet: float | str | None) -> str:
    return rf"\phantom{{ / {format_score(comet)}}}{format_score(bleu)}"


def format_rm_to_de_cell(bleu: float | str | None, comet: float | str | None) -> str:
    return f"{format_score(bleu)} / {format_score(comet)}"


def build_de_to_rm_table(
    wmt_scores: dict[str, dict[str, dict[str, float | None]]],
    bouquet_scores: dict[str, dict[str, float | str]],
) -> str:
    lines = [
        "",
        r"{\footnotesize",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrrr@{}}",
        r"\toprule",
        r"\mbox{\textbf{System}} & \multicolumn{6}{c}{\textbf{WMT24++}} & \textbf{BOUQuET} \\",
        r"\cmidrule(lr){2-7} \cmidrule(lr){8-8}",
        r" & \textbf{RG} & \textbf{Sursilvan} & \textbf{Sutsilvan} & \textbf{Surmiran} & \textbf{Puter} & \textbf{Vallader} & \textbf{RG} \\",
        r"\midrule",
    ]

    for system_key, display_name in SYSTEMS[:3]:
        cells = [
            format_de_to_rm_cell(
                wmt_scores[system_key]["de_to_rm_bleu"][variety],
                wmt_scores[system_key]["rm_to_de_comet"][variety],
            )
            for variety in WMT_VARIETY_ORDER
        ]
        cells.append(
            format_de_to_rm_cell(
                bouquet_scores[system_key]["de_to_rm_bleu"],
                bouquet_scores[system_key]["rm_to_de_comet"],
            )
        )
        lines.append(display_name + " & " + " & ".join(cells) + r" \\")

    lines.extend([
        r"\midrule",
        r"\mbox{\textit{Fine-tuned NLLB}} & & & & & & & \\[0.2em]",
    ])

    for index, (system_key, display_name) in enumerate(SYSTEMS[3:]):
        cells = [
            format_de_to_rm_cell(
                wmt_scores[system_key]["de_to_rm_bleu"][variety],
                wmt_scores[system_key]["rm_to_de_comet"][variety],
            )
            for variety in WMT_VARIETY_ORDER
        ]
        cells.append(
            format_de_to_rm_cell(
                bouquet_scores[system_key]["de_to_rm_bleu"],
                bouquet_scores[system_key]["rm_to_de_comet"],
            )
        )
        suffix = r" \\[0.2em]" if index < 2 else r" \\"
        lines.append(display_name + " & " + " & ".join(cells) + suffix)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular*}",
        r"}",
    ])
    return "\n".join(lines) + "\n"


def build_rm_to_de_table(
    wmt_scores: dict[str, dict[str, dict[str, float | None]]],
    bouquet_scores: dict[str, dict[str, float | str]],
) -> str:
    lines = [
        r"{\footnotesize",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrrr@{}}",
        r"\toprule",
        r"\mbox{\textbf{System}} & \multicolumn{6}{c}{\textbf{WMT24++}} & \textbf{BOUQuET} \\",
        r"\cmidrule(lr){2-7} \cmidrule(lr){8-8}",
        r" & \textbf{RG} & \textbf{Sursilvan} & \textbf{Sutsilvan} & \textbf{Surmiran} & \textbf{Puter} & \textbf{Vallader} & \textbf{RG} \\",
        r"\midrule",
    ]

    for system_key, display_name in SYSTEMS[:3]:
        cells = [
            format_rm_to_de_cell(
                wmt_scores[system_key]["rm_to_de_bleu"][variety],
                wmt_scores[system_key]["rm_to_de_comet"][variety],
            )
            for variety in WMT_VARIETY_ORDER
        ]
        cells.append(
            format_rm_to_de_cell(
                bouquet_scores[system_key]["rm_to_de_bleu"],
                bouquet_scores[system_key]["rm_to_de_comet"],
            )
        )
        lines.append(display_name + " & " + " & ".join(cells) + r" \\")

    lines.extend([
        r"\midrule",
        r"\mbox{\textit{Fine-tuned NLLB}} & & & & & & & \\[0.2em]",
    ])

    for index, (system_key, display_name) in enumerate(SYSTEMS[3:]):
        cells = [
            format_rm_to_de_cell(
                wmt_scores[system_key]["rm_to_de_bleu"][variety],
                wmt_scores[system_key]["rm_to_de_comet"][variety],
            )
            for variety in WMT_VARIETY_ORDER
        ]
        cells.append(
            format_rm_to_de_cell(
                bouquet_scores[system_key]["rm_to_de_bleu"],
                bouquet_scores[system_key]["rm_to_de_comet"],
            )
        )
        suffix = r" \\[0.2em]" if index < 2 else r" \\"
        lines.append(display_name + " & " + " & ".join(cells) + suffix)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular*}",
        r"}",
    ])
    return "\n".join(lines) + "\n"


def write_output_file(filename: str, content: str) -> None:
    dotenv.load_dotenv()
    paper_directory = os.getenv("PAPER_DIR")
    if paper_directory is None:
        return

    output_path = Path(paper_directory) / "include" / filename
    if output_path.parent.exists():
        with output_path.open("w", encoding="utf-8") as handle:
            handle.write(content)
        print(f"\nResults written to {output_path}")
    else:
        print(f"\nWarning: Output directory does not exist: {output_path.parent}")


def main() -> None:
    args = parse_args()
    workspace_root = Path(__file__).resolve().parent.parent.parent
    split_path = workspace_root / "benchmarking" / "wmt24pp_split.json"
    document_ids_filter = load_split_document_ids(split_path, "second")

    evaluation = RomanshWMT24Evaluation()
    wmt_scores: dict[str, dict[str, dict[str, float | None]]] = {}
    for system_key, system_path in WMT_SYSTEM_MAPPINGS.items():
        print(f"Evaluating WMT24++ {system_key}...")
        wmt_scores[system_key] = evaluate_system_detailed(
            system_path,
            document_ids_filter,
            evaluation,
        )

    bouquet_jsonl_path = (
        args.bouquet_jsonl.resolve()
        if args.bouquet_jsonl is not None
        else default_bouquet_jsonl_path(workspace_root)
    )
    bouquet_scores = evaluate_bouquet_systems(
        workspace_root,
        bouquet_jsonl_path=bouquet_jsonl_path,
        refresh_bouquet_cache=args.refresh_bouquet_cache,
        trust_remote_code=args.trust_remote_code,
    )

    de_to_rm_table = build_de_to_rm_table(wmt_scores, bouquet_scores)
    rm_to_de_table = build_rm_to_de_table(wmt_scores, bouquet_scores)

    print("\nde_to_rm_bleu table:")
    print(de_to_rm_table)
    print("\nrm_to_de_bleu_comet table:")
    print(rm_to_de_table)

    write_output_file("results_detailed_automatic_de_to_rm_bleu.tex", de_to_rm_table)
    write_output_file("results_detailed_automatic_rm_to_de_bleu_comet.tex", rm_to_de_table)


if __name__ == "__main__":
    main()
