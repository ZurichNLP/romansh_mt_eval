from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import dotenv
from datasets import load_dataset
from sacrebleu import BLEU

from romansh_mt_eval.benchmarking.comet_client import Comet
from romansh_mt_eval.benchmarking.constants import VARIETIES
from romansh_mt_eval.benchmarking.evaluation import RomanshWMT24Evaluation
from romansh_mt_eval.benchmarking.system_translations import load_llm_translations

DISPLAY_METRIC_DECIMAL_PLACES = 1

WMT_VARIETY_ORDER = [
    "rm-rumgr",
    "rm-sursilv",
    "rm-sutsilv",
    "rm-surmiran",
    "rm-puter",
    "rm-vallader",
]

WMT_AND_BOUQUET_METRIC_COLUMN_COUNT = len(WMT_VARIETY_ORDER) + 2

SYSTEMS_GEMINI: list[tuple[str, str]] = [
    ("gemini_25_flash", r"\mbox{Gemini 2.5 Flash}"),
    ("gemini_3_pro", r"\mbox{Gemini 3 Pro (preview)}"),
    ("gemini_37_flash", r"\mbox{Gemini 3.7 Flash}"),
]

FORWARD_TRANSLATION_HEADER = r"\mbox{HR$\rightarrow$LR augmentation:}"

LR_TO_HR_AUGMENTATION_HEADER = r"\mbox{LR$\rightarrow$HR augmentation:}"

FORWARD_TRANSLATION_ROWS: list[tuple[str, str]] = [
    ("forward_translation_europarl", r"\mbox{– Europarl data}"),
    ("forward_translation_newscrawl_fineweb2", r"\mbox{– Newscrawl / FineWeb2}"),
]

LR_TO_HR_AUGMENTATION_ROWS: list[tuple[str, str]] = [
    ("back_translation", r"\mbox{– Baseline prompt}"),
    ("dict_prompting", r"\mbox{– Dictionary prompting}"),
]

NO_DATA_AUG_SYSTEM: tuple[str, str] = ("no_data_aug", r"\mbox{No data augmentation}")

GEMINI_SYSTEM_KEYS: tuple[str, ...] = tuple(system_key for system_key, _ in SYSTEMS_GEMINI)
FINE_TUNED_NLLB_SYSTEM_KEYS: tuple[str, ...] = (
    (NO_DATA_AUG_SYSTEM[0],)
    + tuple(system_key for system_key, _ in FORWARD_TRANSLATION_ROWS)
    + tuple(system_key for system_key, _ in LR_TO_HR_AUGMENTATION_ROWS)
)
TABLE_SYSTEM_KEYS_IN_ORDER: tuple[str, ...] = GEMINI_SYSTEM_KEYS + FINE_TUNED_NLLB_SYSTEM_KEYS

BOUQUET_REPO_ID = "facebook/bouquet"
BOUQUET_ROMANSH_CONFIG = "roh_Latn"
BOUQUET_GERMAN_CONFIG = "deu_Latn"
BOUQUET_SPLIT = "test"
BOUQUET_EXPORT_BASENAME = "bouquet.test"
BOUQUET_VARIETY = "rm-rumgr"

WMT_SYSTEM_MAPPINGS: dict[str, str] = {
    "gemini_25_flash": "system_translations/mt_paper/second_half/Gemini-2.5-Flash",
    "gemini_3_pro": "system_translations/mt_paper/second_half/Gemini-3-Pro",
    "gemini_37_flash": "system_translations/mt_paper/second_half/Gemini-3.7-Flash",
    "no_data_aug": "system_translations/mt_paper/second_half/ctranslate2_fairseq_nllb-200-distilled-1.3B.norm.temp1.5.10e.noback.withdict_ct2",
    "forward_translation_europarl": "system_translations/mt_paper/second_half/ctranslate2_fairseq_nllb-200-distilled-1.3B.norm.temp1.5.10e.forward_override.withdict_ct2",
    "forward_translation_newscrawl_fineweb2": "system_translations/mt_paper/second_half/ctranslate2_fairseq_nllb-200-distilled-1.3B.norm.temp1.5.10e.forward_override_newscrawl_fineweb2.withdict_ct2",
    "back_translation": "system_translations/mt_paper/second_half/ctranslate2_fairseq_nllb-200-distilled-1.3B.norm.temp1.5.10e.withdict_ct2",
    "dict_prompting": "system_translations/mt_paper/second_half/ctranslate2_fairseq_nllb-200-distilled-1.3B.norm.temp1.5.10e.withdict.dict_prompting_ct2",
}

BOUQUET_SYSTEM_MAPPINGS: dict[str, str] = {
    system_key: relative_path.replace(
        "system_translations/mt_paper/second_half/",
        "system_translations/mt_paper/bouquet/",
        1,
    )
    for system_key, relative_path in WMT_SYSTEM_MAPPINGS.items()
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
        decimals = DISPLAY_METRIC_DECIMAL_PLACES
        return f"{score:.{decimals}f}"
    if score is None:
        return "--"
    return str(score)


def _numeric_metric(score: float | str | None) -> float | None:
    if isinstance(score, (int, float)):
        return float(score)
    return None


def _average_over_wmt_varieties_and_bouquet(
    wmt_scores_by_variety: dict[str, float | None],
    bouquet_score: float | str | None,
) -> float | None:
    values: list[float] = []
    for variety in WMT_VARIETY_ORDER:
        numeric = _numeric_metric(wmt_scores_by_variety.get(variety))
        if numeric is None:
            return None
        values.append(numeric)
    bouquet_numeric = _numeric_metric(bouquet_score)
    if bouquet_numeric is None:
        return None
    values.append(bouquet_numeric)
    return sum(values) / len(values)


def _metric_rounded_display_string(numeric: float) -> str:
    decimals = DISPLAY_METRIC_DECIMAL_PLACES
    return f"{numeric:.{decimals}f}"


def _winning_system_keys_for_column(
    system_keys: Iterable[str],
    getter: Callable[[str], float | None],
) -> set[str]:
    rounded_display_by_system_key: dict[str, str] = {}
    for system_key in system_keys:
        numeric = getter(system_key)
        if numeric is None:
            continue
        rounded_display_by_system_key[system_key] = _metric_rounded_display_string(numeric)
    if not rounded_display_by_system_key:
        return set()
    best_display = max(
        rounded_display_by_system_key.values(),
        key=lambda display_text: float(display_text),
    )
    return {
        system_key
        for system_key, display_text in rounded_display_by_system_key.items()
        if display_text == best_display
    }


def _wrap_metric_for_latex(metric_text: str, *, bold: bool, underline: bool) -> str:
    if bold and underline:
        return rf"\textbf{{\underline{{{metric_text}}}}}"
    if bold:
        return rf"\textbf{{{metric_text}}}"
    if underline:
        return rf"\underline{{{metric_text}}}"
    return metric_text


def _de_to_rm_bleu_for_column_index(
    system_key: str,
    column_index: int,
    wmt_scores: dict[str, dict[str, dict[str, float | None]]],
    bouquet_scores: dict[str, dict[str, float | str]],
) -> float | None:
    bouquet_column_index = len(WMT_VARIETY_ORDER)
    average_column_index = bouquet_column_index + 1
    if column_index < bouquet_column_index:
        variety = WMT_VARIETY_ORDER[column_index]
        return _numeric_metric(wmt_scores[system_key]["de_to_rm_bleu"][variety])
    if column_index == bouquet_column_index:
        return _numeric_metric(bouquet_scores[system_key]["de_to_rm_bleu"])
    if column_index == average_column_index:
        return _average_over_wmt_varieties_and_bouquet(
            wmt_scores[system_key]["de_to_rm_bleu"],
            bouquet_scores[system_key]["de_to_rm_bleu"],
        )
    raise ValueError(f"Invalid column_index for de→RM BLEU: {column_index}")


def _rm_to_de_bleu_for_column_index(
    system_key: str,
    column_index: int,
    wmt_scores: dict[str, dict[str, dict[str, float | None]]],
    bouquet_scores: dict[str, dict[str, float | str]],
) -> float | None:
    bouquet_column_index = len(WMT_VARIETY_ORDER)
    average_column_index = bouquet_column_index + 1
    if column_index < bouquet_column_index:
        variety = WMT_VARIETY_ORDER[column_index]
        return _numeric_metric(wmt_scores[system_key]["rm_to_de_bleu"][variety])
    if column_index == bouquet_column_index:
        return _numeric_metric(bouquet_scores[system_key]["rm_to_de_bleu"])
    if column_index == average_column_index:
        return _average_over_wmt_varieties_and_bouquet(
            wmt_scores[system_key]["rm_to_de_bleu"],
            bouquet_scores[system_key]["rm_to_de_bleu"],
        )
    raise ValueError(f"Invalid column_index for RM→DE BLEU: {column_index}")


def _rm_to_de_comet_for_column_index(
    system_key: str,
    column_index: int,
    wmt_scores: dict[str, dict[str, dict[str, float | None]]],
    bouquet_scores: dict[str, dict[str, float | str]],
) -> float | None:
    bouquet_column_index = len(WMT_VARIETY_ORDER)
    average_column_index = bouquet_column_index + 1
    if column_index < bouquet_column_index:
        variety = WMT_VARIETY_ORDER[column_index]
        return _numeric_metric(wmt_scores[system_key]["rm_to_de_comet"][variety])
    if column_index == bouquet_column_index:
        return _numeric_metric(bouquet_scores[system_key]["rm_to_de_comet"])
    if column_index == average_column_index:
        return _average_over_wmt_varieties_and_bouquet(
            wmt_scores[system_key]["rm_to_de_comet"],
            bouquet_scores[system_key]["rm_to_de_comet"],
        )
    raise ValueError(f"Invalid column_index for RM→DE COMET: {column_index}")


def compute_de_to_rm_column_highlights(
    wmt_scores: dict[str, dict[str, dict[str, float | None]]],
    bouquet_scores: dict[str, dict[str, float | str]],
) -> tuple[list[set[str]], list[set[str]]]:
    column_count = WMT_AND_BOUQUET_METRIC_COLUMN_COUNT
    overall_best_per_column: list[set[str]] = []
    nllb_best_per_column: list[set[str]] = []
    for column_index in range(column_count):

        def bleu_getter(
            system_key: str,
            bound_column_index: int = column_index,
        ) -> float | None:
            return _de_to_rm_bleu_for_column_index(
                system_key, bound_column_index, wmt_scores, bouquet_scores
            )

        overall_best_per_column.append(
            _winning_system_keys_for_column(TABLE_SYSTEM_KEYS_IN_ORDER, bleu_getter)
        )
        nllb_best_per_column.append(
            _winning_system_keys_for_column(FINE_TUNED_NLLB_SYSTEM_KEYS, bleu_getter)
        )
    return overall_best_per_column, nllb_best_per_column


def compute_rm_to_de_column_highlights(
    wmt_scores: dict[str, dict[str, dict[str, float | None]]],
    bouquet_scores: dict[str, dict[str, float | str]],
) -> tuple[list[set[str]], list[set[str]], list[set[str]], list[set[str]]]:
    column_count = WMT_AND_BOUQUET_METRIC_COLUMN_COUNT
    overall_best_bleu: list[set[str]] = []
    overall_best_comet: list[set[str]] = []
    nllb_best_bleu: list[set[str]] = []
    nllb_best_comet: list[set[str]] = []
    for column_index in range(column_count):

        def bleu_getter(
            system_key: str,
            bound_column_index: int = column_index,
        ) -> float | None:
            return _rm_to_de_bleu_for_column_index(
                system_key, bound_column_index, wmt_scores, bouquet_scores
            )

        def comet_getter(
            system_key: str,
            bound_column_index: int = column_index,
        ) -> float | None:
            return _rm_to_de_comet_for_column_index(
                system_key, bound_column_index, wmt_scores, bouquet_scores
            )

        overall_best_bleu.append(
            _winning_system_keys_for_column(TABLE_SYSTEM_KEYS_IN_ORDER, bleu_getter)
        )
        overall_best_comet.append(
            _winning_system_keys_for_column(TABLE_SYSTEM_KEYS_IN_ORDER, comet_getter)
        )
        nllb_best_bleu.append(
            _winning_system_keys_for_column(FINE_TUNED_NLLB_SYSTEM_KEYS, bleu_getter)
        )
        nllb_best_comet.append(
            _winning_system_keys_for_column(FINE_TUNED_NLLB_SYSTEM_KEYS, comet_getter)
        )
    return overall_best_bleu, overall_best_comet, nllb_best_bleu, nllb_best_comet


def format_de_to_rm_cell(
    bleu: float | str | None,
    *,
    bold_bleu: bool = False,
    underline_bleu: bool = False,
) -> str:
    bleu_text = format_score(bleu)
    return _wrap_metric_for_latex(
        bleu_text,
        bold=bold_bleu,
        underline=underline_bleu,
    )


def format_single_metric_cell(
    value: float | str | None,
    *,
    bold: bool = False,
    underline: bool = False,
) -> str:
    return _wrap_metric_for_latex(
        format_score(value),
        bold=bold,
        underline=underline,
    )


def de_to_rm_data_row_line(
    display_name: str,
    system_key: str,
    wmt_scores: dict[str, dict[str, dict[str, float | None]]],
    bouquet_scores: dict[str, dict[str, float | str]],
    line_suffix: str,
    *,
    overall_best_bleu_per_column: list[set[str]],
    nllb_best_bleu_per_column: list[set[str]],
) -> str:
    cells: list[str] = []
    for column_index, variety in enumerate(WMT_VARIETY_ORDER):
        cells.append(
            format_de_to_rm_cell(
                wmt_scores[system_key]["de_to_rm_bleu"][variety],
                bold_bleu=system_key in overall_best_bleu_per_column[column_index],
                underline_bleu=system_key in nllb_best_bleu_per_column[column_index],
            )
        )
    bouquet_column_index = len(WMT_VARIETY_ORDER)
    cells.append(
        format_de_to_rm_cell(
            bouquet_scores[system_key]["de_to_rm_bleu"],
            bold_bleu=system_key in overall_best_bleu_per_column[bouquet_column_index],
            underline_bleu=system_key in nllb_best_bleu_per_column[bouquet_column_index],
        )
    )
    average_column_index = bouquet_column_index + 1
    average_bleu = _average_over_wmt_varieties_and_bouquet(
        wmt_scores[system_key]["de_to_rm_bleu"],
        bouquet_scores[system_key]["de_to_rm_bleu"],
    )
    cells.append(
        format_de_to_rm_cell(
            average_bleu,
            bold_bleu=system_key in overall_best_bleu_per_column[average_column_index],
            underline_bleu=system_key in nllb_best_bleu_per_column[average_column_index],
        )
    )
    return display_name + " & " + " & ".join(cells) + line_suffix


DATA_COLUMNS_EMPTY_SPACER = r" & & & & & & & &"

TABLE_TABULAR_SPEC = (
    r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrrcr@{}}"
)
TABLE_HEADER_ROW1 = (
    r"\mbox{\textbf{System}} & \multicolumn{6}{c}{\textbf{WMT24++}} & "
    r"\textbf{BOUQuET} & \multirow{2}{*}{\textbf{Avg.}} \\"
)
TABLE_HEADER_ROW2 = (
    r" & \textbf{RG} & \textbf{Sursilvan} & \textbf{Sutsilvan} & \textbf{Surmiran} & "
    r"\textbf{Puter} & \textbf{Vallader} & \textbf{RG} \\"
)
NLLB_AFTER_SUBGROUP_SKIP = r" \\[0.45em]"
NLLB_AFTER_SECTION_TITLE_ROW_SKIP = r" \\[0.32em]"
NLLB_AFTER_FINE_TUNED_HEADER_SKIP = r" \\[0.35em]"


def rm_to_de_single_metric_data_row_line(
    display_name: str,
    system_key: str,
    wmt_scores: dict[str, dict[str, dict[str, float | None]]],
    bouquet_scores: dict[str, dict[str, float | str]],
    metric_key: str,
    line_suffix: str,
    overall_best_per_column: list[set[str]],
    nllb_best_per_column: list[set[str]],
) -> str:
    cells: list[str] = []
    for column_index, variety in enumerate(WMT_VARIETY_ORDER):
        cells.append(
            format_single_metric_cell(
                wmt_scores[system_key][metric_key][variety],
                bold=system_key in overall_best_per_column[column_index],
                underline=system_key in nllb_best_per_column[column_index],
            )
        )
    bouquet_column_index = len(WMT_VARIETY_ORDER)
    cells.append(
        format_single_metric_cell(
            bouquet_scores[system_key][metric_key],
            bold=system_key in overall_best_per_column[bouquet_column_index],
            underline=system_key in nllb_best_per_column[bouquet_column_index],
        )
    )
    average_column_index = bouquet_column_index + 1
    average_value = _average_over_wmt_varieties_and_bouquet(
        wmt_scores[system_key][metric_key],
        bouquet_scores[system_key][metric_key],
    )
    cells.append(
        format_single_metric_cell(
            average_value,
            bold=system_key in overall_best_per_column[average_column_index],
            underline=system_key in nllb_best_per_column[average_column_index],
        )
    )
    return display_name + " & " + " & ".join(cells) + line_suffix


def build_de_to_rm_table(
    wmt_scores: dict[str, dict[str, dict[str, float | None]]],
    bouquet_scores: dict[str, dict[str, float | str]],
) -> str:
    overall_best_bleu, nllb_best_bleu = compute_de_to_rm_column_highlights(
        wmt_scores, bouquet_scores
    )
    lines = [
        "",
        r"{\footnotesize",
        TABLE_TABULAR_SPEC,
        r"\toprule",
        TABLE_HEADER_ROW1,
        r"\cmidrule(lr){2-7} \cmidrule(lr){8-8} \cmidrule(lr){9-9}",
        TABLE_HEADER_ROW2,
        r"\midrule",
    ]

    for system_key, display_name in SYSTEMS_GEMINI:
        lines.append(
            de_to_rm_data_row_line(
                display_name,
                system_key,
                wmt_scores,
                bouquet_scores,
                r" \\",
                overall_best_bleu_per_column=overall_best_bleu,
                nllb_best_bleu_per_column=nllb_best_bleu,
            )
        )

    lines.extend([
        r"\midrule",
        r"\mbox{\textit{Fine-tuned NLLB}}"
        + DATA_COLUMNS_EMPTY_SPACER
        + NLLB_AFTER_FINE_TUNED_HEADER_SKIP,
    ])

    no_data_key, no_data_display = NO_DATA_AUG_SYSTEM
    lines.append(
        de_to_rm_data_row_line(
            no_data_display,
            no_data_key,
            wmt_scores,
            bouquet_scores,
            NLLB_AFTER_SUBGROUP_SKIP,
            overall_best_bleu_per_column=overall_best_bleu,
            nllb_best_bleu_per_column=nllb_best_bleu,
        )
    )

    lines.append(
        FORWARD_TRANSLATION_HEADER
        + DATA_COLUMNS_EMPTY_SPACER
        + NLLB_AFTER_SECTION_TITLE_ROW_SKIP
    )

    for forward_index, (system_key, display_name) in enumerate(FORWARD_TRANSLATION_ROWS):
        forward_suffix = (
            NLLB_AFTER_SUBGROUP_SKIP
            if forward_index == len(FORWARD_TRANSLATION_ROWS) - 1
            else r" \\"
        )
        lines.append(
            de_to_rm_data_row_line(
                display_name,
                system_key,
                wmt_scores,
                bouquet_scores,
                forward_suffix,
                overall_best_bleu_per_column=overall_best_bleu,
                nllb_best_bleu_per_column=nllb_best_bleu,
            )
        )

    lines.append(
        LR_TO_HR_AUGMENTATION_HEADER
        + DATA_COLUMNS_EMPTY_SPACER
        + NLLB_AFTER_SECTION_TITLE_ROW_SKIP
    )

    for system_key, display_name in LR_TO_HR_AUGMENTATION_ROWS:
        lines.append(
            de_to_rm_data_row_line(
                display_name,
                system_key,
                wmt_scores,
                bouquet_scores,
                r" \\",
                overall_best_bleu_per_column=overall_best_bleu,
                nllb_best_bleu_per_column=nllb_best_bleu,
            )
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular*}",
        r"}",
    ])
    return "\n".join(lines) + "\n"


def build_rm_to_de_bleu_table(
    wmt_scores: dict[str, dict[str, dict[str, float | None]]],
    bouquet_scores: dict[str, dict[str, float | str]],
) -> str:
    (
        overall_best_bleu,
        _overall_best_comet,
        nllb_best_bleu,
        _nllb_best_comet,
    ) = compute_rm_to_de_column_highlights(wmt_scores, bouquet_scores)

    lines = [
        "",
        r"{\footnotesize",
        TABLE_TABULAR_SPEC,
        r"\toprule",
        TABLE_HEADER_ROW1,
        r"\cmidrule(lr){2-7} \cmidrule(lr){8-8} \cmidrule(lr){9-9}",
        TABLE_HEADER_ROW2,
        r"\midrule",
    ]

    for system_key, display_name in SYSTEMS_GEMINI:
        lines.append(
            rm_to_de_single_metric_data_row_line(
                display_name,
                system_key,
                wmt_scores,
                bouquet_scores,
                "rm_to_de_bleu",
                r" \\",
                overall_best_bleu,
                nllb_best_bleu,
            )
        )

    lines.extend([
        r"\midrule",
        r"\mbox{\textit{Fine-tuned NLLB}}"
        + DATA_COLUMNS_EMPTY_SPACER
        + NLLB_AFTER_FINE_TUNED_HEADER_SKIP,
    ])

    no_data_key, no_data_display = NO_DATA_AUG_SYSTEM
    lines.append(
        rm_to_de_single_metric_data_row_line(
            no_data_display,
            no_data_key,
            wmt_scores,
            bouquet_scores,
            "rm_to_de_bleu",
            NLLB_AFTER_SUBGROUP_SKIP,
            overall_best_bleu,
            nllb_best_bleu,
        )
    )

    lines.append(
        FORWARD_TRANSLATION_HEADER
        + DATA_COLUMNS_EMPTY_SPACER
        + NLLB_AFTER_SECTION_TITLE_ROW_SKIP
    )

    for forward_index, (system_key, display_name) in enumerate(FORWARD_TRANSLATION_ROWS):
        forward_suffix = (
            NLLB_AFTER_SUBGROUP_SKIP
            if forward_index == len(FORWARD_TRANSLATION_ROWS) - 1
            else r" \\"
        )
        lines.append(
            rm_to_de_single_metric_data_row_line(
                display_name,
                system_key,
                wmt_scores,
                bouquet_scores,
                "rm_to_de_bleu",
                forward_suffix,
                overall_best_bleu,
                nllb_best_bleu,
            )
        )

    lines.append(
        LR_TO_HR_AUGMENTATION_HEADER
        + DATA_COLUMNS_EMPTY_SPACER
        + NLLB_AFTER_SECTION_TITLE_ROW_SKIP
    )

    for system_key, display_name in LR_TO_HR_AUGMENTATION_ROWS:
        lines.append(
            rm_to_de_single_metric_data_row_line(
                display_name,
                system_key,
                wmt_scores,
                bouquet_scores,
                "rm_to_de_bleu",
                r" \\",
                overall_best_bleu,
                nllb_best_bleu,
            )
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular*}",
        r"}",
    ])
    return "\n".join(lines) + "\n"


def build_rm_to_de_comet_table(
    wmt_scores: dict[str, dict[str, dict[str, float | None]]],
    bouquet_scores: dict[str, dict[str, float | str]],
) -> str:
    (
        _overall_best_bleu,
        overall_best_comet,
        _nllb_best_bleu,
        nllb_best_comet,
    ) = compute_rm_to_de_column_highlights(wmt_scores, bouquet_scores)

    lines = [
        "",
        r"{\footnotesize",
        TABLE_TABULAR_SPEC,
        r"\toprule",
        TABLE_HEADER_ROW1,
        r"\cmidrule(lr){2-7} \cmidrule(lr){8-8} \cmidrule(lr){9-9}",
        TABLE_HEADER_ROW2,
        r"\midrule",
    ]

    for system_key, display_name in SYSTEMS_GEMINI:
        lines.append(
            rm_to_de_single_metric_data_row_line(
                display_name,
                system_key,
                wmt_scores,
                bouquet_scores,
                "rm_to_de_comet",
                r" \\",
                overall_best_comet,
                nllb_best_comet,
            )
        )

    lines.extend([
        r"\midrule",
        r"\mbox{\textit{Fine-tuned NLLB}}"
        + DATA_COLUMNS_EMPTY_SPACER
        + NLLB_AFTER_FINE_TUNED_HEADER_SKIP,
    ])

    no_data_key, no_data_display = NO_DATA_AUG_SYSTEM
    lines.append(
        rm_to_de_single_metric_data_row_line(
            no_data_display,
            no_data_key,
            wmt_scores,
            bouquet_scores,
            "rm_to_de_comet",
            NLLB_AFTER_SUBGROUP_SKIP,
            overall_best_comet,
            nllb_best_comet,
        )
    )

    lines.append(
        FORWARD_TRANSLATION_HEADER
        + DATA_COLUMNS_EMPTY_SPACER
        + NLLB_AFTER_SECTION_TITLE_ROW_SKIP
    )

    for forward_index, (system_key, display_name) in enumerate(FORWARD_TRANSLATION_ROWS):
        forward_suffix = (
            NLLB_AFTER_SUBGROUP_SKIP
            if forward_index == len(FORWARD_TRANSLATION_ROWS) - 1
            else r" \\"
        )
        lines.append(
            rm_to_de_single_metric_data_row_line(
                display_name,
                system_key,
                wmt_scores,
                bouquet_scores,
                "rm_to_de_comet",
                forward_suffix,
                overall_best_comet,
                nllb_best_comet,
            )
        )

    lines.append(
        LR_TO_HR_AUGMENTATION_HEADER
        + DATA_COLUMNS_EMPTY_SPACER
        + NLLB_AFTER_SECTION_TITLE_ROW_SKIP
    )

    for system_key, display_name in LR_TO_HR_AUGMENTATION_ROWS:
        lines.append(
            rm_to_de_single_metric_data_row_line(
                display_name,
                system_key,
                wmt_scores,
                bouquet_scores,
                "rm_to_de_comet",
                r" \\",
                overall_best_comet,
                nllb_best_comet,
            )
        )

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
    rm_to_de_bleu_table = build_rm_to_de_bleu_table(wmt_scores, bouquet_scores)
    rm_to_de_comet_table = build_rm_to_de_comet_table(wmt_scores, bouquet_scores)

    print("\nde_to_rm_bleu table:")
    print(de_to_rm_table)
    print("\nrm_to_de_bleu table:")
    print(rm_to_de_bleu_table)
    print("\nrm_to_de_comet table:")
    print(rm_to_de_comet_table)

    write_output_file("results_detailed_automatic_de_to_rm_bleu.tex", de_to_rm_table)
    write_output_file("results_detailed_automatic_rm_to_de_bleu.tex", rm_to_de_bleu_table)
    write_output_file("results_detailed_automatic_rm_to_de_comet.tex", rm_to_de_comet_table)


if __name__ == "__main__":
    main()
