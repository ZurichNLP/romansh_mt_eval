from pathlib import Path

import pandas as pd
from datasets import load_dataset
from spacy.tokens.span_group import deepcopy
import jsonlines

from romansh_mt_eval.benchmarking.evaluation import SystemTranslations
from romansh_mt_eval.benchmarking.constants import VARIETIES


def load_madlad_translations_direct() -> dict[str, SystemTranslations]:
    translations_dir = Path(__file__).parent.parent / "systems" / "madlad" / "translations"
    assert translations_dir.exists()
    translations = {}
    # RM->DE
    for variety in VARIETIES:
        rm_to_de_path = translations_dir / f"madlad400-10b-mt-direct-{variety}-to-de.jsonl"
        with jsonlines.open(rm_to_de_path) as f:
            rm_to_de_translations = [line["target"] for line in f]
        translations[variety] = SystemTranslations(
            sys_name="madlad400-10b-mt_direct",
            variety=variety,
            translations_rm_to_de=rm_to_de_translations,
            translations_de_to_rm=None,  # will be filled in below
        )
    # DE->RM (only Rumantsch Grischun available, which we copy to all varieties)
    rumgr_path = translations_dir / "madlad400-10b-mt-direct-de-to-rm.jsonl"
    with jsonlines.open(rumgr_path) as f:
        rumgr_translations = [line["target"] for line in f]
    for variety in VARIETIES:
        translations[variety].translations_de_to_rm = deepcopy(rumgr_translations)
    return translations


def load_madlad_translations_pivot() -> dict[str, SystemTranslations]:
    translations_dir = Path(__file__).parent.parent / "systems" / "madlad" / "translations"
    assert translations_dir.exists()
    translations = {}
    # RM->DE
    for variety in VARIETIES:
        rm_to_de_path = translations_dir / f"madlad400-10b-mt-{variety}-pivot-en-to-de.jsonl"
        with jsonlines.open(rm_to_de_path) as f:
            rm_to_de_translations = [line["target"] for line in f]
        translations[variety] = SystemTranslations(
            sys_name="madlad400-10b-mt_pivot_en",
            variety=variety,
            translations_rm_to_de=rm_to_de_translations,
            translations_de_to_rm=None,  # will be filled in below
        )
    # DE->RM (only Rumantsch Grischun available, which we copy to all varieties)
    rumgr_path = translations_dir / "madlad400-10b-mt-pivot-en-to-rm.jsonl"
    with jsonlines.open(rumgr_path) as f:
        rumgr_translations = [line["target"] for line in f]
    for variety in VARIETIES:
            translations[variety].translations_de_to_rm = deepcopy(rumgr_translations)
    return translations


def load_translaturia_translations() -> dict[str, SystemTranslations]:
    filepath = Path(__file__).parent.parent / "systems" / "translaturia" / "translations" / "de_DE-rm-rumgr.jsonl"
    assert filepath.exists()
    dataset = load_dataset("json", data_files={"test": str(filepath)})["test"]
    translations = {
        "rm-rumgr": SystemTranslations(
            sys_name="translaturia",
            variety="rm-rumgr",
            translations_rm_to_de=[''] * len(dataset),  # Does not support RM->DE
            translations_de_to_rm=dataset["target"],
        )
    }
    # Copy RG translations to other varieties, will be reported in gray in the results table
    for variety in VARIETIES:
        translations[variety] = deepcopy(translations["rm-rumgr"])
        translations[variety].variety = variety
    return translations


def _extract_non_empty_cells(excel_path):
    """
    Extracts all non-empty cells from a given excel file and concatenates them
    into a list.

    Args:
      excel_path: The path to the excel file.

    Returns:
      A list of non-empty strings from the excel file.
    """
    with pd.ExcelFile(excel_path) as xls:
        all_non_empty_cells = []
        for sheet_name in xls.sheet_names:
            df = xls.parse(sheet_name)
            # Stack the dataframe to get a single series of all cells
            stacked_series = df.stack()
            # Convert the series to a list and filter out empty strings
            non_empty_cells = [str(cell).strip() for cell in stacked_series.tolist() if str(cell).strip()]
            all_non_empty_cells.extend(non_empty_cells)
    return all_non_empty_cells


def load_supertext_translations() -> dict[str, SystemTranslations]:
    translations_dir = Path(__file__).parent.parent / "systems" / "supertext" / "outputs"
    assert translations_dir.exists()
    translations = {}
    
    # RM->DE
    for variety in VARIETIES:
        rm_to_de_path = translations_dir / f"output_{variety}-de-CH.xlsx"
        rm_to_de_translations = _extract_non_empty_cells(rm_to_de_path)
        translations[variety] = SystemTranslations(
            sys_name="supertext",
            variety=variety,
            translations_rm_to_de=rm_to_de_translations,
            translations_de_to_rm=None,  # will be filled in below
            skips_bad_sources=True,
        )
    # DE->RM (only Rumantsch Grischun available, which we copy to all varieties)
    rumgr_path = translations_dir / "output_rm.xlsx"
    rumgr_translations = _extract_non_empty_cells(rumgr_path)
    for variety in VARIETIES:
        translations[variety].translations_de_to_rm = deepcopy(rumgr_translations)
    return translations

def load_llm_translations(system_name: str) -> dict[str, SystemTranslations]:
    """
    Translations were collected via WMT codebase (./wmt-collect-translations)
    """
    translations_dir = Path(__file__).parent.parent / "systems" / system_name
    assert translations_dir.exists()
    translations = {}
    for variety in VARIETIES:
        rm_to_de_path = translations_dir / f"wmttest2024.src.{variety.replace('-', '_')}-de.xml.no-testsuites.{variety}"
        rm_to_de_translations = rm_to_de_path.read_text().splitlines()
        de_to_rm_path = translations_dir / f"wmttest2024.src.de-{variety.replace('-', '_')}.xml.no-testsuites.de"
        de_to_rm_translations = de_to_rm_path.read_text().splitlines()
        assert len(rm_to_de_translations) == len(de_to_rm_translations)
        translations[variety] = SystemTranslations(
            sys_name=system_name,
            variety=variety,
            translations_rm_to_de=rm_to_de_translations,
            translations_de_to_rm=de_to_rm_translations,
        )
    return translations