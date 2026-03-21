from copy import deepcopy
from pathlib import Path

import pandas as pd
from datasets import load_dataset
import jsonlines

from romansh_mt_eval.benchmarking.evaluation import SystemTranslations
from romansh_mt_eval.benchmarking.constants import VARIETIES

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SYSTEM_TRANSLATIONS_WMT25_OLDI = _REPO_ROOT / "system_translations" / "wmt25_oldi"


def load_madlad_translations_direct() -> dict[str, SystemTranslations]:
    translations_dir = _SYSTEM_TRANSLATIONS_WMT25_OLDI / "madlad" / "translations"
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
    translations_dir = _SYSTEM_TRANSLATIONS_WMT25_OLDI / "madlad" / "translations"
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
    filepath = _SYSTEM_TRANSLATIONS_WMT25_OLDI / "translaturia" / "translations" / "de_DE-rm-rumgr.jsonl"
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
    translations_dir = _SYSTEM_TRANSLATIONS_WMT25_OLDI / "supertext" / "outputs"
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

def load_llm_translations(
    system_name: str,
    rm_to_de: bool = True,
    de_to_rm: bool = True,
    document_ids_filter: set[str] | None = None
) -> dict[str, SystemTranslations]:
    """
    Translations were collected via WMT codebase (./wmt-collect-translations)
    
    Args:
        system_name: Name of the system to load translations for
        rm_to_de: Whether to load RM->DE translations
        de_to_rm: Whether to load DE->RM translations
        document_ids_filter: Optional set of document IDs to filter by. If provided,
            only translations for documents with these IDs will be included.
    """
    # Handle paths relative to repo root (e.g. system_translations/mt_paper/second_half/...)
    if "/" in system_name or "\\" in system_name:
        translations_dir = _REPO_ROOT / system_name
    else:
        translations_dir = _SYSTEM_TRANSLATIONS_WMT25_OLDI / system_name
    assert translations_dir.exists(), f"Translations directory not found: {translations_dir}"
    translations = {}
    
    for variety in VARIETIES:
        if rm_to_de:
            rm_to_de_path = translations_dir / f"wmttest2024.src.{variety.replace('-', '_')}-de.xml.no-testsuites.{variety}"
            if not rm_to_de_path.exists():
                # Try alternative naming convention
                rm_to_de_path_alt = translations_dir / f"wmttest2024.src.{variety.replace('-', '_')}-de.xml.no-testsuites.{variety.replace('-', '_')}"
                if rm_to_de_path_alt.exists():
                    rm_to_de_path = rm_to_de_path_alt
            rm_to_de_translations = rm_to_de_path.read_text().splitlines() if rm_to_de_path.exists() else []
        else:
            rm_to_de_translations = []
        if de_to_rm:
            de_to_rm_path = translations_dir / f"wmttest2024.src.de-{variety.replace('-', '_')}.xml.no-testsuites.de"
            if not de_to_rm_path.exists():
                # Try alternative naming convention (ends with variety instead of .de)
                de_to_rm_path_alt = translations_dir / f"wmttest2024.src.de-{variety.replace('-', '_')}.xml.no-testsuites.{variety}"
                if de_to_rm_path_alt.exists():
                    de_to_rm_path = de_to_rm_path_alt
            de_to_rm_translations = de_to_rm_path.read_text().splitlines() if de_to_rm_path.exists() else []
        else:
            de_to_rm_translations = []
        
        # Filter translations if document_ids_filter is provided
        if document_ids_filter is not None:
            dataset_name = "ZurichNLP/wmt24pp-rm"
            variety_dataset = load_dataset(dataset_name, f"de_DE-{variety}", split="test")
            
            # Create a list of indices that match the filter
            filtered_indices = [
                idx for idx, example in enumerate(variety_dataset)
                if example.get("document_id") in document_ids_filter
            ]
            
            # Check if translations are already filtered (i.e., length matches filtered dataset)
            # This handles the case where translation files already contain only the filtered subset
            expected_filtered_length = len(filtered_indices)
            actual_translation_length = len(rm_to_de_translations) if rm_to_de_translations else len(de_to_rm_translations)
            
            if actual_translation_length == expected_filtered_length:
                # Translations are already filtered, use as-is
                pass
            elif actual_translation_length == len(variety_dataset):
                # Translations match full dataset, need to filter
                if rm_to_de_translations:
                    rm_to_de_translations = [rm_to_de_translations[i] for i in filtered_indices]
                if de_to_rm_translations:
                    de_to_rm_translations = [de_to_rm_translations[i] for i in filtered_indices]
            else:
                raise ValueError(
                    f"Translation length mismatch for {variety}: "
                    f"expected {expected_filtered_length} (filtered) or {len(variety_dataset)} (full), "
                    f"got {actual_translation_length}"
                )
        
        if rm_to_de_translations and de_to_rm_translations:
            assert len(rm_to_de_translations) == len(de_to_rm_translations)
        elif not rm_to_de_translations and de_to_rm_translations:
            rm_to_de_translations = [''] * len(de_to_rm_translations)
        elif rm_to_de_translations and not de_to_rm_translations:
            de_to_rm_translations = [''] * len(rm_to_de_translations)
        else:
            raise ValueError("Both directions are false or empty.")
        translations[variety] = SystemTranslations(
            sys_name=system_name,
            variety=variety,
            translations_rm_to_de=rm_to_de_translations,
            translations_de_to_rm=de_to_rm_translations,
        )
    return translations
