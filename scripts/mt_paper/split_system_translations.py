#!/usr/bin/env python3
"""
MT paper only: build ``first_half`` / ``second_half`` trees of system translations.

Reads each system's per-variety files from ``--systems-dir`` (default:
``system_translations/mt_paper/full/<system>/``), filters segments by document id using
``benchmarking/wmt24pp_split.json``, and writes outputs for the MT paper layout:

  system_translations/mt_paper/first_half/<system_name>/
  system_translations/mt_paper/second_half/<system_name>/
"""

import argparse
import json
from pathlib import Path
from typing import Literal

from datasets import Dataset, load_dataset

from romansh_mt_eval.benchmarking.constants import VARIETIES
from romansh_mt_eval.benchmarking.evaluation import RomanshWMT24Evaluation

TranslationDirection = Literal["rm_to_de", "de_to_rm"]


def load_split_document_ids(split_path: Path) -> tuple[set[str], set[str]]:
    """
    Load document IDs from the split JSON file.
    
    Args:
        split_path: Path to wmt24pp_split.json
        
    Returns:
        Tuple of (first_half_ids, second_half_ids)
    """
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    
    with open(split_path, 'r', encoding='utf-8') as f:
        split_data = json.load(f)
    
    return set(split_data["first_half"]), set(split_data["second_half"])


def get_document_indices_by_variety(
    first_half_ids: set[str],
    second_half_ids: set[str],
) -> tuple[dict[str, dict[str, list[int]]], dict[str, Dataset]]:
    """
    Load each variety's test split once and map document IDs to first/second-half indices.

    Returns:
        Tuple of (indices_by_variety, variety_datasets). ``variety_datasets[variety]`` is the
        ``split="test"`` dataset used for indices and for padding omitted ``is_bad_source`` rows.
    """
    dataset_name = "ZurichNLP/wmt24pp-rm"
    indices_by_variety: dict[str, dict[str, list[int]]] = {}
    variety_datasets: dict[str, Dataset] = {}

    for variety in VARIETIES:
        variety_dataset = load_dataset(dataset_name, f"de_DE-{variety}", split="test")
        variety_datasets[variety] = variety_dataset

        first_half_indices = []
        second_half_indices = []

        for idx, example in enumerate(variety_dataset):
            document_id = example.get("document_id")
            if document_id in first_half_ids:
                first_half_indices.append(idx)
            elif document_id in second_half_ids:
                second_half_indices.append(idx)

        indices_by_variety[variety] = {
            "first_half": first_half_indices,
            "second_half": second_half_indices,
        }

        print(f"  {variety}: {len(first_half_indices)} first half, {len(second_half_indices)} second half")

    return indices_by_variety, variety_datasets


def expand_translation_lines_to_full_test(
    translation_lines: list[str],
    variety_dataset: Dataset,
    direction: TranslationDirection,
    *,
    source_path: Path | None = None,
) -> list[str]:
    """
    If ``translation_lines`` omits ``is_bad_source`` rows, insert placeholders so length matches
    the full test split (index-aligned with systems that include every segment).

    RM→DE placeholders use the German ``source`` field; DE→RM placeholders use the Romansh
    ``target`` field, each passed through ``RomanshWMT24Evaluation.postprocess``.
    """
    postprocess = RomanshWMT24Evaluation.postprocess
    full_length = len(variety_dataset)
    lines_without_bad_source = full_length - sum(
        1 for row in variety_dataset if row["is_bad_source"]
    )

    path_suffix = f" ({source_path})" if source_path is not None else ""

    if len(translation_lines) == full_length:
        return translation_lines
    if len(translation_lines) != lines_without_bad_source:
        raise ValueError(
            f"Translation line count mismatch{path_suffix}: expected {full_length} "
            f"(full test set) or {lines_without_bad_source} (without is_bad_source rows), "
            f"got {len(translation_lines)}."
        )

    expanded: list[str] = []
    cursor = 0
    for row in variety_dataset:
        if row["is_bad_source"]:
            raw_placeholder = row["source"] if direction == "rm_to_de" else row["target"]
            expanded.append(postprocess(raw_placeholder))
        else:
            expanded.append(translation_lines[cursor])
            cursor += 1

    if cursor != len(translation_lines):
        raise RuntimeError(
            f"Internal padding error{path_suffix}: used {cursor} non-placeholder lines, "
            f"file had {len(translation_lines)}."
        )

    return expanded


def split_translation_file(
    input_file: Path,
    variety_dataset: Dataset,
    direction: TranslationDirection,
    first_half_indices: list[int],
    second_half_indices: list[int],
    first_half_output: Path,
    second_half_output: Path,
) -> None:
    """
    Split a translation file into first and second halves.

    Args:
        input_file: Path to input translation file
        variety_dataset: Test split for this variety (same order as line indices)
        direction: ``rm_to_de`` or ``de_to_rm`` (used when padding omitted ``is_bad_source`` rows)
        first_half_indices: List of line indices for first half
        second_half_indices: List of line indices for second half
        first_half_output: Path to write first half translations
        second_half_output: Path to write second half translations
    """
    if not input_file.exists():
        print(f"    Warning: File not found: {input_file}, skipping...")
        return

    translations = input_file.read_text(encoding="utf-8").splitlines()
    translations = expand_translation_lines_to_full_test(
        translations,
        variety_dataset,
        direction,
        source_path=input_file,
    )

    # Extract translations for each half
    first_half_translations = [translations[i] for i in first_half_indices]
    second_half_translations = [translations[i] for i in second_half_indices]
    
    # Write first half
    first_half_output.parent.mkdir(parents=True, exist_ok=True)
    with open(first_half_output, "w", encoding="utf-8") as f:
        f.write("\n".join(first_half_translations))
        if first_half_translations:  # Add newline at end if file is not empty
            f.write("\n")
    
    # Write second half
    second_half_output.parent.mkdir(parents=True, exist_ok=True)
    with open(second_half_output, "w", encoding="utf-8") as f:
        f.write("\n".join(second_half_translations))
        if second_half_translations:  # Add newline at end if file is not empty
            f.write("\n")
    
    print(
        f"    ✓ Split {len(translations)} translations (after padding): "
        f"{len(first_half_translations)} first, {len(second_half_translations)} second"
    )


def split_system_translations(
    system_name: str,
    systems_dir: Path,
    split_file: Path,
    output_base_dir: Path
) -> None:
    """
    Split all translation files for a given system.
    
    Args:
        system_name: Name of the system directory
        systems_dir: Base directory containing system directories
        split_file: Path to wmt24pp_split.json
        output_base_dir: Repository root; writes under system_translations/mt_paper/{first_half,second_half}/
    """
    system_dir = systems_dir / system_name
    if not system_dir.exists():
        raise FileNotFoundError(f"System directory not found: {system_dir}")
    
    print(f"Splitting translations for system: {system_name}")
    print(f"Source directory: {system_dir}")
    
    # Load split document IDs
    print("\nLoading split document IDs...")
    first_half_ids, second_half_ids = load_split_document_ids(split_file)
    print(f"  First half: {len(first_half_ids)} documents")
    print(f"  Second half: {len(second_half_ids)} documents")
    
    # Get indices for each variety
    print("\nMapping document IDs to dataset indices...")
    indices_by_variety, variety_datasets = get_document_indices_by_variety(
        first_half_ids, second_half_ids
    )
    
    mt_paper_root = output_base_dir / "system_translations" / "mt_paper"
    first_half_dir = mt_paper_root / "first_half" / system_name
    second_half_dir = mt_paper_root / "second_half" / system_name
    
    print(f"\nOutput directories:")
    print(f"  First half: {first_half_dir}")
    print(f"  Second half: {second_half_dir}")
    
    # Process each variety
    print("\nProcessing translation files...")
    for variety in VARIETIES:
        variety_file = variety.replace("-", "_")
        
        # RM->DE direction
        rm_to_de_filename = f"wmttest2024.src.{variety_file}-de.xml.no-testsuites.{variety}"
        rm_to_de_input = system_dir / rm_to_de_filename
        
        if rm_to_de_input.exists():
            first_half_indices = indices_by_variety[variety]["first_half"]
            second_half_indices = indices_by_variety[variety]["second_half"]
            
            print(f"  {variety} (RM->DE):")
            split_translation_file(
                rm_to_de_input,
                variety_datasets[variety],
                "rm_to_de",
                first_half_indices,
                second_half_indices,
                first_half_dir / rm_to_de_filename,
                second_half_dir / rm_to_de_filename,
            )
        
        # DE->RM direction
        de_to_rm_filename = f"wmttest2024.src.de-{variety_file}.xml.no-testsuites.de"
        de_to_rm_input = system_dir / de_to_rm_filename
        
        if de_to_rm_input.exists():
            first_half_indices = indices_by_variety[variety]["first_half"]
            second_half_indices = indices_by_variety[variety]["second_half"]
            
            print(f"  {variety} (DE->RM):")
            split_translation_file(
                de_to_rm_input,
                variety_datasets[variety],
                "de_to_rm",
                first_half_indices,
                second_half_indices,
                first_half_dir / de_to_rm_filename,
                second_half_dir / de_to_rm_filename,
            )
    
    print(f"\n✓ Successfully split translations for {system_name}")
    print(f"  First half: {first_half_dir}")
    print(f"  Second half: {second_half_dir}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "MT paper: split system translations into first/second half under "
            "system_translations/mt_paper/ (see module docstring)."
        )
    )
    parser.add_argument(
        "system_name",
        help="Name of the system directory under --systems-dir to split"
    )
    parser.add_argument(
        "--systems-dir",
        type=Path,
        default=None,
        help="Directory containing system subdirs (default: system_translations/mt_paper/full/)"
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help="Path to wmt24pp_split.json (default: benchmarking/wmt24pp_split.json under repo root)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Repository root for output (default: romansh_mt_eval/)"
    )
    
    args = parser.parse_args()
    
    # Determine paths (this file lives under scripts/mt_paper/)
    script_dir = Path(__file__).parent
    romansh_mt_eval_dir = script_dir.parent.parent

    if args.systems_dir is None:
        systems_dir = (
            romansh_mt_eval_dir / "system_translations" / "mt_paper" / "full"
        )
    else:
        systems_dir = args.systems_dir

    if args.split_file is None:
        split_file = romansh_mt_eval_dir / "benchmarking" / "wmt24pp_split.json"
    else:
        split_file = args.split_file
    
    if args.output_dir is None:
        output_dir = romansh_mt_eval_dir
    else:
        output_dir = args.output_dir
    
    # Validate paths
    if not systems_dir.exists():
        parser.error(f"Systems directory not found: {systems_dir}")
    
    if not split_file.exists():
        parser.error(f"Split file not found: {split_file}")
    
    # Split translations
    split_system_translations(
        args.system_name,
        systems_dir,
        split_file,
        output_dir
    )


if __name__ == "__main__":
    main()
