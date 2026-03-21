"""
Hugging Face ``datasets`` loading for pairwise human-evaluation exports
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from datasets import Dataset, load_dataset

PAIRWISE_METRIC_KEYS = frozenset(
    {"document_accuracy", "segment_accuracy", "segment_fluency"}
)
DEFAULT_PAIRWISE_SPLIT = "train"
DEFAULT_PAIRWISE_HUB_DATASET_ID = "ZurichNLP/romansh-mt-evaluation"


def add_pairwise_dataset_cli_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``--dataset`` and ``--revision`` for Hub id or local export directory."""
    group = parser.add_argument_group(
        "evaluation data",
        "Local tree from export_jsonl.py or a Hugging Face dataset id.",
    )
    group.add_argument(
        "--dataset",
        default=DEFAULT_PAIRWISE_HUB_DATASET_ID,
        metavar="PATH_OR_HUB_ID",
        help=(
            "Export root containing document_accuracy/, segment_accuracy/, "
            "segment_fluency/, or Hub id "
            f"(default: %(default)s)."
        ),
    )
    group.add_argument(
        "--revision",
        default=None,
        metavar="REV",
        help="Hub revision only (branch, tag, or commit).",
    )


def list_local_jsonl_paths_for_metric(
    data_directory: Path,
    metric_key: str,
) -> List[Path]:
    """Return sorted JSONL paths under ``<data_directory>/<metric_key>/**/*.jsonl``."""
    task_directory = data_directory / metric_key
    if not task_directory.is_dir():
        return []
    return sorted(task_directory.rglob("*.jsonl"))


def _local_export_root_if_present(value: str | Path) -> Optional[Path]:
    """If ``value`` is an existing directory, return its resolved path; else ``None``."""
    path = Path(value).expanduser()
    if path.is_dir():
        return path.resolve()
    return None


def load_metric_dataset(
    dataset_root_or_hub_id: str | Path,
    metric_key: str,
    *,
    split: str = DEFAULT_PAIRWISE_SPLIT,
    hub_revision: Optional[str] = None,
) -> Dataset:
    """
    Load one evaluation metric from a local export directory or from the Hub.

    Raises:
        ValueError: unknown ``metric_key``, or empty Hub id string.
        FileNotFoundError: local directory given but no JSONL files found.
    """
    if metric_key not in PAIRWISE_METRIC_KEYS:
        raise ValueError(
            f"metric_key must be one of {sorted(PAIRWISE_METRIC_KEYS)}, got {metric_key!r}"
        )

    local_root = _local_export_root_if_present(dataset_root_or_hub_id)
    if local_root is not None:
        paths = list_local_jsonl_paths_for_metric(local_root, metric_key)
        if not paths:
            raise FileNotFoundError(
                f"No JSONL found for metric {metric_key!r} under {local_root}"
            )
        path_strings = [str(path) for path in paths]
        return load_dataset(
            "json",
            data_files={split: path_strings},
            split=split,
        )

    hub_identifier = str(dataset_root_or_hub_id).strip()
    if not hub_identifier:
        raise ValueError("dataset_root_or_hub_id must be a non-empty Hub dataset id")

    revision_raw = (hub_revision or "").strip()
    revision: Optional[str] = revision_raw if revision_raw else None
    load_kwargs = {
        "path": hub_identifier,
        "name": metric_key,
        "split": split,
    }
    if revision is not None:
        load_kwargs["revision"] = revision
    return load_dataset(**load_kwargs)
