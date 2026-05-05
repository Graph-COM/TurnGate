from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple


def resolve_dataset_paths(
    dataset_dir: Optional[str],
    split: str,
    benign_path: str,
    harmful_path: str,
) -> Tuple[str, str]:
    """
    Resolve benign/harmful JSONL paths from a dataset directory and split.

    Expected filenames inside dataset_dir:
      - benign_train.jsonl / harmful_train.jsonl
      - benign_valid.jsonl / harmful_valid.jsonl
      - benign_test.jsonl / harmful_test.jsonl

    Args:
        dataset_dir: Dataset directory path or None.
        split: Dataset split to use: "train", "valid", or "test".
        benign_path: Fallback benign JSONL path.
        harmful_path: Fallback harmful JSONL path.

    Returns:
        Tuple of (benign_jsonl_path, harmful_jsonl_path).
    """
    if not dataset_dir:
        return benign_path, harmful_path

    split_norm = (split or "train").strip().lower()
    if split_norm not in {"train", "valid", "test"}:
        raise ValueError("split must be 'train', 'valid', or 'test'")

    base = Path(dataset_dir)
    benign_file = base / f"benign_{split_norm}.jsonl"
    harmful_file = base / f"harmful_{split_norm}.jsonl"

    if not benign_file.exists():
        raise FileNotFoundError(f"Missing benign split file: {benign_file}")
    if not harmful_file.exists():
        raise FileNotFoundError(f"Missing harmful split file: {harmful_file}")

    return str(benign_file), str(harmful_file)
