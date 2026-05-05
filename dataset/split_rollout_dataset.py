#!/usr/bin/env python3
"""
Split rollout JSONL files into a dataset folder with fixed train/test split.

Output structure (default: dataset/gpt52-gen/):
  - benign_train.jsonl
  - benign_test.jsonl
  - harmful_train.jsonl
  - harmful_test.jsonl
  - meta.json

Splitting is done at the sample level using sample_index to keep rollouts together.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_jsonl(path: Path) -> List[Dict]:
    data: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def group_by_sample(rollouts: List[Dict]) -> Dict[int, List[Dict]]:
    grouped: Dict[int, List[Dict]] = {}
    for rollout in rollouts:
        sample_idx = rollout.get("sample_index", 0)
        grouped.setdefault(sample_idx, []).append(rollout)
    return grouped


def split_samples(
    samples: Dict[int, List[Dict]],
    train_ratio: float,
    valid_ratio: float,
) -> (Dict[int, List[Dict]], Dict[int, List[Dict]], Dict[int, List[Dict]]):
    sample_indices = sorted(samples.keys())
    train_split = int(len(sample_indices) * train_ratio)
    valid_split = int(len(sample_indices) * (train_ratio + valid_ratio))

    train_indices = sample_indices[:train_split]
    valid_indices = sample_indices[train_split:valid_split]
    test_indices = sample_indices[valid_split:]

    train_samples = {idx: samples[idx] for idx in train_indices}
    valid_samples = {idx: samples[idx] for idx in valid_indices}
    test_samples = {idx: samples[idx] for idx in test_indices}

    return train_samples, valid_samples, test_samples


def write_jsonl(path: Path, samples: Dict[int, List[Dict]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for rollouts in samples.values():
            for rollout in rollouts:
                f.write(json.dumps(rollout, ensure_ascii=False) + "\n")
                count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split rollout JSONL into train/valid/test dataset folder"
    )
    parser.add_argument(
        "--benign-jsonl", type=str, required=True, help="Path to benign rollout JSONL"
    )
    parser.add_argument(
        "--harmful-jsonl", type=str, required=True, help="Path to harmful rollout JSONL"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset/gpt52-gen",
        help="Output dataset directory",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train split ratio (default: 0.7)",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio (default: 0.15)",
    )
    parser.add_argument(
        "--rollouts-per-sample",
        type=int,
        default=20,
        help="Expected rollouts per sample (for stats only)",
    )
    args = parser.parse_args()

    benign_path = Path(args.benign_jsonl)
    harmful_path = Path(args.harmful_jsonl)
    out_dir = Path(args.output_dir)

    benign_rollouts = load_jsonl(benign_path)
    harmful_rollouts = load_jsonl(harmful_path)

    benign_samples = group_by_sample(benign_rollouts)
    harmful_samples = group_by_sample(harmful_rollouts)

    benign_train, benign_valid, benign_test = split_samples(
        benign_samples, args.train_ratio, args.valid_ratio
    )
    harmful_train, harmful_valid, harmful_test = split_samples(
        harmful_samples, args.train_ratio, args.valid_ratio
    )

    benign_train_path = out_dir / "benign_train.jsonl"
    benign_valid_path = out_dir / "benign_valid.jsonl"
    benign_test_path = out_dir / "benign_test.jsonl"
    harmful_train_path = out_dir / "harmful_train.jsonl"
    harmful_valid_path = out_dir / "harmful_valid.jsonl"
    harmful_test_path = out_dir / "harmful_test.jsonl"

    benign_train_count = write_jsonl(benign_train_path, benign_train)
    benign_valid_count = write_jsonl(benign_valid_path, benign_valid)
    benign_test_count = write_jsonl(benign_test_path, benign_test)
    harmful_train_count = write_jsonl(harmful_train_path, harmful_train)
    harmful_valid_count = write_jsonl(harmful_valid_path, harmful_valid)
    harmful_test_count = write_jsonl(harmful_test_path, harmful_test)

    meta = {
        "benign_samples": len(benign_samples),
        "harmful_samples": len(harmful_samples),
        "benign_train_samples": len(benign_train),
        "benign_valid_samples": len(benign_valid),
        "benign_test_samples": len(benign_test),
        "harmful_train_samples": len(harmful_train),
        "harmful_valid_samples": len(harmful_valid),
        "harmful_test_samples": len(harmful_test),
        "benign_train_rollouts": benign_train_count,
        "benign_valid_rollouts": benign_valid_count,
        "benign_test_rollouts": benign_test_count,
        "harmful_train_rollouts": harmful_train_count,
        "harmful_valid_rollouts": harmful_valid_count,
        "harmful_test_rollouts": harmful_test_count,
        "train_ratio": args.train_ratio,
        "valid_ratio": args.valid_ratio,
        "rollouts_per_sample": args.rollouts_per_sample,
        "input_files": {
            "benign": str(benign_path),
            "harmful": str(harmful_path),
        },
    }

    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Split complete.")
    print(f"  Output dir: {out_dir}")
    print(
        f"  Benign rollouts: train={benign_train_count}, valid={benign_valid_count}, test={benign_test_count}"
    )
    print(
        f"  Harmful rollouts: train={harmful_train_count}, valid={harmful_valid_count}, test={harmful_test_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
