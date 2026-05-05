"""
Pre-generate evaluation conversations for defense evaluation during SFT training.

Generates val and test conversation sets (benign, harmful, mixed_benign, mixed_harmful)
and saves them as pickle files. These are loaded during training to avoid re-generating
mixed conversations at each evaluation step.

Usage:
    python src/sft_defender/prepare_eval_data.py \
        --dataset-dir dataset/gpt52-gen \
        --output-dir data/eval_convs \
        --mixing-strategy random \
        --min-insert-benign 0 --max-insert-benign 2 \
        --min-insert-harmful 0 --max-insert-harmful 3 \
        --seed 42
"""

import os
import sys
import pickle
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.data_loader import ConversationMixer
from src.dataset_utils import resolve_dataset_paths


def generate_eval_conversations(
    dataset_dir: str,
    split: str,
    mixing_strategy: str = "random",
    skip_mixed: bool = False,
    insert_ratio: float = None,
    min_insert_benign: int = 0,
    max_insert_benign: int = 2,
    min_insert_harmful: int = 0,
    max_insert_harmful: int = 3,
    num_benign_convs: int = None,
    num_harmful_convs: int = None,
    seed: int = 42,
):
    """Generate all conversation types for a given split."""
    import random

    random.seed(seed)

    benign_path, harmful_path = resolve_dataset_paths(dataset_dir, split, "", "")
    mixer = ConversationMixer(benign_path, harmful_path)

    benign_convs = mixer.get_benign_conversations()
    harmful_convs = mixer.get_harmful_conversations()

    if num_benign_convs is not None and num_benign_convs < len(benign_convs):
        random.shuffle(benign_convs)
        benign_convs = benign_convs[:num_benign_convs]

    if num_harmful_convs is not None and num_harmful_convs < len(harmful_convs):
        random.shuffle(harmful_convs)
        harmful_convs = harmful_convs[:num_harmful_convs]

    if skip_mixed:
        mixed_benign_convs = []
        mixed_harmful_convs = []
    else:
        if insert_ratio is not None:
            mixed_benign_convs = mixer.generate_mixed_benign(
                insert_ratio=insert_ratio,
                strategy=mixing_strategy,
            )
            mixed_harmful_convs = mixer.generate_mixed_harmful(
                insert_ratio=insert_ratio,
                strategy=mixing_strategy,
            )
        else:
            mixed_benign_convs = mixer.generate_mixed_benign(
                min_insert=min_insert_benign,
                max_insert=max_insert_benign,
                strategy=mixing_strategy,
            )
            mixed_harmful_convs = mixer.generate_mixed_harmful(
                min_insert_per_turn=min_insert_harmful,
                max_insert_per_turn=max_insert_harmful,
                strategy=mixing_strategy,
            )

    all_conversations = (
        benign_convs + harmful_convs + mixed_benign_convs + mixed_harmful_convs
    )

    stats = {
        "benign": len(benign_convs),
        "harmful": len(harmful_convs),
        "mixed_benign": len(mixed_benign_convs),
        "mixed_harmful": len(mixed_harmful_convs),
        "total": len(all_conversations),
    }

    return all_conversations, stats


def prepare_eval_data(
    dataset_dir: str,
    output_dir: str,
    splits: list = None,
    **mixing_kwargs,
):
    """
    Generate and save eval conversations for specified splits.

    Saves:
        {output_dir}/val_conversations.pkl
        {output_dir}/test_conversations.pkl
        {output_dir}/train_conversations.pkl  (sampled subset matching val size/distribution)
    """
    if splits is None:
        splits = ["valid", "test"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    split_name_map = {"valid": "val", "test": "test"}
    all_stats = {}

    for split in splits:
        print(f"\n{'='*60}")
        print(f"Generating {split} conversations...")
        print(f"{'='*60}")

        conversations, stats = generate_eval_conversations(
            dataset_dir=dataset_dir,
            split=split,
            **mixing_kwargs,
        )

        for key, value in stats.items():
            print(f"  {key}: {value}")

        out_name = split_name_map.get(split, split)
        filename = f"{out_name}_conversations.pkl"
        filepath = output_path / filename
        with open(filepath, "wb") as f:
            pickle.dump(conversations, f)

        all_stats[out_name] = stats
        print(f"  Saved to: {filepath}")

    # Generate a train subset matching val's size and distribution
    if "val" in all_stats:
        val_stats = all_stats["val"]
        n_benign = val_stats["benign"]
        n_harmful = val_stats["harmful"]

        print(f"\n{'='*60}")
        print(
            f"Generating train subset (sampled to match val: "
            f"{n_benign} benign, {n_harmful} harmful)..."
        )
        print(f"{'='*60}")

        train_convs, train_stats = generate_eval_conversations(
            dataset_dir=dataset_dir,
            split="train",
            skip_mixed=True,
            num_benign_convs=n_benign,
            num_harmful_convs=n_harmful,
            seed=mixing_kwargs.get("seed", 42),
        )

        for key, value in train_stats.items():
            print(f"  {key}: {value}")

        train_filepath = output_path / "train_conversations.pkl"
        with open(train_filepath, "wb") as f:
            pickle.dump(train_convs, f)

        print(f"  Saved to: {train_filepath}")

    print(f"\nDone! Eval conversations saved to {output_dir}/")
    return output_dir


def load_eval_conversations(eval_data_dir: str, split: str):
    """Load pre-generated conversations from pickle file.

    Args:
        eval_data_dir: Directory containing pkl files.
        split: "val" or "test".

    Returns:
        List of Conversation objects.
    """
    filepath = Path(eval_data_dir) / f"{split}_conversations.pkl"
    if not filepath.exists():
        raise FileNotFoundError(
            f"Eval conversations not found: {filepath}\n"
            f"Run: python src/sft_defender/prepare_eval_data.py "
            f"--dataset-dir <path> --output-dir {eval_data_dir}"
        )
    with open(filepath, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-generate evaluation conversations for SFT training"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Dataset directory (e.g., dataset/gpt52-gen)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for pkl files (e.g., data/eval_convs)",
    )
    parser.add_argument(
        "--mixing-strategy",
        choices=["random", "similarity"],
        default="random",
    )
    parser.add_argument(
        "--skip-mixed",
        action="store_true",
        dest="skip_mixed",
        help="Skip mixed benign/harmful datasets (DEFAULT: False)",
    )
    parser.set_defaults(skip_mixed=False)
    parser.add_argument("--insert-ratio", type=float, default=None)
    parser.add_argument("--min-insert-benign", type=int, default=0)
    parser.add_argument("--max-insert-benign", type=int, default=2)
    parser.add_argument("--min-insert-harmful", type=int, default=0)
    parser.add_argument("--max-insert-harmful", type=int, default=3)
    parser.add_argument(
        "--num-benign-convs",
        type=int,
        default=None,
        help="Limit benign conversations per split (default: all)",
    )
    parser.add_argument(
        "--num-harmful-convs",
        type=int,
        default=None,
        help="Limit harmful conversations per split (default: all)",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    prepare_eval_data(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        mixing_strategy=args.mixing_strategy,
        skip_mixed=args.skip_mixed,
        insert_ratio=args.insert_ratio,
        min_insert_benign=args.min_insert_benign,
        max_insert_benign=args.max_insert_benign,
        min_insert_harmful=args.min_insert_harmful,
        max_insert_harmful=args.max_insert_harmful,
        num_benign_convs=args.num_benign_convs,
        num_harmful_convs=args.num_harmful_convs,
        seed=args.seed,
    )
