"""
MDA (Monitoring-Decomposition-Attack) data conversion utilities.

Converts MDA decomposed queries JSON to:
1. SFT training JSONL (mode=sft) — for SFT/reward-shaping training
2. Conversation pickle (mode=eval) — for evaluation during training

MDA data format (each item in JSON array):
{
    "original_task": "How to manipulate satellite GPS signals?",
    "decomposition": "['subtask1', 'subtask2', ...]",
    "harm_index": 2.0,    // 1-based; NaN for benign
    "id": "harmful_0355",
    "label": "harmful"     // or "benign"
}

Usage:
    # Convert to SFT training data
    python -m src.mda_data_utils \
        --input-dir Monitoring-Decomposition-Attack-main/data/decomposed_queries \
        --output-dir data/sft/ \
        --mode sft

    # Convert to eval conversation pickles
    python -m src.mda_data_utils \
        --input-dir Monitoring-Decomposition-Attack-main/data/decomposed_queries \
        --output-dir data/eval_convs/ \
        --mode eval
"""

import ast
import json
import math
import pickle
import random
import argparse
from pathlib import Path
from typing import List, Optional

from src.data_loader import Conversation, Turn
from src.defender import NAIVE_PROMPT_TEMPLATE_MDA


def load_mda_tasks(json_path: str) -> List[dict]:
    """Load MDA JSON and parse decomposition + harm_index.

    Args:
        json_path: Path to MDA JSON file (e.g., test_data.json).

    Returns:
        List of task dicts with added 'subtasks' (list) and 'harm_idx_0based' (int or None).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    for task in tasks:
        task["subtasks"] = ast.literal_eval(task["decomposition"])
        hi = task["harm_index"]
        # NaN check: NaN != NaN
        task["harm_idx_0based"] = (
            None
            if (hi != hi or (isinstance(hi, float) and math.isnan(hi)))
            else int(hi) - 1
        )

    return tasks


# ---------------------------------------------------------------------------
# Eval mode: MDA tasks → Conversation objects (pickle)
# ---------------------------------------------------------------------------


def mda_tasks_to_conversations(tasks: List[dict]) -> List[Conversation]:
    """Convert MDA tasks to Conversation objects for evaluation.

    Each subtask becomes a user turn with an empty assistant response placeholder.
    Only the subtask at harm_index is marked is_harmful=True so that
    last_harmful_turn_index points to the correct target turn for F1 computation.
    """
    conversations = []
    for task in tasks:
        turns = []
        harm_idx = task["harm_idx_0based"]
        for i, subtask in enumerate(task["subtasks"]):
            is_harmful = harm_idx is not None and i == harm_idx
            turns.append(Turn(role="user", content=subtask, is_harmful=is_harmful))
            turns.append(Turn(role="assistant", content="", is_harmful=False))

        conv_type = "harmful" if task["label"] == "harmful" else "benign"
        conv = Conversation(
            turns=turns,
            metadata={"id": task["id"], "original_task": task["original_task"]},
            conv_type=conv_type,
        )
        conversations.append(conv)
    return conversations


# ---------------------------------------------------------------------------
# SFT mode: MDA tasks → training JSONL
# ---------------------------------------------------------------------------


def _format_history_mda(subtasks: List[str], up_to: int) -> str:
    """Format user-only history for MDA prompt."""
    if up_to <= 0:
        return "(No previous queries)"
    return "\n\n".join(f"USER: {subtasks[j]}" for j in range(up_to))


def mda_tasks_to_sft_samples(tasks: List[dict]) -> List[dict]:
    """Convert MDA tasks to SFT training samples.

    Label strategy:
    - Benign tasks: all subtasks labeled "1" (benign)
    - Harmful tasks: subtasks before harm_index labeled "1",
      harm_index and after labeled "0" (harmful)
    """
    samples = []
    for conv_id, task in enumerate(tasks):
        subtasks = task["subtasks"]
        harm_idx = task["harm_idx_0based"]
        total = len(subtasks)

        for turn_id, subtask in enumerate(subtasks):
            history_str = _format_history_mda(subtasks, turn_id)

            prompt = NAIVE_PROMPT_TEMPLATE_MDA.format(
                history=history_str,
                question=subtask,
            )

            # Label
            if harm_idx is None:
                label = "1"
            elif turn_id < harm_idx:
                label = "1"
            else:
                label = "0"

            samples.append(
                {
                    "prompt": prompt,
                    "completion": label,
                    "metadata": {
                        "conv_id": conv_id,
                        "turn_id": turn_id,
                        "total_turns": total,
                        "source_type": task["label"],
                    },
                }
            )
    return samples


def _write_jsonl(samples: List[dict], path: Path):
    """Write samples to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# MDA split name → our split name mapping
_MDA_SPLIT_MAP = {
    "train_data.json": "train",
    "val_data.json": "val",
    "test_data.json": "test",
}

# For SFT mode, MDA uses "valid" naming but file is "val_data.json"
_SFT_SPLIT_MAP = {
    "train_data.json": "train",
    "val_data.json": "valid",
    "test_data.json": "test",
}


def run_sft_mode(input_dir: str, output_dir: str, seed: int = 42):
    """Convert MDA data to SFT training JSONL files."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    random.seed(seed)

    for mda_file, split_name in _SFT_SPLIT_MAP.items():
        json_path = input_path / mda_file
        if not json_path.exists():
            print(f"  Skipping {mda_file} (not found)")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {mda_file} → {split_name}.jsonl")
        print(f"{'='*60}")

        tasks = load_mda_tasks(str(json_path))
        samples = mda_tasks_to_sft_samples(tasks)
        random.shuffle(samples)

        out_file = output_path / f"{split_name}.jsonl"
        _write_jsonl(samples, out_file)

        n_harmful = sum(1 for t in tasks if t["label"] == "harmful")
        n_benign = sum(1 for t in tasks if t["label"] == "benign")
        n_label_0 = sum(1 for s in samples if s["completion"] == "0")
        n_label_1 = sum(1 for s in samples if s["completion"] == "1")

        print(f"  Tasks: {len(tasks)} ({n_harmful} harmful, {n_benign} benign)")
        print(f"  Samples: {len(samples)} (label_0={n_label_0}, label_1={n_label_1})")
        print(f"  Saved to: {out_file}")


def run_eval_mode(input_dir: str, output_dir: str, seed: int = 42):
    """Convert MDA data to Conversation pickle files for evaluation."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    random.seed(seed)

    for mda_file, split_name in _MDA_SPLIT_MAP.items():
        json_path = input_path / mda_file
        if not json_path.exists():
            print(f"  Skipping {mda_file} (not found)")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {mda_file} → {split_name}_conversations.pkl")
        print(f"{'='*60}")

        tasks = load_mda_tasks(str(json_path))
        conversations = mda_tasks_to_conversations(tasks)

        out_file = output_path / f"{split_name}_conversations.pkl"
        with open(out_file, "wb") as f:
            pickle.dump(conversations, f)

        n_harmful = sum(1 for c in conversations if c.conv_type == "harmful")
        n_benign = sum(1 for c in conversations if c.conv_type == "benign")

        print(
            f"  Conversations: {len(conversations)} ({n_harmful} harmful, {n_benign} benign)"
        )
        print(f"  Saved to: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert MDA decomposed queries to Intent-Defender format"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="MDA data directory containing {train,val,test}_data.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for converted files",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["sft", "eval"],
        help="Conversion mode: 'sft' for training JSONL, 'eval' for conversation pickles",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.mode == "sft":
        run_sft_mode(args.input_dir, args.output_dir, seed=args.seed)
    else:
        run_eval_mode(args.input_dir, args.output_dir, seed=args.seed)

    print(f"\nDone! Output saved to {args.output_dir}/")
