"""
PyTorch Dataset for offline TurnGate training.

Each sample corresponds to ONE user turn from ONE rollout.
Unlike GRPO (shared trajectory advantage), each turn has its own
GAE advantage computed during data preparation.

Expected input JSONL format (produced by data_preparation.py):
  {
    "group_id":     <int>,
    "rollout_id":   <int>,
    "conv_type":    "harmful" | "benign",
    "total_reward": <float>,
    "turns": [
      {
        "turn_idx":          <int>,
        "context_prompt":    <str>,
        "action":            "0"|"1",
        "log_prob_block":    <float>,
        "log_prob_pass":     <float>,
        "turn_reward":       <float>,
        "normalized_reward": <float>,
        "gae_advantage":     <float>
      }, ...
    ]
  }

Token / index convention (matches SFTTrainer / GRPO):
  2-token array:  index 0 -> PASS token "1",  index 1 -> BLOCK token "0"
  action_id:      0 = PASS,  1 = BLOCK
"""

import json

import torch
from torch.utils.data import Dataset

_TURNGATE_DEBUG = __import__("os").environ.get("TURNGATE_DEBUG", "0") == "1"


class TurnGateDataset(Dataset):
    """
    Flattened turn-level dataset for offline TurnGate training.

    Each item corresponds to one user turn with its own GAE advantage.
    """

    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._debug_count = 0

        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                total_reward = record["total_reward"]
                is_harmful = 1 if record.get("conv_type", "benign") == "harmful" else 0
                for turn in record["turns"]:
                    self.samples.append(
                        {
                            "context_prompt": turn["context_prompt"],
                            "action": turn["action"],
                            "log_prob_block": turn["log_prob_block"],
                            "log_prob_pass": turn["log_prob_pass"],
                            "gae_advantage": turn["gae_advantage"],
                            "total_reward": total_reward,
                            "is_harmful": is_harmful,
                        }
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        prompt_text = sample["context_prompt"]
        action_char = sample["action"]

        # Apply chat template
        try:
            messages = [{"role": "user", "content": prompt_text}]
            try:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        except Exception:
            formatted_prompt = prompt_text

        prompt_enc = self.tokenizer(
            formatted_prompt,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )
        full_text = formatted_prompt + action_char
        full_enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            add_special_tokens=False,
        )

        prompt_len = len(prompt_enc["input_ids"])
        action_id = 1 if action_char == "0" else 0

        if _TURNGATE_DEBUG and self._debug_count < 3:
            self._debug_count += 1
            print(
                f"\n[TURNGATE_DEBUG] Sample #{idx}: action={action_char!r} "
                f"(id={action_id}), gae_adv={sample['gae_advantage']:.3f}, "
                f"prompt_len={prompt_len}, "
                f"full_len={sum(full_enc['attention_mask'])}"
            )

        return {
            "input_ids": torch.tensor(full_enc["input_ids"]),
            "attention_mask": torch.tensor(full_enc["attention_mask"]),
            "labels": torch.tensor(full_enc["input_ids"]),
            "prompt_lengths": prompt_len,
            "action_ids": action_id,
            "advantages": float(sample["gae_advantage"]),
            "ref_log_prob_pass": float(sample["log_prob_pass"]),
            "ref_log_prob_block": float(sample["log_prob_block"]),
            "total_reward": float(sample["total_reward"]),
            "is_harmful": int(sample["is_harmful"]),
        }
