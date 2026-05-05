"""
PyTorch Dataset for offline GRPO training of the RL Defender.

Each sample in the dataset corresponds to ONE user turn from ONE rollout.
All turns belonging to the same rollout share the same GRPO advantage.

Expected input JSONL format (produced by data_preparation.py):
  {
    "group_id":     <int>,
    "rollout_id":   <int>,
    "conv_type":    "harmful" | "benign",
    "total_reward": <float>,
    "advantage":    <float>,       ← shared by all turns of this rollout
    "turns": [
      {
        "turn_idx":       <int>,
        "context_prompt": <str>,   ← raw text prompt (before chat template)
        "action":         "0"|"1", ← "0"=BLOCK, "1"=PASS (model's argmax)
        "log_prob_block": <float>, ← log-prob of "0" under 2-token softmax
        "log_prob_pass":  <float>, ← log-prob of "1" under 2-token softmax
        "turn_reward":    <float>
      }, ...
    ]
  }

One dataset item (returned by __getitem__):
  {
    "input_ids":          tensor([...]),  ← tokenised full text (prompt + action token)
    "attention_mask":     tensor([...]),
    "labels":             tensor([...]),  ← same as input_ids (unused by GRPO trainer)
    "prompt_lengths":     <int>,          ← number of tokens in the prompt portion
    "action_ids":         <int>,          ← 0=PASS, 1=BLOCK  (2-class CE convention)
    "advantages":         <float>,        ← GRPO advantage
    "ref_log_prob_pass":  <float>,        ← log π_ref("1" | ctx)  under 2-token softmax
    "ref_log_prob_block": <float>,        ← log π_ref("0" | ctx)  under 2-token softmax
  }

Token / index convention (matches SFTTrainer):
  2-token array:  index 0 → PASS token "1",  index 1 → BLOCK token "0"
  action_id:      0 = PASS,  1 = BLOCK
  "action" field: "1" = PASS, "0" = BLOCK
"""

import json
from typing import Optional

import torch
from torch.utils.data import Dataset

_RL_DEBUG = __import__("os").environ.get("RL_DEBUG", "0") == "1"


class GRPODataset(Dataset):
    """
    Flattened turn-level dataset for offline GRPO training.

    Each item corresponds to one user turn from one rollout conversation.
    The GRPO advantage is the conversation-level (group-normalised) reward.
    """

    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 2048):
        """
        Args:
            jsonl_path: Path to prepared GRPO JSONL (output of data_preparation.py).
            tokenizer:  HuggingFace tokenizer with chat-template support.
            max_length: Maximum tokenisation length.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._debug_count = 0

        self.samples = []  # list of flat dicts, one per turn
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                advantage = record["advantage"]
                total_reward = record["total_reward"]
                is_harmful = 1 if record.get("conv_type", "benign") == "harmful" else 0
                for turn in record["turns"]:
                    self.samples.append(
                        {
                            "context_prompt": turn["context_prompt"],
                            "action": turn["action"],  # "0" or "1"
                            "log_prob_block": turn["log_prob_block"],
                            "log_prob_pass": turn["log_prob_pass"],
                            "advantage": advantage,
                            "total_reward": total_reward,
                            "is_harmful": is_harmful,
                        }
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        prompt_text = sample["context_prompt"]
        action_char = sample["action"]  # "0"=BLOCK or "1"=PASS

        # ---- apply chat template (mirrors VLLMClient._format_prompt) ----
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

        # ---- tokenise prompt and full text separately ----
        # Use add_special_tokens=False because the chat template already includes them.
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

        # ---- action_id convention (matches SFTTrainer) ----
        #   action_char "0" → BLOCK → action_id 1
        #   action_char "1" → PASS  → action_id 0
        action_id = 1 if action_char == "0" else 0

        if _RL_DEBUG and self._debug_count < 3:
            self._debug_count += 1
            _tok = lambda s: self.tokenizer.decode(
                self.tokenizer.encode(s, add_special_tokens=False)
            )
            print(
                f"\n[RL_DEBUG] Sample #{idx}: action={action_char!r} "
                f"(id={action_id}), adv={sample['advantage']:.3f}, "
                f"prompt_len={prompt_len}, "
                f"full_len={sum(full_enc['attention_mask'])}"
            )

        return {
            "input_ids": torch.tensor(full_enc["input_ids"]),
            "attention_mask": torch.tensor(full_enc["attention_mask"]),
            "labels": torch.tensor(full_enc["input_ids"]),  # placeholder
            "prompt_lengths": prompt_len,
            "action_ids": action_id,
            "advantages": float(sample["advantage"]),
            "ref_log_prob_pass": float(sample["log_prob_pass"]),
            "ref_log_prob_block": float(sample["log_prob_block"]),
            "total_reward": float(sample["total_reward"]),
            "is_harmful": int(sample["is_harmful"]),
        }
