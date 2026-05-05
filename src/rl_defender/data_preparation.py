"""
Offline GRPO data preparation for the RL Defender.

This script runs a (reference) model on pre-existing rollout conversations,
computes per-turn rewards and GRPO advantages within each rollout group, and
saves the processed dataset as a JSONL file for subsequent GRPO training.

Output format (one JSON object per line):
  {
    "group_id":      <int>    — sample_index shared by rollouts of the same prompt,
    "rollout_id":    <int>    — index within the group,
    "conv_type":     "harmful" | "benign",
    "total_reward":  <float>  — sum of turn-level rewards for this rollout,
    "advantage":     <float>  — GRPO-normalised advantage within the group,
    "turns": [
      {
        "turn_idx":       <int>,    — 0-indexed user-turn position,
        "context_prompt": <str>,    — formatted prompt fed to the defender,
        "action":         "0"|"1",  — "0"=BLOCK, "1"=PASS (model's argmax),
        "log_prob_block": <float>,  — log-prob of "0" under 2-token softmax,
        "log_prob_pass":  <float>,  — log-prob of "1" under 2-token softmax,
        "turn_reward":    <float>
      }, ...
    ]
  }

Usage:
  python -m src.rl_defender.data_preparation \\
      --harmful-rollout-path  data/harmful_rollouts.jsonl \\
      --benign-rollout-path   data/benign_rollouts.jsonl  \\
      --model                 Qwen/Qwen3-4B-Instruct-2507 \\
      --output-path           data/rl_defender/grpo_train.jsonl
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.defender import NAIVE_PROMPT_TEMPLATE_WITH_RESPONSE, _format_history_for_prompt
from src.rl_defender.config import RLDefenderConfig
from src.rl_defender.reward import (
    compute_turn_reward,
    compute_grpo_advantages,
)

# ------------------------------------------------------------------ helpers


def _load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def _group_by_sample(records: List[Dict]) -> Dict[int, List[Dict]]:
    """Group rollout records by their sample_index."""
    groups: Dict[int, List[Dict]] = {}
    for rec in records:
        idx = rec.get("sample_index", 0)
        groups.setdefault(idx, []).append(rec)
    return groups


def _extract_turn_pairs(
    conversation: List[Dict],
) -> List[Tuple[List[Dict], str, str]]:
    """
    Walk a conversation and produce one (history, user_content, assistant_content)
    triple per completed user→assistant exchange.

    The defender operates on the OUTPUT side: it sees the full exchange including
    the assistant's response before deciding to BLOCK or PASS.

    history contains all turns *before* the current exchange (both user and assistant).
    """
    triples = []
    history_so_far: List[Dict] = []
    pending_user: Optional[Dict] = None

    for entry in conversation:
        role = entry.get("role", "")
        content = entry.get("content", "")

        if role == "user":
            pending_user = entry
        elif role == "assistant" and pending_user is not None:
            # Full exchange (user_t, assistant_t) is now available — record it.
            triples.append((list(history_so_far), pending_user["content"], content))
            history_so_far.append(
                {
                    "role": "user",
                    "content": pending_user["content"],
                    "is_harmful": False,
                    "original_info": pending_user,
                }
            )
            history_so_far.append(
                {
                    "role": "assistant",
                    "content": content,
                    "is_harmful": False,
                    "original_info": entry,
                }
            )
            pending_user = None

    return triples


def _format_defender_prompt(
    history: List[Dict], current_prompt: str, assistant_response: str
) -> str:
    """Format context into the same prompt template used at inference (output-side)."""
    history_str = _format_history_for_prompt(history)
    return NAIVE_PROMPT_TEMPLATE_WITH_RESPONSE.format(
        history=history_str,
        question=current_prompt,
        assistant_response=assistant_response,
    )


# ------------------------------------------------- model loading & inference


def _load_model_and_tokenizer(
    model_path: str,
    lora_path: Optional[str],
    hf_token: Optional[str],
    device: str,
):
    """
    Load a (possibly LoRA-adapted) causal LM for inference.

    Args:
        model_path: HuggingFace hub name or local path.
        lora_path:  Optional path to LoRA adapter directory.
        hf_token:   HuggingFace token for private repos.
        device:     PyTorch device string, e.g. "cuda" or "cpu".
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading tokenizer from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, token=hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # left-padding for batch inference

    print(f"Loading model from {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        token=hf_token,
    )

    if lora_path:
        from peft import PeftModel

        print(f"Loading LoRA adapter from {lora_path} ...")
        model = PeftModel.from_pretrained(model, lora_path)

    model.eval()
    model.to(device)
    return tokenizer, model


@torch.no_grad()
def _get_action_log_probs_batch(
    model,
    tokenizer,
    prompts: List[str],
    block_token_id: int,
    pass_token_id: int,
    max_length: int,
    device: str,
) -> List[Tuple[float, float]]:
    """
    Run a batch of prompts through the model and return per-prompt
    (log_prob_pass, log_prob_block) pairs using 2-token binary softmax.

    Ordering follows the SFT convention:
      2-token array index 0 → PASS ("1"), index 1 → BLOCK ("0").
    """
    # Apply chat template to each prompt
    formatted = []
    for p in prompts:
        try:
            msgs = [{"role": "user", "content": p}]
            try:
                fp = tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                fp = tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        except Exception:
            fp = p
        formatted.append(fp)

    encoding = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
        add_special_tokens=False,
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # [B, seq_len, vocab]

    results = []
    for i in range(len(prompts)):
        # Last non-padded position predicts the first completion token.
        seq_len = attention_mask[i].sum().item()
        last_pos = seq_len - 1  # index of last real token
        pos_logits = logits[i, last_pos, :]  # [vocab]

        action_logits = torch.stack(
            [
                pos_logits[pass_token_id],  # index 0 → PASS
                pos_logits[block_token_id],  # index 1 → BLOCK
            ]
        )
        log_probs = F.log_softmax(action_logits, dim=0)
        results.append(
            (log_probs[0].item(), log_probs[1].item())
        )  # (lp_pass, lp_block)

    return results


# ------------------------------------------------------------ main function


def prepare_grpo_dataset(
    rollout_path: str,
    conv_type: str,  # "harmful" or "benign"
    model_path: str,
    output_path: str,
    config: RLDefenderConfig,
    lora_path: Optional[str] = None,
    device: str = "cuda",
    batch_size: int = 8,
    max_seq_length: int = 2048,
):
    """
    Process one rollout JSONL file (harmful or benign) into a GRPO training JSONL.

    Steps:
      1. Load and group rollouts by sample_index.
      2. For each group run the model to get per-turn log-probs & argmax actions.
      3. Compute turn-level rewards and aggregate to conversation-level reward.
      4. Normalise rewards within each group to obtain GRPO advantages.
      5. Write processed records to output_path (append mode).
    """
    assert conv_type in ("harmful", "benign"), "conv_type must be 'harmful' or 'benign'"

    tokenizer, model = _load_model_and_tokenizer(
        model_path, lora_path, config.hf_token, device
    )

    # Get token IDs for "0" (BLOCK) and "1" (PASS)
    block_tokens = tokenizer.encode("0", add_special_tokens=False)
    pass_tokens = tokenizer.encode("1", add_special_tokens=False)
    if len(block_tokens) != 1 or len(pass_tokens) != 1:
        raise ValueError(
            f"'0' tokenises to {block_tokens}, '1' tokenises to {pass_tokens}. "
            "Both must be single tokens for this model."
        )
    block_token_id = block_tokens[0]
    pass_token_id = pass_tokens[0]
    print(f"Token IDs  →  '0' (BLOCK): {block_token_id},  '1' (PASS): {pass_token_id}")

    print(f"Loading rollout data from {rollout_path} ...")
    raw_data = _load_jsonl(rollout_path)
    groups = _group_by_sample(raw_data)
    print(f"  {len(raw_data)} rollouts across {len(groups)} prompt groups.")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    written = 0

    with open(output_path, "a", encoding="utf-8") as out_f:
        for group_id, rollouts in tqdm(groups.items(), desc=f"Processing {conv_type}"):
            group_results = []

            for rollout_idx, rollout in enumerate(rollouts):
                conversation = rollout.get("conversation", [])
                turn_pairs = _extract_turn_pairs(conversation)

                if not turn_pairs:
                    continue

                # ---- target turn index ----
                # For harmful: last exchange is the one to block.
                # For benign: -1 (no target).
                target_idx = len(turn_pairs) - 1 if conv_type == "harmful" else -1

                # ---- batch inference for all exchanges in this rollout ----
                prompts = [
                    _format_defender_prompt(history, user_q, asst_r)
                    for history, user_q, asst_r in turn_pairs
                ]

                all_lp_pairs: List[Tuple[float, float]] = []
                for batch_start in range(0, len(prompts), batch_size):
                    batch_prompts = prompts[batch_start : batch_start + batch_size]
                    pairs = _get_action_log_probs_batch(
                        model,
                        tokenizer,
                        batch_prompts,
                        block_token_id,
                        pass_token_id,
                        max_seq_length,
                        device,
                    )
                    all_lp_pairs.extend(pairs)

                # ---- compute per-turn rewards ----
                turn_records = []
                for t, ((lp_pass, lp_block), prompt_str) in enumerate(
                    zip(all_lp_pairs, prompts)
                ):
                    # Determine argmax action from 2-token binary distribution
                    action = "block" if lp_block > lp_pass else "pass"
                    reward = compute_turn_reward(t, target_idx, action, config)
                    turn_records.append(
                        {
                            "turn_idx": t,
                            "context_prompt": prompt_str,
                            "action": "0" if action == "block" else "1",
                            "log_prob_block": lp_block,
                            "log_prob_pass": lp_pass,
                            "turn_reward": reward,
                        }
                    )

                total_reward = sum(tr["turn_reward"] for tr in turn_records)
                group_results.append(
                    {
                        "rollout_id": rollout_idx,
                        "total_reward": total_reward,
                        "turns": turn_records,
                    }
                )

            if not group_results:
                continue

            # ---- GRPO advantage normalisation within the group ----
            rewards = [r["total_reward"] for r in group_results]
            advantages = compute_grpo_advantages(rewards, config.advantage_epsilon)

            for res, adv in zip(group_results, advantages):
                record = {
                    "group_id": group_id,
                    "rollout_id": res["rollout_id"],
                    "conv_type": conv_type,
                    "total_reward": res["total_reward"],
                    "advantage": adv,
                    "turns": res["turns"],
                }
                out_f.write(json.dumps(record) + "\n")
                written += 1

    print(f"Wrote {written} rollout records to {output_path}")


# -------------------------------------------------------------------- CLI


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare offline GRPO training data for the RL Defender."
    )
    parser.add_argument(
        "--harmful-rollout-path",
        type=str,
        default=None,
        help="Path to harmful rollout JSONL (e.g. data/harmful_*.jsonl).",
    )
    parser.add_argument(
        "--benign-rollout-path",
        type=str,
        default=None,
        help="Path to benign rollout JSONL.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name or path to use as the reference policy.",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Optional LoRA adapter directory to load on top of --model.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output JSONL path for the processed GRPO dataset.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="PyTorch device (default: cuda).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Model forward-pass batch size during data preparation (default: 8).",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Max tokenisation length (default: 2048).",
    )
    parser.add_argument(
        "--num-rollouts-per-group",
        type=int,
        default=10,
        help="Expected rollouts per prompt group (informational, default: 10).",
    )
    parser.add_argument("--hf-token", type=str, default=None)
    # Reward weights (optional overrides)
    parser.add_argument("--accurate-block-reward", type=float, default=1.0)
    parser.add_argument("--correct-pass-reward", type=float, default=0.1)
    parser.add_argument("--early-block-base", type=float, default=-0.3)
    parser.add_argument("--early-block-gap-scale", type=float, default=0.1)
    parser.add_argument("--miss-penalty", type=float, default=-1.0)
    parser.add_argument("--false-block-penalty", type=float, default=-0.5)
    parser.add_argument("--late-block-penalty", type=float, default=-0.3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = RLDefenderConfig(
        hf_token=args.hf_token,
        num_rollouts_per_group=args.num_rollouts_per_group,
        accurate_block_reward=args.accurate_block_reward,
        correct_pass_reward=args.correct_pass_reward,
        early_block_base=args.early_block_base,
        early_block_gap_scale=args.early_block_gap_scale,
        miss_penalty=args.miss_penalty,
        false_block_penalty=args.false_block_penalty,
        late_block_penalty=args.late_block_penalty,
    )

    # Remove existing output to avoid mixing runs
    if os.path.exists(args.output_path):
        os.remove(args.output_path)
        print(f"Removed existing {args.output_path}")

    if args.harmful_rollout_path:
        prepare_grpo_dataset(
            rollout_path=args.harmful_rollout_path,
            conv_type="harmful",
            model_path=args.model,
            output_path=args.output_path,
            config=cfg,
            lora_path=args.lora_path,
            device=args.device,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
        )

    if args.benign_rollout_path:
        prepare_grpo_dataset(
            rollout_path=args.benign_rollout_path,
            conv_type="benign",
            model_path=args.model,
            output_path=args.output_path,
            config=cfg,
            lora_path=args.lora_path,
            device=args.device,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
        )

    print("Data preparation complete.")
