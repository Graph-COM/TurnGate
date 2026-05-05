"""
Offline TurnGate data preparation.

Like GRPO data_preparation, but computes:
  1. Per-turn rewards (same reward function).
  2. Turn-level normalisation: collect ALL turn rewards across ALL rollouts
     in a prompt group, compute mu/sigma, normalise each turn reward.
  3. GAE backwards pass per rollout to get per-turn advantages.

Output JSONL format (one JSON object per line):
  {
    "group_id":      <int>,
    "rollout_id":    <int>,
    "conv_type":     "harmful" | "benign",
    "total_reward":  <float>,
    "turns": [
      {
        "turn_idx":        <int>,
        "context_prompt":  <str>,
        "action":          "0"|"1",
        "log_prob_block":  <float>,
        "log_prob_pass":   <float>,
        "turn_reward":     <float>,
        "normalized_reward": <float>,
        "gae_advantage":   <float>
      }, ...
    ]
  }

Usage:
  python -m src.turngate_defender.data_preparation \\
      --harmful-rollout-path  data/harmful_rollouts.jsonl \\
      --benign-rollout-path   data/benign_rollouts.jsonl  \\
      --model                 Qwen/Qwen3-4B-Instruct-2507 \\
      --output-path           data/turngate_defender/turngate_train.jsonl
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.rl_defender.data_preparation import (
    _load_jsonl,
    _group_by_sample,
    _extract_turn_pairs,
    _format_defender_prompt,
    _load_model_and_tokenizer,
    _get_action_log_probs_batch,
)
from src.rl_defender.reward import compute_turn_reward
from src.turngate_defender.config import TurnGateDefenderConfig

# ------------------------------------------------------------------ GAE


def compute_gae_advantages(
    normalized_rewards: List[float],
    gamma: float = 1.0,
    lam: float = 1.0,
) -> List[float]:
    """
    Compute GAE advantages from back to front for a single rollout.

    Without a value function, we treat normalised turn rewards as TD residuals:
        A_T     = r̂_T
        A_t     = r̂_t + gamma * lam * A_{t+1}

    Args:
        normalized_rewards: Per-turn normalised rewards for one rollout.
        gamma:  Discount factor (default 1.0, no discounting for short episodes).
        lam:    GAE lambda  (default 1.0, full Monte Carlo return).

    Returns:
        List of per-turn GAE advantages (same length as normalized_rewards).
    """
    T = len(normalized_rewards)
    if T == 0:
        return []
    advantages = [0.0] * T
    advantages[T - 1] = normalized_rewards[T - 1]
    for t in range(T - 2, -1, -1):
        advantages[t] = normalized_rewards[t] + gamma * lam * advantages[t + 1]
    return advantages


def normalize_turn_rewards(
    all_turn_rewards: List[float],
    epsilon: float = 1e-6,
) -> List[float]:
    """
    Normalise a flat list of turn rewards: r̂ = (r - mu) / (sigma + eps).

    Args:
        all_turn_rewards: Flat list of ALL turn rewards across ALL rollouts in a group.
        epsilon: Denominator stabiliser.

    Returns:
        Normalised rewards (same length and order).
    """
    n = len(all_turn_rewards)
    if n == 0:
        return []
    mu = sum(all_turn_rewards) / n
    variance = sum((r - mu) ** 2 for r in all_turn_rewards) / n
    sigma = math.sqrt(variance)
    return [(r - mu) / (sigma + epsilon) for r in all_turn_rewards]


# ------------------------------------------------------------ main function


def prepare_turngate_dataset(
    rollout_path: str,
    conv_type: str,
    model_or_engine,  # HF model or vLLM engine
    tokenizer,
    output_path: str,
    config: TurnGateDefenderConfig,
    device: str = "cuda",
    batch_size: int = 8,
    max_seq_length: int = 2048,
    use_vllm: bool = False,
):
    """
    Process one rollout JSONL file into a TurnGate training JSONL.
    """
    assert conv_type in ("harmful", "benign")

    block_tokens = tokenizer.encode("0", add_special_tokens=False)
    pass_tokens = tokenizer.encode("1", add_special_tokens=False)
    if len(block_tokens) != 1 or len(pass_tokens) != 1:
        raise ValueError(
            f"'0' tokenises to {block_tokens}, '1' tokenises to {pass_tokens}. "
            "Both must be single tokens."
        )
    block_token_id = block_tokens[0]
    pass_token_id = pass_tokens[0]

    print(f"Loading rollout data from {rollout_path} ...")
    raw_data = _load_jsonl(rollout_path)
    groups = _group_by_sample(raw_data)
    print(f"  {len(raw_data)} rollouts across {len(groups)} prompt groups.")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Phase 0: Collect ALL prompts across ALL groups and ALL rollouts for batching
    print(f"Generating prompts for {conv_type} ...")
    all_prompts = []
    # (group_id, rollout_idx_in_group, turn_idx)
    metadata = []

    # Pre-structured results
    # group_results[group_id][rollout_idx] = {rollout_id, total_reward, turns: [...]}
    group_results_map = {}

    for group_id, rollouts in groups.items():
        group_results_map[group_id] = []
        for rollout_idx, rollout in enumerate(rollouts):
            conversation = rollout.get("conversation", [])
            turn_pairs = _extract_turn_pairs(conversation)
            if not turn_pairs:
                group_results_map[group_id].append(None)
                continue

            prompts = [
                _format_defender_prompt(history, user_q, asst_r)
                for history, user_q, asst_r in turn_pairs
            ]

            for t, prompt_str in enumerate(prompts):
                all_prompts.append(prompt_str)
                metadata.append((group_id, rollout_idx, t))

            group_results_map[group_id].append(
                {
                    "rollout_id": rollout_idx,
                    "total_reward": 0.0,
                    "turns": [None] * len(prompts),
                }
            )

    # Phase 1: Batch inference for ALL prompts
    print(
        f"Running inference on {len(all_prompts)} prompts (batch_size={batch_size}, vLLM={use_vllm}) ..."
    )
    all_lp_pairs = []

    if use_vllm:
        from vllm import SamplingParams

        # vLLM's logprobs: set to a sufficient number to capture 0 and 1
        # Qwen-3B-Instruct usually outputs '0' and '1' as top-1/2 tokens
        sp = SamplingParams(max_tokens=1, logprobs=5, prompt_logprobs=0)
        outputs = model_or_engine.generate(all_prompts, sp, use_tqdm=True)

        for output in outputs:
            # Get logprobs of first generated token
            lp_dict = output.outputs[0].logprobs[0]
            # vLLM logprobs are {token_id: Logprob_obj, ...}
            # Logprob_obj has property .logprob

            # Find logprobs of 0 and 1. If not found, use a very small number.
            lp_pass = -100.0
            lp_block = -100.0
            for tid, lp_obj in lp_dict.items():
                if tid == pass_token_id:
                    lp_pass = lp_obj.logprob
                if tid == block_token_id:
                    lp_block = lp_obj.logprob

            # Re-softmax for binary decision
            t_pass = torch.tensor(lp_pass)
            t_block = torch.tensor(lp_block)
            action_logits = torch.stack([t_pass, t_block])
            lp_norm = torch.log_softmax(action_logits, dim=0)
            all_lp_pairs.append((lp_norm[0].item(), lp_norm[1].item()))
    else:
        for batch_start in tqdm(
            range(0, len(all_prompts), batch_size), desc="HF Inference"
        ):
            batch_prompts = all_prompts[batch_start : batch_start + batch_size]
            pairs = _get_action_log_probs_batch(
                model_or_engine,
                tokenizer,
                batch_prompts,
                block_token_id,
                pass_token_id,
                max_seq_length,
                device,
            )
            all_lp_pairs.extend(pairs)

    # Phase 2: Map results back and compute turn-level rewards
    print(f"Mapping results and computing rewards for {conv_type} ...")
    for i, ((lp_pass, lp_block), (group_id, rollout_idx, t)) in enumerate(
        zip(all_lp_pairs, metadata)
    ):
        res = group_results_map[group_id][rollout_idx]
        target_idx = len(res["turns"]) - 1 if conv_type == "harmful" else -1

        action = "block" if lp_block > lp_pass else "pass"
        reward = compute_turn_reward(t, target_idx, action, config)

        res["turns"][t] = {
            "turn_idx": t,
            "context_prompt": all_prompts[i],
            "action": "0" if action == "block" else "1",
            "log_prob_block": lp_block,
            "log_prob_pass": lp_pass,
            "turn_reward": reward,
        }

    # Phase 3: Group-level processing and write
    written = 0
    with open(output_path, "a", encoding="utf-8") as out_f:
        for group_id, results in group_results_map.items():
            valid_results = [r for r in results if r is not None]
            if not valid_results:
                continue

            # Turn-level normalisation across ALL turns in this group
            all_turn_rewards = []
            for res in valid_results:
                all_turn_rewards.extend([turn["turn_reward"] for turn in res["turns"]])

            normalized = normalize_turn_rewards(
                all_turn_rewards, config.advantage_epsilon
            )

            # Map normalised rewards back
            idx = 0
            for res in valid_results:
                for turn in res["turns"]:
                    turn["normalized_reward"] = normalized[idx]
                    idx += 1

            # GAE backwards and write
            for res in valid_results:
                norm_rewards = [t["normalized_reward"] for t in res["turns"]]
                gae_advs = compute_gae_advantages(
                    norm_rewards, config.gae_gamma, config.gae_lambda
                )
                for t, adv in zip(res["turns"], gae_advs):
                    t["gae_advantage"] = adv

                res["total_reward"] = sum(t["turn_reward"] for t in res["turns"])
                record = {
                    "group_id": group_id,
                    "rollout_id": res["rollout_id"],
                    "conv_type": conv_type,
                    "total_reward": res["total_reward"],
                    "turns": res["turns"],
                }
                out_f.write(json.dumps(record) + "\n")
                written += 1

    print(f"Wrote {written} rollout records to {output_path}")


# -------------------------------------------------------------------- CLI


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare offline TurnGate training data (turn-level GAE advantages)."
    )
    parser.add_argument("--harmful-rollout-path", type=str, default=None)
    parser.add_argument("--benign-rollout-path", type=str, default=None)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name or path for the reference policy.",
    )
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--hf-token", type=str, default=None)

    # GAE parameters
    parser.add_argument(
        "--gae-gamma",
        type=float,
        default=1.0,
        help="GAE discount factor (default: 1.0).",
    )
    parser.add_argument(
        "--gae-lambda", type=float, default=1.0, help="GAE lambda (default: 1.0)."
    )

    # Reward weights
    parser.add_argument("--accurate-block-reward", type=float, default=1.0)
    parser.add_argument("--correct-pass-reward", type=float, default=0.1)
    parser.add_argument("--early-block-base", type=float, default=-0.3)
    parser.add_argument("--early-block-gap-scale", type=float, default=0.1)
    parser.add_argument("--miss-penalty", type=float, default=-1.0)
    parser.add_argument("--false-block-penalty", type=float, default=-0.5)
    parser.add_argument("--late-block-penalty", type=float, default=-0.3)

    # Speedup options
    parser.add_argument(
        "--use-vllm", action="store_true", help="Use vLLM for faster inference."
    )
    parser.add_argument(
        "--tp-size", type=int, default=1, help="Tensor parallel size for vLLM."
    )
    parser.add_argument(
        "--gpu-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = TurnGateDefenderConfig(
        hf_token=args.hf_token,
        gae_gamma=args.gae_gamma,
        gae_lambda=args.gae_lambda,
        accurate_block_reward=args.accurate_block_reward,
        correct_pass_reward=args.correct_pass_reward,
        early_block_base=args.early_block_base,
        early_block_gap_scale=args.early_block_gap_scale,
        miss_penalty=args.miss_penalty,
        false_block_penalty=args.false_block_penalty,
        late_block_penalty=args.late_block_penalty,
    )

    if os.path.exists(args.output_path):
        os.remove(args.output_path)
        print(f"Removed existing {args.output_path}")

    # Load model ONCE
    if args.use_vllm:
        print(f"Loading vLLM engine from {args.model} (TP={args.tp_size}) ...")
        from vllm import LLM

        model_or_engine = LLM(
            model=args.model,
            tensor_parallel_size=args.tp_size,
            trust_remote_code=True,
            gpu_memory_utilization=args.gpu_utilization,
        )
        tokenizer = model_or_engine.get_tokenizer()
    else:
        tokenizer, model_or_engine = _load_model_and_tokenizer(
            args.model, args.lora_path, args.hf_token, args.device
        )

    if args.harmful_rollout_path:
        prepare_turngate_dataset(
            rollout_path=args.harmful_rollout_path,
            conv_type="harmful",
            model_or_engine=model_or_engine,
            tokenizer=tokenizer,
            output_path=args.output_path,
            config=cfg,
            device=args.device,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            use_vllm=args.use_vllm,
        )

    if args.benign_rollout_path:
        prepare_turngate_dataset(
            rollout_path=args.benign_rollout_path,
            conv_type="benign",
            model_or_engine=model_or_engine,
            tokenizer=tokenizer,
            output_path=args.output_path,
            config=cfg,
            device=args.device,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            use_vllm=args.use_vllm,
        )

    print("TurnGate data preparation complete.")
