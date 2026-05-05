"""
Offline TurnGate Trainer.

Loss per turn-sample (identical structure to GRPO, but advantages are per-turn GAE):
  L_i = pg_loss_i  +  beta * KL_i

  pg_loss_i:
    Standard:     -A_i * log pi_theta(a_i | ctx_i)
    IS-ratio:     -min(r_i * A_i, clip(r_i) * A_i)
    Clip higher:  clip lower bound only (DAPO-style): clamp(r_i, 1-eps, inf)

  KL_i = KL(pi_theta(.|ctx_i) || pi_ref(.|ctx_i))
       = sum_{a in {pass,block}} pi_theta(a) * (log pi_theta(a) - log pi_ref(a))

Index / token convention (matches SFTTrainer & GRPODataset):
  2-token array order:  [PASS ("1"), BLOCK ("0")]
  action_id:            0 = PASS,  1 = BLOCK
"""

import os

import torch
import torch.nn.functional as F
from transformers import Trainer

_TurnGate_DEBUG = os.environ.get("TurnGate_DEBUG", "0") == "1"


def _log(msg: str):
    if int(os.environ.get("LOCAL_RANK", -1)) in (-1, 0):
        print(msg)


class TurnGateDefenderTrainer(Trainer):
    """
    Custom Trainer implementing offline TurnGate for the defender model.

    Structurally identical to GRPODefenderTrainer, but:
      - Advantages come from per-turn GAE (not shared trajectory advantage).
      - Supports DAPO-style clip_higher out of the box.
    """

    def __init__(
        self,
        block_token_id: int,
        pass_token_id: int,
        kl_coeff: float = 0.04,
        use_is_ratio: bool = False,
        clip_epsilon: float = 0.2,
        clip_higher: bool = False,
        clip_epsilon_high: float = 0.28,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.block_token_id = block_token_id
        self.pass_token_id = pass_token_id
        self.kl_coeff = kl_coeff
        self.use_is_ratio = use_is_ratio
        self.clip_epsilon = clip_epsilon
        self.clip_higher = clip_higher
        self.clip_epsilon_high = clip_epsilon_high
        self._step_count = 0
        self._metric_buffer: list = []

    # ---------------------------------------------------------------- loss

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        action_ids = inputs.pop("action_ids")
        advantages = inputs.pop("advantages")
        prompt_lengths = inputs.pop("prompt_lengths")
        ref_log_prob_pass = inputs.pop("ref_log_prob_pass")
        ref_log_prob_block = inputs.pop("ref_log_prob_block")
        total_rewards = inputs.pop("total_reward")
        is_harmful = inputs.pop("is_harmful")
        inputs.pop("labels", None)

        outputs = model(**inputs)
        logits = outputs.logits

        should_debug = _TurnGate_DEBUG and self._step_count < 2

        batch_size = logits.shape[0]
        if should_debug:
            _log(f"\n[TurnGate_DEBUG] === step #{self._step_count} ===")
            _log(
                f"  batch_size={batch_size}, kl_coeff={self.kl_coeff}, "
                f"use_is_ratio={self.use_is_ratio}, clip_higher={self.clip_higher}"
            )

        losses = []
        pg_losses = []
        kl_losses = []
        for i in range(batch_size):
            prompt_len = prompt_lengths[i].item()
            pos_logits = logits[i, prompt_len - 1, :]

            action_pair = torch.stack(
                [
                    pos_logits[self.pass_token_id],
                    pos_logits[self.block_token_id],
                ]
            )
            log_probs_new = F.log_softmax(action_pair, dim=0)
            probs_new = log_probs_new.exp()

            action_id = action_ids[i].item()
            log_prob_new_a = log_probs_new[action_id]

            advantage = advantages[i].to(log_prob_new_a.device)

            lp_ref_pass = ref_log_prob_pass[i].to(log_probs_new.device)
            lp_ref_block = ref_log_prob_block[i].to(log_probs_new.device)
            log_probs_ref = torch.stack([lp_ref_pass, lp_ref_block])

            # ── policy gradient loss ──────────────────────────────────────
            if self.use_is_ratio:
                log_prob_ref_a = log_probs_ref[action_id]
                is_ratio = torch.exp(log_prob_new_a - log_prob_ref_a)
                if self.clip_higher:
                    # DAPO clip higher: asymmetric — larger upper bound
                    is_clip = torch.clamp(
                        is_ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon_high
                    )
                else:
                    is_clip = torch.clamp(
                        is_ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                    )
                pg_loss = -torch.min(is_ratio * advantage, is_clip * advantage)
            else:
                pg_loss = -advantage * log_prob_new_a

            # ── KL divergence ─────────────────────────────────────────────
            kl_loss = (probs_new * (log_probs_new - log_probs_ref)).sum()

            sample_loss = pg_loss + self.kl_coeff * kl_loss
            losses.append(sample_loss)
            pg_losses.append(pg_loss.detach())
            kl_losses.append(kl_loss.detach())

            if should_debug and i < 4:
                _log(
                    f"  [{i}] action={action_id}, adv={advantage.item():.3f}, "
                    f"log_p_new={log_prob_new_a.item():.4f}, "
                    f"pg={pg_loss.item():.4f}, kl={kl_loss.item():.4f}, "
                    f"total={sample_loss.item():.4f}"
                )

        if losses:
            loss = torch.stack(losses).mean()
        else:
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

        # ── accumulate metrics for WandB ──────────────────────────────────
        if losses:
            rewards_f = total_rewards.float().cpu()
            harmful_mask = is_harmful.bool().cpu()
            benign_mask = ~harmful_mask

            step_metrics = {
                "reward/mean": rewards_f.mean().item(),
                "reward/std": rewards_f.std().item() if rewards_f.numel() > 1 else 0.0,
                "advantage/mean": advantages.float().mean().item(),
                "advantage/std": (
                    advantages.float().std().item() if advantages.numel() > 1 else 0.0
                ),
                "loss_parts/pg": torch.stack(pg_losses).mean().item(),
                "loss_parts/kl": torch.stack(kl_losses).mean().item(),
            }
            if harmful_mask.any():
                step_metrics["reward/harmful_mean"] = (
                    rewards_f[harmful_mask].mean().item()
                )
            if benign_mask.any():
                step_metrics["reward/benign_mean"] = (
                    rewards_f[benign_mask].mean().item()
                )

            self._metric_buffer.append(step_metrics)

        if should_debug:
            _log(f"  mean loss = {loss.item():.4f}")

        self._step_count += 1
        return (loss, outputs) if return_outputs else loss

    # ---------------------------------------------------------------- logging

    def log(self, logs: dict, start_time=None) -> None:
        if self._metric_buffer:
            all_keys = set().union(*self._metric_buffer)
            for key in all_keys:
                vals = [m[key] for m in self._metric_buffer if key in m]
                logs[key] = sum(vals) / len(vals)
            self._metric_buffer.clear()

        try:
            super().log(logs, start_time=start_time)
        except TypeError:
            super().log(logs)
