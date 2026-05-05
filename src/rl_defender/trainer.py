"""
Offline GRPO Trainer for the RL Defender.

Loss per turn-sample:
  L_i = pg_loss_i  +  β · KL_i

  where:
    pg_loss_i = −â_i · log π_θ(a_i | ctx_i)          (simple REINFORCE baseline)

    KL_i = KL(π_θ(·|ctx_i) ‖ π_ref(·|ctx_i))
         = Σ_{a∈{pass,block}} π_θ(a) · (log π_θ(a) − log π_ref(a))

  Both distributions are 2-class binary: {PASS, BLOCK}.
  Reference log-probs (log π_ref) are pre-computed and stored in the dataset
  by data_preparation.py — no separate reference model is loaded during training.

  Optional IS-ratio clipping (PPO-style GRPO):
    pg_loss_i = −min(r_i · â_i, clip(r_i, 1−ε, 1+ε) · â_i)
    where r_i = π_θ(a_i) / π_ref(a_i).

Batch total loss:
  L = (1/N) Σ_i L_i

Index / token convention (matches SFTTrainer & GRPODataset):
  2-token array order:  [PASS ("1"), BLOCK ("0")]
  action_id:            0 = PASS,  1 = BLOCK
"""

import os

import torch
import torch.nn.functional as F
from transformers import Trainer

_RL_DEBUG = os.environ.get("RL_DEBUG", "0") == "1"


def _log(msg: str):
    """Print only on rank-0 in distributed training."""
    if int(os.environ.get("LOCAL_RANK", -1)) in (-1, 0):
        print(msg)


class GRPODefenderTrainer(Trainer):
    """
    Custom Trainer implementing offline GRPO for the defender model.

    The model predicts a binary BLOCK / PASS decision (tokens "0" / "1")
    at the position immediately following the context prompt.  Loss is
    computed only at that single decision position per sample.
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
        """
        Args:
            block_token_id:    Vocabulary index of token "0" (BLOCK / harmful).
            pass_token_id:     Vocabulary index of token "1" (PASS  / benign).
            kl_coeff:          β — weight of the KL-divergence penalty.
            use_is_ratio:      If True, apply IS ratio clipping.
            clip_epsilon:      Lower clip ε (ratio min = 1−ε).
            clip_higher:       DAPO-style asymmetric clip (larger upper bound).
            clip_epsilon_high: Upper clip ε (ratio max = 1+ε_high, used when clip_higher=True).
        """
        super().__init__(*args, **kwargs)
        self.block_token_id = block_token_id
        self.pass_token_id = pass_token_id
        self.kl_coeff = kl_coeff
        self.use_is_ratio = use_is_ratio
        self.clip_epsilon = clip_epsilon
        self.clip_higher = clip_higher
        self.clip_epsilon_high = clip_epsilon_high
        self._step_count = 0
        # Buffers for accumulating metrics across gradient-accumulation steps.
        # Drained and averaged in log() before each WandB write.
        self._metric_buffer: list = []

    # ---------------------------------------------------------------- loss

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        Offline GRPO loss.

        Pops all custom fields from inputs before calling model(**inputs) so
        that the standard HF model forward signature is respected.
        """
        # Pop all custom tensors (they are NOT part of the model's signature)
        action_ids = inputs.pop("action_ids")  # [B] int
        advantages = inputs.pop("advantages")  # [B] float
        prompt_lengths = inputs.pop("prompt_lengths")  # [B] int
        ref_log_prob_pass = inputs.pop("ref_log_prob_pass")  # [B] float
        ref_log_prob_block = inputs.pop("ref_log_prob_block")  # [B] float
        total_rewards = inputs.pop("total_reward")  # [B] float
        is_harmful = inputs.pop("is_harmful")  # [B] int
        inputs.pop("labels", None)  # placeholder, not used

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits  # [B, seq_len, vocab_size]

        should_debug = _RL_DEBUG and self._step_count < 2

        batch_size = logits.shape[0]
        if should_debug:
            _log(f"\n[GRPO_DEBUG] === step #{self._step_count} ===")
            _log(
                f"  batch_size={batch_size}, kl_coeff={self.kl_coeff}, "
                f"use_is_ratio={self.use_is_ratio}, clip_higher={self.clip_higher}"
            )

        losses = []
        pg_losses = []
        kl_losses = []
        for i in range(batch_size):
            # ── position where the action token is predicted ──────────────
            prompt_len = prompt_lengths[i].item()
            pos_logits = logits[i, prompt_len - 1, :]  # [vocab]

            # ── 2-token binary distribution: [PASS idx=0, BLOCK idx=1] ───
            action_pair = torch.stack(
                [
                    pos_logits[self.pass_token_id],  # index 0 → PASS
                    pos_logits[self.block_token_id],  # index 1 → BLOCK
                ]
            )
            log_probs_new = F.log_softmax(action_pair, dim=0)  # [2]
            probs_new = log_probs_new.exp()  # [2]

            action_id = action_ids[i].item()  # 0=PASS, 1=BLOCK
            log_prob_new_a = log_probs_new[action_id]

            advantage = advantages[i].to(log_prob_new_a.device)

            # ── reference log-probs (pre-computed, stored in dataset) ─────
            lp_ref_pass = ref_log_prob_pass[i].to(log_probs_new.device)
            lp_ref_block = ref_log_prob_block[i].to(log_probs_new.device)
            log_probs_ref = torch.stack([lp_ref_pass, lp_ref_block])  # [2]

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

            # ── KL divergence: KL(π_θ ‖ π_ref) over {PASS, BLOCK} ────────
            # KL = Σ_a π_θ(a) · (log π_θ(a) − log π_ref(a))
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

        # ── accumulate reward / loss-component metrics for WandB ──────────
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
        """
        Override Trainer.log() to inject buffered reward/loss-component metrics
        before they are sent to WandB (or any other report_to backend).

        The buffer is drained here — all accumulated calls since the last log()
        are averaged so gradient-accumulation steps collapse into one data point.
        """
        if self._metric_buffer:
            all_keys = set().union(*self._metric_buffer)
            for key in all_keys:
                vals = [m[key] for m in self._metric_buffer if key in m]
                logs[key] = sum(vals) / len(vals)
            self._metric_buffer.clear()

        # Call parent with the unified kwarg signature (start_time added in HF ≥ 4.40)
        try:
            super().log(logs, start_time=start_time)
        except TypeError:
            super().log(logs)
