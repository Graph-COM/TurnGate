"""
Multi-turn RL Defender — offline GRPO training for the defender model.

Pipeline:
  1. data_preparation.py  — run a reference model on rollout conversations,
                            compute turn-level rewards & GRPO advantages, save JSONL.
  2. train.py             — offline GRPO training using the prepared JSONL.
  3. RLDefender (defender.py) — inference wrapper (same API as SFTDefender).

Note on verl: this module implements offline GRPO from scratch on top of the
Hugging Face Trainer, matching the existing sft_defender/ style.  Migration to
verl (Volcano Engine RL) is straightforward if online rollout generation is
needed in the future.
"""

from .config import RLDefenderConfig
from .reward import compute_turn_reward, compute_conversation_reward

__all__ = [
    "RLDefenderConfig",
    "compute_turn_reward",
    "compute_conversation_reward",
]
