"""
Turn-level and conversation-level reward functions for the RL Defender.

Reward structure (per user turn):
  Harmful conversation (target_user_turn_idx = last user turn):
    turn < target  : PASS  → +r_bp  (correct, not yet harmful)
                     BLOCK → -(r_early + gap * gap_scale)  (early block)
    turn = target  : BLOCK → +r_tp  (accurate block)
                     PASS  → -r_miss (missed the harmful turn)
    turn > target  : PASS  → +r_bp  (conversation should have been blocked,
                                      but at least we're not adding noise)
                     BLOCK → -r_late (late block — too late, target was missed)

  Benign conversation (target_user_turn_idx = -1):
    any turn       : PASS  → +r_bp
                     BLOCK → -r_fp  (false positive)

Conversation reward: R = Σ_t r(t) over all user turns.
GRPO advantage within group: â_i = (R_i − μ) / (σ + ε).
"""

from typing import List
from .config import RLDefenderConfig


def compute_turn_reward(
    user_turn_idx: int,
    target_user_turn_idx: int,
    action: str,
    config: RLDefenderConfig,
) -> float:
    """
    Compute the reward for a single defender BLOCK/PASS decision.

    Args:
        user_turn_idx:        0-indexed position among *user* turns only.
        target_user_turn_idx: Index of the user turn that should be blocked.
                              -1 signals a benign conversation (no target turn).
        action:               "block" or "pass".
        config:               RLDefenderConfig carrying reward weights.

    Returns:
        float: Scalar turn-level reward.
    """
    is_harmful_conv = target_user_turn_idx >= 0

    if not is_harmful_conv:
        # Benign conversation — every block is a false positive.
        return (
            config.correct_pass_reward
            if action == "pass"
            else config.false_block_penalty
        )

    # Harmful conversation
    if user_turn_idx == target_user_turn_idx:
        # Critical turn: accurate block vs. miss.
        return (
            config.accurate_block_reward if action == "block" else config.miss_penalty
        )
    elif user_turn_idx < target_user_turn_idx:
        # Pre-target turns: pass is correct; block is too early.
        if action == "pass":
            return config.correct_pass_reward
        else:  # early block
            gap = target_user_turn_idx - user_turn_idx
            return config.early_block_base - gap * config.early_block_gap_scale
    else:
        # Post-target turns: the target was missed; late block or continue passing.
        return (
            config.late_block_penalty
            if action == "block"
            else config.correct_pass_reward
        )


def compute_conversation_reward(
    actions: List[str],
    target_user_turn_idx: int,
    config: RLDefenderConfig,
) -> float:
    """
    Aggregate turn-level rewards into a single conversation-level reward.

    Args:
        actions:              List of "block"/"pass" strings, one per user turn.
        target_user_turn_idx: -1 for benign, last user-turn index for harmful.
        config:               RLDefenderConfig.

    Returns:
        float: Sum of turn-level rewards (R = Σ_t r(t)).
    """
    total = 0.0
    for t, action in enumerate(actions):
        total += compute_turn_reward(t, target_user_turn_idx, action, config)
    return total


def compute_grpo_advantages(
    rewards: List[float],
    epsilon: float = 1e-6,
) -> List[float]:
    """
    Normalise rewards within a rollout group to produce GRPO advantages.

    â_i = (R_i − μ) / (σ + ε)

    If all rewards in the group are identical (σ = 0), advantages are all 0.

    Args:
        rewards: List of conversation-level rewards for each rollout in a group.
        epsilon: Denominator stabiliser.

    Returns:
        List of GRPO-normalised advantages (same length as rewards).
    """
    n = len(rewards)
    if n == 0:
        return []
    mu = sum(rewards) / n
    variance = sum((r - mu) ** 2 for r in rewards) / n
    sigma = variance**0.5
    return [(r - mu) / (sigma + epsilon) for r in rewards]
