"""
Intent-Defender integration for CKA online defense (SFT/RL only).

Uses SFT/RL trained models from Intent-Defender framework to perform
turn-level output defense during CKA tree search. When BLOCK is triggered,
the entire search branch is terminated.

Usage:
    Called from CKA-Agent after evaluation selects best child.
    BLOCK → terminate search, backpropagate penalty.
"""

import os
import sys
import logging
from typing import Dict, Any, List, Tuple

# Add Intent-Defender root to path so we can import src.*
_INTENT_DEFENDER_ROOT = os.path.join(os.path.dirname(__file__), "../..")
if _INTENT_DEFENDER_ROOT not in sys.path:
    sys.path.insert(0, _INTENT_DEFENDER_ROOT)

from src.intention_analysis import VLLMClient
from src.defender import SFTDefender, RLDefender, Action

REFUSAL_RESPONSE = (
    "I'm sorry, but I cannot provide this information "
    "due to safety policy restrictions."
)


class IntentDefenderOnline:
    """
    Online defense using Intent-Defender SFT/RL models.

    Provides check_with_history() that accepts conversation trajectory
    and returns block/pass decision. Model is lazily loaded on first call.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.defender_type = config.get("defender_type", "sft")
        self.logger = logging.getLogger(self.__class__.__name__)

        # Per-query stats (reset after each generate_jailbreak call)
        self.stats = {"total_checked": 0, "blocked": 0, "passed": 0}

        # Lazy init
        self._defender = None
        self._initialized = False

    def _initialize(self):
        """Lazy-load defender model via vLLM."""
        if self._initialized:
            return

        checkpoint = self.config["checkpoint"]
        training_type = self.config.get("training_type", "full")
        base_model = self.config.get("base_model")
        gpu_memory = self.config.get("gpu_memory_utilization", 0.8)
        hf_token = self.config.get("hf_token")

        # Pin to specific GPU if allocated by GPUResourceManager
        gpu_id = self.config.get("_gpu_id")
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            self.logger.info(f"Intent-Defender pinned to GPU {gpu_id}")

        self.logger.info(
            f"Loading Intent-Defender ({self.defender_type}) from {checkpoint}..."
        )

        if training_type == "lora":
            if not base_model:
                raise ValueError(
                    "base_model is required for lora training_type in intent_defender config"
                )
            llm_client = VLLMClient(
                model=base_model,
                tensor_parallel_size=1,
                gpu_memory_utilization=gpu_memory,
                hf_token=hf_token,
                enforce_eager=True,
                enable_lora=True,
                lora_modules={"defender": checkpoint},
                max_lora_rank=64,
            )
        else:
            llm_client = VLLMClient(
                model=checkpoint,
                tensor_parallel_size=1,
                gpu_memory_utilization=gpu_memory,
                hf_token=hf_token,
                enforce_eager=True,
            )

        if self.defender_type == "rl":
            self._defender = RLDefender(llm_client=llm_client)
        elif self.defender_type == "sft":
            self._defender = SFTDefender(llm_client=llm_client)
        else:
            raise ValueError(
                f"Unknown defender_type: {self.defender_type}. " f"Supported: sft, rl"
            )

        self._initialized = True
        self.logger.info(f"Intent-Defender ({self.defender_type}) loaded successfully.")

    def check_with_history(
        self,
        history: List[Dict[str, str]],
        current_query: str,
        current_response: str,
    ) -> Tuple[bool, str]:
        """
        Check if current turn should be blocked, given conversation history.

        Args:
            history: List of {"role": "user"/"assistant", "content": "..."} dicts
                     from ancestor nodes (depth 1..n-1), excluding root.
            current_query: The sub-query sent to target model (action content).
            current_response: The target model's response (observation).

        Returns:
            (is_safe, response):
                - (True, original_response) if passed
                - (False, refusal_response) if blocked
        """
        self._initialize()

        batch = [
            {
                "history": history,
                "current_prompt": current_query,
                "assistant_response": current_response,
            }
        ]

        actions = self._defender.predict_batch(batch)
        action = actions[0]

        self.stats["total_checked"] += 1

        if action == Action.BLOCK:
            self.stats["blocked"] += 1
            return False, REFUSAL_RESPONSE
        else:
            self.stats["passed"] += 1
            return True, current_response

    def get_stats(self) -> Dict[str, Any]:
        """Get defense statistics for current query."""
        total = self.stats["total_checked"]
        return {
            **self.stats,
            "block_rate": self.stats["blocked"] / total if total > 0 else 0.0,
        }

    def reset_stats(self):
        """Reset per-query statistics."""
        self.stats = {"total_checked": 0, "blocked": 0, "passed": 0}
