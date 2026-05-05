"""
Input Filter for CKA tree search.

Filters queries BEFORE they reach the target model. If BLOCK, the query is
not sent to the target — a refusal response is used instead. Does NOT
terminate the search; the node simply gets a low evaluation score.

Supported defender_type:
    - "intention_analysis":  Two-stage intention extraction + classification
    - "lsm":                Lightweight Sequential Monitor

Both support whitebox (local vLLM) and blackbox (API) via use_api config.
"""

import os
import sys
import logging
from typing import Dict, Any, List, Tuple

_INTENT_DEFENDER_ROOT = os.path.join(os.path.dirname(__file__), "../..")
if _INTENT_DEFENDER_ROOT not in sys.path:
    sys.path.insert(0, _INTENT_DEFENDER_ROOT)

from src.intention_analysis import VLLMClient, OpenAILLMClient
from src.defender import IntentionAnalysisDefender, SequentialMonitorDefender, Action

REFUSAL_RESPONSE = (
    "I'm sorry, but I cannot provide this information "
    "due to safety policy restrictions."
)


class InputFilter:
    """
    Input filter: checks query + history before target model.
    BLOCK → replace response with refusal, do NOT terminate search.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.defender_type = config.get("defender_type")
        self.logger = logging.getLogger(self.__class__.__name__)
        self.stats = {"total_checked": 0, "blocked": 0, "passed": 0}
        self._defender = None
        self._initialized = False

    def _initialize(self):
        if self._initialized:
            return

        checkpoint = self.config["checkpoint"]
        gpu_memory = self.config.get("gpu_memory_utilization", 0.8)
        hf_token = self.config.get("hf_token")
        use_api = self.config.get("use_api", False)

        gpu_id = self.config.get("_gpu_id")
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            self.logger.info(f"Input filter pinned to GPU {gpu_id}")

        self.logger.info(
            f"Loading input filter ({self.defender_type}) from {checkpoint}..."
        )

        if use_api:
            llm_client = OpenAILLMClient(
                model=checkpoint,
                api_key=self.config.get("api_key"),
                base_url=self.config.get("base_url"),
            )
        else:
            llm_client = VLLMClient(
                model=checkpoint,
                tensor_parallel_size=1,
                gpu_memory_utilization=gpu_memory,
                hf_token=hf_token,
                enforce_eager=True,
            )

        if self.defender_type == "intention_analysis":
            self._defender = IntentionAnalysisDefender(llm_client=llm_client)
        elif self.defender_type == "lsm":
            threshold = self.config.get("lsm_threshold", 0.5)
            self._defender = SequentialMonitorDefender(
                llm_client=llm_client,
                threshold=threshold,
            )
        else:
            raise ValueError(
                f"Unknown input filter type: {self.defender_type}. "
                f"Supported: intention_analysis, lsm"
            )

        self._initialized = True
        self.logger.info(f"Input filter ({self.defender_type}) loaded successfully.")

    def check(
        self,
        history: List[Dict[str, str]],
        current_query: str,
    ) -> bool:
        """
        Check if query should be allowed through to target model.

        Args:
            history: Conversation history from ancestor nodes.
            current_query: The sub-query to be sent to target.

        Returns:
            True if safe (PASS), False if blocked (BLOCK).
        """
        self._initialize()

        batch = [
            {
                "history": history,
                "current_prompt": current_query,
                "assistant_response": "",  # no response yet
            }
        ]

        actions = self._defender.predict_batch(batch)
        action = actions[0]

        self.stats["total_checked"] += 1
        if action == Action.BLOCK:
            self.stats["blocked"] += 1
            return False
        else:
            self.stats["passed"] += 1
            return True

    def get_stats(self) -> Dict[str, Any]:
        total = self.stats["total_checked"]
        return {
            **self.stats,
            "block_rate": self.stats["blocked"] / total if total > 0 else 0.0,
        }

    def reset_stats(self):
        self.stats = {"total_checked": 0, "blocked": 0, "passed": 0}
