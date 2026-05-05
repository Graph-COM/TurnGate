"""
Output Filter for CKA tree search.

Filters target model responses AFTER they are generated. If BLOCK, the
response is replaced with a refusal message. Does NOT terminate the search;
the node simply gets a low evaluation score.

Supported defender_type:
    - "naive_llm":              Direct LLM classification (0/1)
    - "llama_guard":            Llama Guard 3 safety moderation
    - "qwen_guard":             Qwen3Guard-Gen safety moderation
    - "synthesis_guard":        Synthesis + Qwen Guard (two-stage)
    - "synthesis_llama_guard":  Synthesis + Llama Guard (two-stage)

All support whitebox (local vLLM) and blackbox (API) via use_api config.
"""

import os
import sys
import logging
from typing import Dict, Any, List, Tuple

_INTENT_DEFENDER_ROOT = os.path.join(os.path.dirname(__file__), "../..")
if _INTENT_DEFENDER_ROOT not in sys.path:
    sys.path.insert(0, _INTENT_DEFENDER_ROOT)

from src.intention_analysis import VLLMClient, OpenAILLMClient
from src.defender import NaiveLLMDefender, Action
from src.guard_defender.defender import LlamaGuardDefender, QwenGuardDefender
from src.synthesis_defender.defender import (
    SynthesisGuardDefender,
    SynthesisLlamaGuardDefender,
)

REFUSAL_RESPONSE = (
    "I'm sorry, but I cannot provide this information "
    "due to safety policy restrictions."
)


class OutputFilter:
    """
    Output filter: checks history + query + response after target model.
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
            self.logger.info(f"Output filter pinned to GPU {gpu_id}")

        self.logger.info(
            f"Loading output filter ({self.defender_type}) from {checkpoint}..."
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

        if self.defender_type == "naive_llm":
            self._defender = NaiveLLMDefender(llm_client=llm_client)
        elif self.defender_type == "llama_guard":
            self._defender = LlamaGuardDefender(llm_client=llm_client)
        elif self.defender_type == "qwen_guard":
            self._defender = QwenGuardDefender(llm_client=llm_client)
        elif self.defender_type in ("synthesis_guard", "synthesis_llama_guard"):
            synthesis_model = self.config.get(
                "synthesis_model", "Qwen/Qwen3-4B-Instruct-2507"
            )
            synthesis_client = VLLMClient(
                model=synthesis_model,
                tensor_parallel_size=1,
                gpu_memory_utilization=gpu_memory,
                hf_token=hf_token,
                enforce_eager=True,
            )
            if self.defender_type == "synthesis_guard":
                self._defender = SynthesisGuardDefender(
                    synthesis_client=synthesis_client,
                    guard_client=llm_client,
                )
            else:
                self._defender = SynthesisLlamaGuardDefender(
                    synthesis_client=synthesis_client,
                    guard_client=llm_client,
                )
        else:
            raise ValueError(
                f"Unknown output filter type: {self.defender_type}. "
                f"Supported: naive_llm, llama_guard, qwen_guard, "
                f"synthesis_guard, synthesis_llama_guard"
            )

        self._initialized = True
        self.logger.info(f"Output filter ({self.defender_type}) loaded successfully.")

    def check(
        self,
        history: List[Dict[str, str]],
        current_query: str,
        current_response: str,
    ) -> bool:
        """
        Check if target response should be allowed.

        Args:
            history: Conversation history from ancestor nodes.
            current_query: The sub-query sent to target.
            current_response: The target model's response.

        Returns:
            True if safe (PASS), False if blocked (BLOCK).
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
