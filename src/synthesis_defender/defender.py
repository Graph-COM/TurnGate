from typing import List, Dict, Optional
from src.defender import BaseDefender, Action


class SynthesisGuardDefender(BaseDefender):
    """
    Defender that first synthesizes/summarizes the conversation,
    then passes the synthesis to a guard model for safety classification.

    This approach helps the guard focus on the core intent by removing noise
    and irrelevant details from the conversation.

    MEMORY OPTIMIZATION: Strict sequential model usage per predict_batch call:
    1) load/use synthesis model
    2) unload synthesis model
    3) load/use guard model
    This prevents two vLLM models from occupying VRAM simultaneously.
    """

    SYNTHESIS_PROMPT_TEMPLATE = """Summarize the following conversation, focusing on the key intent and content:

{conversation}

Summary:"""

    def __init__(self, synthesis_client: "LLMClient", guard_client: "LLMClient"):
        """
        Args:
            synthesis_client: LLM client for synthesizing conversations (e.g., Qwen3 4B)
            guard_client: LLM client for safety classification (e.g., Qwen3Guard-Gen)
        """
        self.synthesis_client = synthesis_client
        self.guard_client = guard_client
        self._synthesis_finalized = False
        self.force_full_batch = False

    def predict(
        self,
        history: List[Dict],
        current_prompt: str,
        assistant_response: Optional[str] = None,
    ) -> Action:
        return self.predict_batch(
            [
                {
                    "history": history,
                    "current_prompt": current_prompt,
                    "assistant_response": assistant_response,
                }
            ]
        )[0]

    def _to_conversation_text(self, item: Dict) -> str:
        conv_lines = []
        for turn in item["history"]:
            role = turn.get("role", "user").upper()
            content = turn.get("content", "")
            conv_lines.append(f"{role}: {content}")
        conv_lines.append(f"USER: {item['current_prompt']}")
        if item.get("assistant_response"):
            conv_lines.append(f"ASSISTANT: {item['assistant_response']}")
        return "\n".join(conv_lines)

    def _cleanup_cuda(self):
        try:
            import gc

            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    def predict_batch(self, batch: List[Dict]) -> List[Action]:
        # Step 1: Synthesize this batch
        synthesis_prompts = [
            self.SYNTHESIS_PROMPT_TEMPLATE.format(
                conversation=self._to_conversation_text(item)
            )
            for item in batch
        ]
        syntheses = self.synthesis_client.generate_batch(
            synthesis_prompts, max_tokens=256, temperature=0.5
        )

        # Step 2: Build messages list from syntheses and send to guard
        messages_list = []
        for synthesis in syntheses:
            messages = [{"role": "user", "content": synthesis}]
            messages_list.append(messages)

        # Use temperature=0 for deterministic guard classification
        guard_responses = self.guard_client.generate_batch_with_messages(
            messages_list, max_tokens=64, temperature=0
        )

        # Step 3: Parse guard outputs
        actions = []
        for response in guard_responses:
            response_clean = (response or "").strip().lower()
            # Handle both Llama Guard ("unsafe") and Qwen Guard ("Safety: Unsafe")
            if (
                response_clean.startswith("unsafe")
                or "safety: unsafe" in response_clean
                or "safety: controversial" in response_clean
            ):
                actions.append(Action.BLOCK)
            else:
                actions.append(Action.PASS)
        return actions

    def finalize_synthesis(self):
        """Best-effort final cleanup hook used by main.py after evaluation."""
        self._synthesis_finalized = True
        if hasattr(self.synthesis_client, "unload"):
            self.synthesis_client.unload()
        if hasattr(self.guard_client, "unload"):
            self.guard_client.unload()
        self._cleanup_cuda()
        print("Synthesis phase finalized. Unloaded synthesis/guard models from VRAM.")


class SynthesisLlamaGuardDefender(BaseDefender):
    """
    Two-stage defender that synthesizes conversations with Qwen3 4B,
    then classifies with Llama Guard 3.

    Memory-optimized strict sequential mode per predict_batch call:
    synthesize -> unload synthesis model -> load/use Llama Guard.

    Uses the same synthesis pipeline as SynthesisGuardDefender but with
    Llama Guard 3 (meta-llama/Llama-Guard-3-8B) instead of Qwen Guard.
    """

    SYNTHESIS_PROMPT_TEMPLATE = """Summarize the following conversation, focusing on the key intent and content:
{conversation}

Summary:"""

    def __init__(self, synthesis_client: "LLMClient", guard_client: "LLMClient"):
        """
        Args:
            synthesis_client: LLM client for synthesizing conversations (e.g., Qwen3 4B)
            guard_client: LLM client for Llama Guard 3 classification
        """
        self.synthesis_client = synthesis_client
        self.guard_client = guard_client
        self._synthesis_finalized = False
        self.force_full_batch = False

    def predict(
        self,
        history: List[Dict],
        current_prompt: str,
        assistant_response: Optional[str] = None,
    ) -> Action:
        return self.predict_batch(
            [
                {
                    "history": history,
                    "current_prompt": current_prompt,
                    "assistant_response": assistant_response,
                }
            ]
        )[0]

    def _to_conversation_text(self, item: Dict) -> str:
        conv_lines = []
        for turn in item["history"]:
            role = turn.get("role", "user").upper()
            content = turn.get("content", "")
            conv_lines.append(f"{role}: {content}")
        conv_lines.append(f"USER: {item['current_prompt']}")
        if item.get("assistant_response"):
            conv_lines.append(f"ASSISTANT: {item['assistant_response']}")
        return "\n".join(conv_lines)

    def _cleanup_cuda(self):
        try:
            import gc

            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    def predict_batch(self, batch: List[Dict]) -> List[Action]:
        # Step 1: Synthesize this batch
        synthesis_prompts = [
            self.SYNTHESIS_PROMPT_TEMPLATE.format(
                conversation=self._to_conversation_text(item)
            )
            for item in batch
        ]
        syntheses = self.synthesis_client.generate_batch(
            synthesis_prompts, max_tokens=256, temperature=0.5
        )

        # Step 2: Build messages list from syntheses and send to Llama Guard
        messages_list = []
        for synthesis in syntheses:
            messages = [{"role": "user", "content": synthesis}]
            messages_list.append(messages)

        # Use temperature=0 for deterministic guard classification
        guard_responses = self.guard_client.generate_batch_with_messages(
            messages_list, max_tokens=64, temperature=0
        )

        # Step 3: Parse Llama Guard outputs
        actions = []
        for response in guard_responses:
            response_clean = (response or "").strip().lower()
            # Llama Guard 3 output: "safe" or "unsafe [categories]"
            if response_clean.startswith("unsafe"):
                actions.append(Action.BLOCK)
            else:
                actions.append(Action.PASS)
        return actions

    def finalize_synthesis(self):
        """Best-effort final cleanup hook used by main.py after evaluation."""
        self._synthesis_finalized = True
        if hasattr(self.synthesis_client, "unload"):
            self.synthesis_client.unload()
        if hasattr(self.guard_client, "unload"):
            self.guard_client.unload()
        self._cleanup_cuda()
        print("Synthesis phase finalized. Unloaded synthesis/guard models from VRAM.")
