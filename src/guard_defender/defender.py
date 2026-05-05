from typing import List, Dict, Optional
from src.defender import BaseDefender, Action


class LlamaGuardDefender(BaseDefender):
    """
    Defender using Llama Guard 3 for safety moderation.
    Expects 'safe' or 'unsafe' in the output.
    """

    def __init__(self, llm_client: "LLMClient"):
        self.llm_client = llm_client

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

    def predict_batch(self, batch: List[Dict]) -> List[Action]:
        # Build a standard messages list for each item so that the model's own
        # chat template (which contains the S1-S14 hazard categories) is applied
        # correctly via generate_batch_with_messages.
        messages_list = []
        for item in batch:
            messages = [
                {"role": turn.get("role", "user"), "content": turn.get("content", "")}
                for turn in item["history"]
            ]
            messages.append({"role": "user", "content": item["current_prompt"]})
            if item.get("assistant_response"):
                messages.append(
                    {"role": "assistant", "content": item["assistant_response"]}
                )
            messages_list.append(messages)

        # Use temperature=0 for deterministic, reproducible classification.
        responses = self.llm_client.generate_batch_with_messages(
            messages_list, max_tokens=64, temperature=0
        )

        actions = []
        for response in responses:
            response_clean = (response or "").strip().lower()
            # Llama Guard 3 output: first line is "safe" or "unsafe"
            if response_clean.startswith("unsafe"):
                actions.append(Action.BLOCK)
            else:
                actions.append(Action.PASS)
        return actions


class QwenGuardDefender(BaseDefender):
    """
    Defender using Qwen3Guard-Gen for safety moderation.
    Expects 'Safety: Safe' or 'Safety: Unsafe' in the output.
    """

    def __init__(self, llm_client: "LLMClient"):
        self.llm_client = llm_client

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

    def predict_batch(self, batch: List[Dict]) -> List[Action]:
        # Build a standard messages list for each item so that Qwen3Guard-Gen's
        # chat template (which encodes the full safety policy) is applied correctly
        # via generate_batch_with_messages.
        messages_list = []
        for item in batch:
            messages = [
                {"role": turn.get("role", "user"), "content": turn.get("content", "")}
                for turn in item["history"]
            ]
            messages.append({"role": "user", "content": item["current_prompt"]})
            if item.get("assistant_response"):
                messages.append(
                    {"role": "assistant", "content": item["assistant_response"]}
                )
            messages_list.append(messages)

        # Use temperature=0 for deterministic, reproducible classification.
        responses = self.llm_client.generate_batch_with_messages(
            messages_list, max_tokens=64, temperature=0
        )

        actions = []
        for response in responses:
            response_clean = (response or "").strip().lower()
            # Qwen3Guard-Gen output: "Safety: Safe / Unsafe / Controversial"
            # Both "Unsafe" and "Controversial" are treated as BLOCK.
            if (
                "safety: unsafe" in response_clean
                or "safety: controversial" in response_clean
            ):
                actions.append(Action.BLOCK)
            else:
                actions.append(Action.PASS)
        return actions
