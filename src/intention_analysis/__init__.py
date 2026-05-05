"""
Intention Analysis defense: two-stage pipeline (identify intention -> aligned response).
"""

from .llm_client import LLMClient, OpenAILLMClient, VLLMClient
from .pipeline import (
    REFUSAL_RESPONSE,
    STAGE1_PROMPT_TEMPLATE,
    STAGE2_CLASSIFY_TEMPLATE,
    IntentionAnalysisPipeline,
    intention_suggests_harmful,
)

__all__ = [
    "LLMClient",
    "OpenAILLMClient",
    "VLLMClient",
    "REFUSAL_RESPONSE",
    "STAGE1_PROMPT_TEMPLATE",
    "STAGE2_CLASSIFY_TEMPLATE",
    "IntentionAnalysisPipeline",
    "intention_suggests_harmful",
]
