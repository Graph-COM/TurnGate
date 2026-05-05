"""
Factory for creating defense mechanisms.
Provides centralized defense instantiation based on configuration.
"""

from typing import Dict, Any, Optional
import logging

from defense.defense_base import BaseDefense
from defense.llm_guard import LLMGuardDefense
from defense.grayswanai_guard import GraySwanAIGuardDefense
from defense.rephrasing_defense import RephrasingDefense
from defense.perturbation_defense import PerturbationDefense


def create_defense(defense_type: str, config: Dict[str, Any]) -> Optional[BaseDefense]:
    """
    Factory function to create defense mechanisms.

    Args:
        defense_type: Type of defense to create ('llm_guard', 'perplexity_filter', etc.)
        config: Defense configuration dictionary

    Returns:
        BaseDefense instance or None if defense_type is 'none' or invalid

    Raises:
        ValueError: If defense_type is not supported
    """
    logger = logging.getLogger("DefenseFactory")

    if not defense_type or defense_type.lower() == "none":
        logger.info("No defense mechanism enabled")
        return None

    defense_type = defense_type.lower()

    if defense_type == "llm_guard":
        logger.info(
            f"Creating LLM Guard defense with model: {config.get('guard_model_name', 'default')}"
        )
        return LLMGuardDefense(config)

    elif defense_type == "grayswanai_guard":
        logger.info(
            f"Creating GraySwanAI Guard defense with model: {config.get('guard_model_name', 'default')}"
        )
        return GraySwanAIGuardDefense(config)

    elif defense_type == "rephrasing":
        logger.info(
            f"Creating Rephrasing defense with model: {config.get('rephrase_model_name', 'default')}"
        )
        return RephrasingDefense(config)

    elif defense_type == "perturbation":
        logger.info("Creating Perturbation defense")
        return PerturbationDefense(config)

    else:
        raise ValueError(
            f"Unsupported defense type: {defense_type}. "
            f"Supported types: {supported_defenses()}"
        )


def supported_defenses():
    """Return list of supported defense mechanisms."""
    return [
        "none",
        "llm_guard",
        "grayswanai_guard",
        "rephrasing",
        "perturbation",
    ]  # Add more as implemented
