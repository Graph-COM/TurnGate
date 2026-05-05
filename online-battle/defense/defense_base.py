"""
Base class for input-level defense mechanisms.
Provides abstract interface for checking and filtering jailbreak prompts.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import logging


class BaseDefense(ABC):
    """
    Abstract base class for all defense mechanisms.

    Each defense method should implement the check_prompt method to
    detect unsafe prompts and optionally generate refusal responses.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize defense mechanism.

        Args:
            config: Defense configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.name = self.__class__.__name__

        # Statistics tracking
        self.stats = {"total_checked": 0, "blocked": 0, "allowed": 0, "errors": 0}

    @abstractmethod
    def check_prompt(self, prompt: str) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Check if a prompt is safe or should be blocked.

        Args:
            prompt: Input prompt to check

        Returns:
            Tuple containing:
                - is_safe (bool): True if prompt is safe, False if should be blocked
                - refusal_response (Optional[str]): Pre-generated refusal if blocked, None otherwise
                - metadata (Dict[str, Any]): Additional information about the check
                    - 'confidence': confidence score of the decision
                    - 'reason': explanation of the decision
                    - 'category': unsafe category if blocked
        """
        pass

    def update_stats(self, is_safe: bool, had_error: bool = False):
        """Update defense statistics."""
        self.stats["total_checked"] += 1
        if had_error:
            self.stats["errors"] += 1
        elif is_safe:
            self.stats["allowed"] += 1
        else:
            self.stats["blocked"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get defense statistics."""
        return {
            **self.stats,
            "block_rate": (
                self.stats["blocked"] / self.stats["total_checked"]
                if self.stats["total_checked"] > 0
                else 0.0
            ),
        }

    def reset_stats(self):
        """Reset statistics counters."""
        self.stats = {"total_checked": 0, "blocked": 0, "allowed": 0, "errors": 0}
