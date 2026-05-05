from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import math


class ActionType(Enum):
    """
    Unified action type enum shared by controller and agent.

    Keeping this in a standalone module prevents circular imports between
    cka_agent.py and controller_llm.py.
    """

    DECOMPOSE = "decompose"  # Break down into sub-goals
    QUERY_TARGET = "query_target"  # Query the target LLM
    USE_TOOL = "use_tool"  # Use an external tool
    SYNTHESIZE = "synthesize"  # Synthesize final answer
    REFLECT = "reflect"  # Reflect on failure


@dataclass
class TreeNode:
    """Node in the search tree."""

    node_id: str
    parent_id: Optional[str]
    depth: int
    state: Dict[str, Any]  # Current state including goal, observations, etc.
    action: Optional[Dict[str, Any]] = None  # Action taken to reach this node
    observation: Optional[str] = None  # Result of the action
    value_score: float = 0.0  # Value estimate for UCT
    visit_count: int = 0  # Number of visits for UCT
    is_terminal: bool = False
    is_success: bool = False
    children: List[str] = field(default_factory=list)  # Child node IDs
    synthesized_attempt: Optional[str] = None  # Stores synthesis even if failed

    # Additional fields for better tracking
    trajectory_from_root: List[str] = field(default_factory=list)  # Path from root
    accumulated_context: str = ""  # Accumulated context from root to this node

    # ASR evaluation fields
    asr_classification: Optional[str] = None  # "A", "B", "C", or "D"
    asr_confidence: float = 0.0  # Confidence in classification
    synthesized_response: Optional[str] = None  # Stored synthesized response
    is_fallback_candidate: bool = False  # Mark as potential fallback

    def uct_score(self, parent_visits: int, exploration_weight: float = 1.414) -> float:
        """Calculate UCT score for node selection."""
        if self.visit_count == 0:
            return float("inf")

        exploitation = self.value_score / self.visit_count
        exploration = exploration_weight * math.sqrt(
            (2 * math.log(parent_visits)) / self.visit_count
        )
        return exploitation + exploration
