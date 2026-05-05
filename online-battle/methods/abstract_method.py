from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging


class AbstractJailbreakMethod(ABC):
    """
    Abstract base class for all jailbreak methods.

    This class defines the common interface that all jailbreak methods
    must implement, ensuring consistency across different approaches.
    """

    def __init__(self, name: str, config: Dict[str, Any], model=None):
        """
        Initialize the jailbreak method.

        Args:
            name (str): Name of the method
            config (Dict[str, Any]): Method-specific configuration
            model: Target model instance (optional, can be set later)
        """
        self.name = name
        self.config = config
        self.model = model
        self.logger = logging.getLogger(f"JailbreakMethod.{name}")

        # Method statistics
        self.stats = {
            "total_attempts": 0,
            "successful_attempts": 0,
            "failed_attempts": 0,
            "error_attempts": 0,
        }

    @abstractmethod
    def generate_jailbreak(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a jailbreak attempt for the given query.

        This is the core method that each jailbreak technique must implement.
        It takes a harmful query and returns the model's response along with
        metadata about the attempt.

        Args:
            query (str): The original harmful query/prompt
            **kwargs: Additional method-specific parameters

        Returns:
            Dict[str, Any]: Result dictionary containing:
                - 'original_query': Original input query
                - 'jailbreak_prompt': Modified prompt used for jailbreak
                - 'response': Model's response to the jailbreak prompt
                - 'success': Boolean indicating if jailbreak was successful
                - 'metadata': Additional method-specific information
                - 'error': Error message if something went wrong (optional)

        Raises:
            NotImplementedError: If the method is not implemented by subclass
        """
        pass

    # def generate_jailbreak_batch(self, queries: List[str], **kwargs) -> List[Dict[str, Any]]:
    #     """
    #     Generate jailbreak attempts for a batch of queries.

    #     This is an optional method that can be overridden by subclasses for
    #     more efficient batch processing. The default implementation calls
    #     generate_jailbreak for each query individually.

    #     Args:
    #         queries (List[str]): List of original harmful queries
    #         **kwargs: Additional method-specific parameters

    #     Returns:
    #         List[Dict[str, Any]]: List of result dictionaries (same format as generate_jailbreak)
    #     """
    #     results = []
    #     for query in queries:
    #         result = self.generate_jailbreak(query, **kwargs)
    #         results.append(result)
    #     return results

    @abstractmethod
    def prepare_prompt(self, query: str, **kwargs) -> str:
        """
        Prepare the jailbreak prompt for the given query.

        This method transforms the original query into a format that
        attempts to bypass the model's safety mechanisms.

        Args:
            query (str): Original harmful query
            **kwargs: Method-specific parameters

        Returns:
            str: Modified prompt for jailbreak attempt

        Raises:
            NotImplementedError: If the method is not implemented by subclass
        """
        pass

    def set_model(self, model) -> None:
        """
        Set the target model for this method.

        Args:
            model: Target model instance
        """
        self.model = model
        self.logger.info(f"Model set for method {self.name}")

    def validate_config(self) -> bool:
        """
        Validate the method configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        # Base validation - can be overridden by subclasses
        if not isinstance(self.config, dict):
            self.logger.error("Configuration must be a dictionary")
            return False

        return True

    def update_stats(self, success: bool, error: bool = False) -> None:
        """
        Update method statistics.

        Args:
            success (bool): Whether the attempt was successful
            error (bool): Whether an error occurred
        """
        self.stats["total_attempts"] += 1

        if error:
            self.stats["error_attempts"] += 1
        elif success:
            self.stats["successful_attempts"] += 1
        else:
            self.stats["failed_attempts"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """
        Get method statistics.

        Returns:
            Dict[str, Any]: Statistics dictionary with success rates
        """
        total = self.stats["total_attempts"]
        if total == 0:
            return {**self.stats, "success_rate": 0.0, "error_rate": 0.0}

        success_rate = self.stats["successful_attempts"] / total
        error_rate = self.stats["error_attempts"] / total

        return {
            **self.stats,
            "success_rate": round(success_rate, 4),
            "error_rate": round(error_rate, 4),
        }

    def reset_stats(self) -> None:
        """Reset method statistics."""
        self.stats = {
            "total_attempts": 0,
            "successful_attempts": 0,
            "failed_attempts": 0,
            "error_attempts": 0,
        }

    def __str__(self) -> str:
        """String representation of the method."""
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __repr__(self) -> str:
        """Detailed string representation of the method."""
        return f"{self.__class__.__name__}(name='{self.name}', config={self.config})"
