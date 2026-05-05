#!/usr/bin/env python3
"""
GPU Resource Manager for Jailbreak Experiments

This module provides centralized GPU allocation management for different components:
- Target model
- Attack models (controller, judge, etc.)
- Defense models (LLM Guard, etc.)
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ComponentType(Enum):
    """Types of components that need GPU allocation"""

    TARGET_MODEL = "target_model"
    ATTACK_MODEL = "attack_model"
    JUDGE_MODEL = "judge_model"
    DEFENSE_MODEL = "defense_model"


@dataclass
class GPUAllocation:
    """Represents a GPU allocation for a component"""

    component_type: ComponentType
    component_name: str
    gpu_ids: List[str]
    description: str = ""


class GPUResourceManager:
    """
    Centralized GPU resource manager for jailbreak experiments.

    Allocation strategy:
    1. Target model gets the first GPU
    2. Defense model (if enabled) gets the last GPU
    3. Remaining GPUs are distributed among attack methods
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize GPU resource manager.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger("GPUResourceManager")

        # Parse available GPUs from CUDA_VISIBLE_DEVICES
        self.available_gpus = self._parse_available_gpus()
        self.allocations: Dict[str, GPUAllocation] = {}

        self.logger.info(
            f"GPU Resource Manager initialized with {len(self.available_gpus)} GPUs: {self.available_gpus}"
        )

    def _parse_available_gpus(self) -> List[str]:
        """Parse available GPUs from CUDA_VISIBLE_DEVICES environment variable"""
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")
        gpus = [x.strip() for x in cuda_devices.split(",") if x.strip()]
        return gpus

    def allocate_target_model(self, model_name: str) -> List[str]:
        """
        Allocate GPU for target model.

        Args:
            model_name: Name of the target model

        Returns:
            List of GPU IDs allocated to target model
        """
        if len(self.available_gpus) == 0:
            raise ValueError("No GPUs available for allocation")

        # Target model gets the first GPU
        target_gpu = [self.available_gpus[0]]

        allocation = GPUAllocation(
            component_type=ComponentType.TARGET_MODEL,
            component_name=model_name,
            gpu_ids=target_gpu,
            description=f"Target model {model_name}",
        )

        self.allocations["target_model"] = allocation
        self.logger.info(f"Allocated GPU {target_gpu} for target model: {model_name}")

        return target_gpu

    def allocate_defense_model(self, model_name: str) -> List[str]:
        """
        Allocate GPU for defense model.

        Args:
            model_name: Name of the defense model

        Returns:
            List of GPU IDs allocated to defense model
        """
        if len(self.available_gpus) < 2:
            self.logger.warning("Not enough GPUs for defense model allocation")
            return []

        # Defense model gets the last GPU
        defense_gpu = [self.available_gpus[-1]]

        allocation = GPUAllocation(
            component_type=ComponentType.DEFENSE_MODEL,
            component_name=model_name,
            gpu_ids=defense_gpu,
            description=f"Defense model {model_name}",
        )

        self.allocations["defense_model"] = allocation
        self.logger.info(f"Allocated GPU {defense_gpu} for defense model: {model_name}")

        return defense_gpu

    def get_remaining_gpus(self) -> List[str]:
        """
        Get remaining GPUs available for attack methods.

        Returns:
            List of remaining GPU IDs
        """
        used_gpus = set()

        # Collect used GPUs
        for allocation in self.allocations.values():
            used_gpus.update(allocation.gpu_ids)

        # Return remaining GPUs
        remaining = [gpu for gpu in self.available_gpus if gpu not in used_gpus]
        self.logger.info(f"Remaining GPUs for attack methods: {remaining}")

        return remaining

    def allocate_attack_method(
        self,
        method_name: str,
        needs_controller: bool = True,
        needs_judge: bool = False,
        controller_model: str = "",
        judge_model: str = "",
    ) -> Dict[str, List[str]]:
        """
        Allocate GPUs for attack method components.

        Args:
            method_name: Name of the attack method
            needs_controller: Whether method needs controller model
            needs_judge: Whether method needs judge model
            controller_model: Name of controller model
            judge_model: Name of judge model

        Returns:
            Dictionary mapping component names to GPU IDs
        """
        remaining_gpus = self.get_remaining_gpus()

        if not remaining_gpus:
            self.logger.warning(f"No remaining GPUs for attack method: {method_name}")
            return {}

        allocations = {}

        if needs_controller and controller_model:
            # Controller gets one GPU
            controller_gpu = [remaining_gpus[0]]

            allocation = GPUAllocation(
                component_type=ComponentType.ATTACK_MODEL,
                component_name=f"{method_name}_controller",
                gpu_ids=controller_gpu,
                description=f"Controller model {controller_model} for {method_name}",
            )

            self.allocations[f"{method_name}_controller"] = allocation
            allocations["controller"] = controller_gpu

            self.logger.info(
                f"Allocated GPU {controller_gpu} for {method_name} controller: {controller_model}"
            )

            # Remove used GPU from remaining
            remaining_gpus = remaining_gpus[1:]

        if needs_judge and judge_model and remaining_gpus:
            # Judge gets remaining GPUs (can use multiple for large models)
            judge_gpus = remaining_gpus

            allocation = GPUAllocation(
                component_type=ComponentType.JUDGE_MODEL,
                component_name=f"{method_name}_judge",
                gpu_ids=judge_gpus,
                description=f"Judge model {judge_model} for {method_name}",
            )

            self.allocations[f"{method_name}_judge"] = allocation
            allocations["judge"] = judge_gpus

            self.logger.info(
                f"Allocated GPUs {judge_gpus} for {method_name} judge: {judge_model}"
            )

        return allocations

    def get_allocation(self, component_name: str) -> Optional[GPUAllocation]:
        """
        Get GPU allocation for a specific component.

        Args:
            component_name: Name of the component

        Returns:
            GPUAllocation object or None if not found
        """
        return self.allocations.get(component_name)

    def get_all_allocations(self) -> Dict[str, GPUAllocation]:
        """
        Get all GPU allocations.

        Returns:
            Dictionary of all allocations
        """
        return self.allocations.copy()

    def print_allocation_summary(self) -> None:
        """Print a summary of all GPU allocations"""
        self.logger.info("=" * 80)
        self.logger.info("GPU Allocation Summary")
        self.logger.info("=" * 80)

        for name, allocation in self.allocations.items():
            self.logger.info(
                f"{allocation.component_type.value.upper()}: {allocation.component_name}"
            )
            self.logger.info(f"  GPUs: {allocation.gpu_ids}")
            self.logger.info(f"  Description: {allocation.description}")
            self.logger.info("")

        remaining = self.get_remaining_gpus()
        if remaining:
            self.logger.info(f"Remaining GPUs: {remaining}")
        else:
            self.logger.info("No remaining GPUs")

        self.logger.info("=" * 80)


# Global GPU resource manager instance
_gpu_manager: Optional[GPUResourceManager] = None


def get_gpu_manager() -> GPUResourceManager:
    """
    Get the global GPU resource manager instance.

    Returns:
        GPUResourceManager instance
    """
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUResourceManager()
    return _gpu_manager


def initialize_gpu_manager(
    logger: Optional[logging.Logger] = None,
) -> GPUResourceManager:
    """
    Initialize the global GPU resource manager.

    Args:
        logger: Optional logger instance

    Returns:
        GPUResourceManager instance
    """
    global _gpu_manager
    _gpu_manager = GPUResourceManager(logger)
    return _gpu_manager
