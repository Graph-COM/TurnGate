"""
SFT Defender: Supervised Fine-Tuning based defense method.

This module provides training and inference capabilities for a defender
that uses a fine-tuned model (via LoRA or full fine-tuning) to classify
harmful vs benign queries in multi-turn conversations.
"""

from .data_preparation import prepare_sft_data_all_splits, SFTDataset
from .train import train_sft_defender, SFTTrainingConfig

__all__ = [
    "prepare_sft_data_all_splits",
    "SFTDataset",
    "train_sft_defender",
    "SFTTrainingConfig",
]
