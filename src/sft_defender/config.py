"""
Configuration classes for SFT training.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class SFTTrainingConfig:
    """Configuration for SFT Defender training."""

    # Training type
    training_type: str = "lora"  # "lora" or "full"

    # Model configuration
    base_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    max_seq_length: int = 2048

    # LoRA-specific parameters (only used when training_type="lora")
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    lora_bias: str = "none"  # "none", "all", or "lora_only"

    # Training parameters
    output_dir: str = "checkpoints/sft_defender"
    num_epochs: int = 5
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Optimizer
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"

    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3
    eval_steps: Optional[int] = None
    evaluation_strategy: str = "no"  # "no", "steps", or "epoch"

    # Data
    train_data_path: str = "data/sft_training_data.jsonl"
    val_data_path: Optional[str] = None

    # Hardware
    fp16: bool = False
    bf16: bool = True  # Better for modern GPUs
    gradient_checkpointing: bool = True

    # Distributed training
    deepspeed: Optional[str] = (
        None  # Path to DeepSpeed config JSON (e.g., "src/sft_defender/ds_zero2.json")
    )
    fsdp: Optional[str] = (
        None  # FSDP strategy: "full_shard", "shard_grad_op", "offload"
    )
    fsdp_config: Optional[str] = None  # Path to FSDP config JSON
    ddp_find_unused_parameters: bool = False
    ddp_timeout: int = (
        7200  # DDP timeout in seconds (default 2h; covers long vLLM eval)
    )

    # Resume training
    resume: bool = False  # Whether to resume training from a saved checkpoint
    resume_model_path: Optional[str] = (
        None  # Path to checkpoint dir (e.g., "checkpoints/sft_defender_full/epoch_2")
    )
    resume_init_eval: bool = (
        False  # Run val+test defense eval on resumed model before training starts
    )

    # Reward Shaping (only used when reward_shaping=True)
    reward_shaping: bool = False  # Enable reward shaping weighted loss
    balanced_mode: bool = (
        False  # If True, use class-balancing mode (only block_weight/pass_weight, ignores prediction)
    )
    block_weight: float = 1.0  # Weight for Accurate Block (r_tp, baseline)
    miss_block_weight: float = 3.0  # Weight for Miss Block (r_miss, heaviest penalty)
    early_block_weight: float = 0.3  # Base weight for Early Block (r_early)
    gap_penalty_scale: float = 0.5  # Scale factor for Early Block gap penalty
    false_block_weight: float = (
        1.0  # Weight for False Block (r_fp, false alarm on benign)
    )
    pass_weight: float = 0.3  # Weight for Correct Pass (r_bp)

    # W&B
    wandb_project: Optional[str] = None  # Set to enable W&B logging; None = disabled
    wandb_run_name: Optional[str] = None  # W&B run name; None = auto-generated

    # Defense evaluation during training
    eval_data_dir: Optional[str] = (
        None  # Dir with pre-generated eval conversations (pkl files)
    )
    eval_gpu: Optional[int] = (
        None  # GPU index (within CUDA_VISIBLE_DEVICES) for vLLM eval; None = last visible GPU
    )
    defense_batch_size: int = 500  # batch_size for Evaluator.evaluate()
    gpu_memory_utilization: float = 0.8  # vLLM GPU memory utilization for evaluation

    # MDA mode: use input-only prompts (no assistant response) for MDA dataset
    mda_mode: bool = False

    # Hugging Face
    hf_token: Optional[str] = None

    # Random seed
    seed: int = 42

    def __post_init__(self):
        """Validate and adjust configuration."""
        if self.training_type not in ["lora", "full"]:
            raise ValueError(
                f"training_type must be 'lora' or 'full', got {self.training_type}"
            )

        # Adjust learning rate for full fine-tuning (typically lower)
        if self.training_type == "full" and self.learning_rate > 1e-5:
            print(
                f"Warning: learning_rate {self.learning_rate} might be too high for full fine-tuning."
            )
            print(f"Consider using 1e-5 or lower.")

        # Resume validation
        if self.resume and not self.resume_model_path:
            raise ValueError("--resume requires --resume-model-path")
        if self.resume_init_eval and not self.resume:
            raise ValueError("--resume-init-eval requires --resume")

        # Append training type to output dir
        self.output_dir = f"{self.output_dir}_{self.training_type}"

    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})
