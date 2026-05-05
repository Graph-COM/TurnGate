"""
Configuration for TurnGate (GAE-Augmented Policy Optimization) Defender.

Extends the RL Defender config with GAE-specific parameters for
turn-level advantage estimation.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class TurnGateDefenderConfig:
    """
    Configuration for offline TurnGate training.

    Differences from GRPO:
      - Turn-level reward normalisation (across all turns in a group, not trajectory-level).
      - GAE (Generalised Advantage Estimation) backwards pass per rollout.
      - Per-turn advantages instead of shared trajectory advantage.
    """

    # ------------------------------------------------------------------ model
    base_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    training_type: str = "lora"  # "lora" or "full"

    # LoRA parameters (ignored when training_type="full")
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    lora_bias: str = "none"

    # ------------------------------------------------------------------ data
    prepared_train_data: str = ""
    prepared_val_data: Optional[str] = None
    output_dir: str = "checkpoints/turngate_defender"

    # ------------------------------------------------------------------ TurnGate
    # KL-divergence penalty weight.
    kl_coeff: float = 0.1
    # Denominator stability for turn-level reward normalisation.
    advantage_epsilon: float = 1e-6
    # IS ratio clipping.
    use_is_ratio: bool = False
    clip_epsilon: float = 0.2
    # DAPO-style clip higher: asymmetric clipping with a larger upper bound.
    clip_higher: bool = False
    # Upper clip epsilon (only used when clip_higher=True). ratio ∈ [1-clip_epsilon, 1+clip_epsilon_high].
    clip_epsilon_high: float = 0.28

    # GAE parameters
    gae_gamma: float = 1.0  # discount factor (1.0 = no discount, episodes are short)
    gae_lambda: float = 1.0  # GAE lambda (1.0 = full Monte Carlo return)

    # ------------------------------------------------------------------ reward
    accurate_block_reward: float = 1.0
    correct_pass_reward: float = 0.1
    early_block_base: float = -0.3
    early_block_gap_scale: float = 0.1
    miss_penalty: float = -1.0
    false_block_penalty: float = -0.5
    late_block_penalty: float = -0.3

    # ------------------------------------------------------------ data-prep
    num_rollouts_per_group: int = 10
    data_prep_max_seq_length: int = 2048
    data_prep_batch_size: int = 8

    # --------------------------------------------------------------- training
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    max_seq_length: int = 2048

    optim: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"

    # ---------------------------------------------------------------- hardware
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True

    deepspeed: Optional[str] = None
    fsdp: Optional[str] = None
    fsdp_config: Optional[str] = None
    ddp_find_unused_parameters: bool = False
    ddp_timeout: int = 7200

    # Resume
    resume: bool = False
    resume_model_path: Optional[str] = None
    resume_init_eval: bool = False

    # ------------------------------------------------------------------- W&B
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None

    # ----------------------------------------- defense evaluation during training
    eval_data_dir: Optional[str] = None
    eval_gpu: Optional[int] = None
    eval_steps: Optional[int] = None
    eval_epochs_only: bool = False
    defense_batch_size: int = 500
    gpu_memory_utilization: float = 0.8

    # --------------------------------------------------------- checkpoint saving
    save_steps: Optional[int] = None

    # ----------------------------------------------------------------- other
    hf_token: Optional[str] = None
    seed: int = 42

    def __post_init__(self):
        if self.training_type not in ("lora", "full"):
            raise ValueError(
                f"training_type must be 'lora' or 'full', got '{self.training_type}'"
            )
        if self.resume and not self.resume_model_path:
            raise ValueError("--resume requires --resume-model-path")
        if not self.output_dir.endswith(f"_{self.training_type}"):
            self.output_dir = f"{self.output_dir}_{self.training_type}"

    @classmethod
    def from_dict(cls, d: dict) -> "TurnGateDefenderConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
