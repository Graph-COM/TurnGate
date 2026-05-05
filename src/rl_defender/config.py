"""
Configuration for Multi-turn RL Defender (offline GRPO).
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class RLDefenderConfig:
    """
    Configuration for offline GRPO training of the RL Defender.

    Training pipeline:
      1. Run data_preparation.py with a reference model checkpoint to produce
         a prepared JSONL (per-turn context, action, ref log-probs, advantage).
      2. Run train.py with this config to train the policy via offline GRPO.
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
    # Path(s) to the prepared GRPO JSONL file(s) output by data_preparation.py.
    prepared_train_data: str = ""
    prepared_val_data: Optional[str] = None

    # Output directory (training_type is appended automatically in __post_init__)
    output_dir: str = "checkpoints/rl_defender"

    # ------------------------------------------------------------------ GRPO
    # β: KL-divergence penalty weight (keeps policy close to reference).
    kl_coeff: float = 0.1
    # ε: denominator stability for within-group advantage normalisation.
    advantage_epsilon: float = 1e-6
    # Whether to apply importance-sampling (IS) ratio clipping (PPO-style).
    use_is_ratio: bool = False
    # PPO clip range (only used when use_is_ratio=True).
    clip_epsilon: float = 0.2
    # DAPO-style clip higher: asymmetric clipping with a larger upper bound.
    clip_higher: bool = False
    # Upper clip epsilon (only used when clip_higher=True). ratio ∈ [1-clip_epsilon, 1+clip_epsilon_high].
    clip_epsilon_high: float = 0.28

    # ------------------------------------------------------------------ reward
    # Turn-level reward weights used by data_preparation.py.
    # These map to the reward structure in the image:
    #   r_tp   = accurate_block_reward   (block at target turn)
    #   r_bp   = correct_pass_reward     (pass at non-target turn)
    #   r_early= early_block_base + gap * early_block_gap_scale  (early block penalty)
    #   r_miss = miss_penalty            (fail to block target turn)
    #   r_fp   = false_block_penalty     (block in a benign conv)
    #   r_late = late_block_penalty      (block AFTER target turn was missed)
    accurate_block_reward: float = 1.0
    correct_pass_reward: float = 0.1
    early_block_base: float = -0.3
    early_block_gap_scale: float = 0.1
    miss_penalty: float = -1.0
    false_block_penalty: float = -0.5
    late_block_penalty: float = -0.3

    # ------------------------------------------------------------ data-prep
    # Number of rollouts per prompt group (G in GRPO).
    num_rollouts_per_group: int = 10
    # Max sequence length for tokenisation during data preparation.
    data_prep_max_seq_length: int = 2048
    # Batch size for model forward passes during data preparation.
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

    # Optimiser
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"

    # ---------------------------------------------------------------- hardware
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True

    # Distributed training
    deepspeed: Optional[str] = None
    fsdp: Optional[str] = None
    fsdp_config: Optional[str] = None
    ddp_find_unused_parameters: bool = False
    ddp_timeout: int = 7200

    # Resume
    resume: bool = False
    resume_model_path: Optional[str] = None
    resume_init_eval: bool = False  # run val+test eval on resumed model before training

    # ------------------------------------------------------------------- W&B
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None

    # ----------------------------------------- defense evaluation during training
    # Set eval_data_dir to enable periodic defense evaluation via vLLM subprocess,
    # identical to the SFT DefenseEvalCallback mechanism.
    eval_data_dir: Optional[str] = (
        None  # dir with val_conversations.pkl / test_conversations.pkl
    )
    eval_gpu: Optional[int] = (
        None  # GPU index for vLLM eval (default: last visible GPU)
    )
    eval_steps: Optional[int] = (
        None  # eval every N steps (default: steps_per_epoch // 10)
    )
    eval_epochs_only: bool = (
        False  # if True, only eval at epoch end (no mid-epoch step evals)
    )
    defense_batch_size: int = 500  # batch size for Evaluator.evaluate()
    gpu_memory_utilization: float = 0.8  # vLLM GPU memory utilization

    # --------------------------------------------------------- checkpoint saving
    # save_steps=None  → save once per epoch (save_strategy="epoch")
    # save_steps=N     → save every N optimizer steps (save_strategy="steps")
    save_steps: Optional[int] = None

    # ----------------------------------------------------------------- other
    mda_mode: bool = False  # Use MDA input-only prompts (no assistant response)
    hf_token: Optional[str] = None
    seed: int = 42

    def __post_init__(self):
        if self.training_type not in ("lora", "full"):
            raise ValueError(
                f"training_type must be 'lora' or 'full', got '{self.training_type}'"
            )
        if self.resume and not self.resume_model_path:
            raise ValueError("--resume requires --resume-model-path")
        # Append training type to avoid collision with SFT checkpoints.
        if not self.output_dir.endswith(f"_{self.training_type}"):
            self.output_dir = f"{self.output_dir}_{self.training_type}"

    @classmethod
    def from_dict(cls, d: dict) -> "RLDefenderConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
