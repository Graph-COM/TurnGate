"""
Training script for the TurnGate (GAE-Augmented Policy Optimization) Defender.

Requires a prepared JSONL dataset produced by turngate data_preparation.py.

Typical usage (LoRA):
  python -m src.turngate_defender.train \\
      --prepared-train-data data/turngate_defender/turngate_train.jsonl \\
      --base-model          Qwen/Qwen3-4B-Instruct-2507 \\
      --training-type       lora \\
      --output-dir          checkpoints/turngate_defender \\
      --num-epochs          3

Multi-GPU with DDP:
  torchrun --nproc_per_node 4 -m src.turngate_defender.train \\
      --prepared-train-data ... \\
      --training-type full
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    default_data_collator,
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.turngate_defender.config import TurnGateDefenderConfig
from src.turngate_defender.dataset import TurnGateDataset
from src.turngate_defender.trainer import TurnGateDefenderTrainer
from src.sft_defender.eval_callback import DefenseEvalCallback


def _log(msg: str):
    if int(os.environ.get("LOCAL_RANK", -1)) in (-1, 0):
        print(msg)


# ----------------------------------------------------------------- model setup


def load_model_and_tokenizer(config: TurnGateDefenderConfig):
    if config.resume and config.resume_model_path and config.training_type == "full":
        model_path = config.resume_model_path
        _log(f"[Resume] Loading full-FT model from {model_path}")
    else:
        model_path = config.base_model
        _log(f"Loading model from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, token=config.hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=(
            torch.bfloat16
            if config.bf16
            else (torch.float16 if config.fp16 else torch.float32)
        ),
        token=config.hf_token,
    )

    if config.gradient_checkpointing and not config.deepspeed:
        model.gradient_checkpointing_enable()

    return model, tokenizer


def setup_lora(model, config: TurnGateDefenderConfig):
    if config.resume and config.resume_model_path:
        _log(f"[Resume] Loading LoRA adapter from {config.resume_model_path}")
        model = PeftModel.from_pretrained(model, config.resume_model_path)
        model.train()
    else:
        _log("Setting up LoRA adapter ...")
        lora_cfg = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias=config.lora_bias,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


# ------------------------------------------------------------------ training


def train_turngate_defender(config: TurnGateDefenderConfig) -> str:
    torch.manual_seed(config.seed)

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_main = local_rank in (-1, 0)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_main:
        with open(output_dir / "turngate_training_config.json", "w") as f:
            json.dump(config.__dict__, f, indent=2, default=str)

    # ── model & tokenizer ─────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(config)

    if config.training_type == "lora":
        model = setup_lora(model, config)
    else:
        _log("Full fine-tuning mode")
        _log(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── token IDs ─────────────────────────────────────────────────────────
    block_tokens = tokenizer.encode("0", add_special_tokens=False)
    pass_tokens = tokenizer.encode("1", add_special_tokens=False)
    if len(block_tokens) != 1 or len(pass_tokens) != 1:
        raise ValueError(
            f"'0' tokenises to {block_tokens}, '1' tokenises to {pass_tokens}. "
            "Both must be single tokens."
        )
    block_token_id = block_tokens[0]
    pass_token_id = pass_tokens[0]
    _log(f"Token IDs  ->  '0' (BLOCK): {block_token_id},  '1' (PASS): {pass_token_id}")

    # ── dataset ───────────────────────────────────────────────────────────
    _log(f"Loading training data from {config.prepared_train_data} ...")
    train_dataset = TurnGateDataset(
        config.prepared_train_data, tokenizer, config.max_seq_length
    )
    _log(f"  Training turn-samples: {len(train_dataset)}")

    val_dataset = None
    if config.prepared_val_data:
        _log(f"Loading validation data from {config.prepared_val_data} ...")
        val_dataset = TurnGateDataset(
            config.prepared_val_data, tokenizer, config.max_seq_length
        )
        _log(f"  Validation turn-samples: {len(val_dataset)}")

    # ── step calculations ─────────────────────────────────────────────────
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    effective_batch = (
        config.per_device_train_batch_size
        * config.gradient_accumulation_steps
        * world_size
    )
    steps_per_epoch = math.ceil(len(train_dataset) / effective_batch)
    if config.eval_epochs_only:
        eval_steps = 10**9
    else:
        eval_steps = config.eval_steps or max(1, steps_per_epoch // 10)
    eval_freq_str = (
        "epoch end only" if config.eval_epochs_only else f"every {eval_steps} steps"
    )
    _log(f"Steps per epoch: {steps_per_epoch}, defense eval: {eval_freq_str}")

    # ── W&B ───────────────────────────────────────────────────────────────
    wandb_enabled = config.wandb_project is not None
    if wandb_enabled and is_main:
        import wandb

        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config={
                k: str(v) if isinstance(v, list) else v
                for k, v in config.__dict__.items()
            },
        )
        wandb.define_metric("defense_step")
        for prefix in [
            "train/*",
            "val/*",
            "test/*",
            "init_train/*",
            "init_val/*",
            "init_test/*",
        ]:
            wandb.define_metric(prefix, step_metric="defense_step")
        _log(f"W&B: project={config.wandb_project}, run={wandb.run.name}")

    # ── TrainingArguments ─────────────────────────────────────────────────
    if config.save_steps:
        _save_strategy = "steps"
        _save_steps = config.save_steps
    else:
        _save_strategy = "epoch"
        _save_steps = 500

    training_kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        logging_steps=1,
        save_strategy=_save_strategy,
        save_steps=_save_steps,
        save_total_limit=3,
        fp16=config.fp16,
        bf16=config.bf16,
        optim=config.optim,
        lr_scheduler_type=config.lr_scheduler_type,
        report_to="wandb" if wandb_enabled else "none",
        seed=config.seed,
        gradient_checkpointing=config.gradient_checkpointing,
        remove_unused_columns=False,
        ddp_find_unused_parameters=config.ddp_find_unused_parameters,
        ddp_timeout=config.ddp_timeout,
        eval_strategy="epoch" if val_dataset else "no",
    )
    if config.deepspeed:
        training_kwargs["deepspeed"] = config.deepspeed
    if config.fsdp:
        training_kwargs["fsdp"] = config.fsdp
    if config.fsdp_config:
        training_kwargs["fsdp_config"] = config.fsdp_config

    training_args = TrainingArguments(**training_kwargs)

    # ── trainer ───────────────────────────────────────────────────────────
    _log(
        f"\nTurnGate hyperparameters:"
        f"\n  kl_coeff={config.kl_coeff}"
        f"\n  gae_gamma={config.gae_gamma},  gae_lambda={config.gae_lambda}"
        f"\n  use_is_ratio={config.use_is_ratio}"
        f"{'  clip_epsilon=' + str(config.clip_epsilon) if config.use_is_ratio else ''}"
        f"{'  clip_higher=' + str(config.clip_higher) if config.use_is_ratio else ''}"
        f"{'  clip_epsilon_high=' + str(config.clip_epsilon_high) if config.use_is_ratio and config.clip_higher else ''}"
    )

    trainer = TurnGateDefenderTrainer(
        block_token_id=block_token_id,
        pass_token_id=pass_token_id,
        kl_coeff=config.kl_coeff,
        use_is_ratio=config.use_is_ratio,
        clip_epsilon=config.clip_epsilon,
        clip_higher=config.clip_higher,
        clip_epsilon_high=config.clip_epsilon_high,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        processing_class=tokenizer,
    )

    # ── defense evaluation callback ────────────────────────────────────────
    defense_callback = None
    if config.eval_data_dir:
        num_gpus = torch.cuda.device_count()
        eval_gpu_index = (
            config.eval_gpu if config.eval_gpu is not None else max(0, num_gpus - 1)
        )
        _log(f"\nDefense evaluation callback:")
        _log(f"  Eval GPU (cuda device index): {eval_gpu_index}")
        _log(f"  Eval data dir: {config.eval_data_dir}")
        _log(f"  Defense batch size: {config.defense_batch_size}")
        _log(f"  Eval: {eval_freq_str}")

        defense_callback = DefenseEvalCallback(
            trainer=trainer,
            config=config,
            eval_steps=eval_steps,
            steps_per_epoch=steps_per_epoch,
            eval_gpu_index=eval_gpu_index,
            eval_data_dir=config.eval_data_dir,
            output_dir=str(output_dir),
        )
        trainer.add_callback(defense_callback)

    # ── resume init eval ───────────────────────────────────────────────────
    if config.resume_init_eval and defense_callback is not None:
        _log("\n" + "=" * 70)
        _log("[Resume Init Eval] Evaluating resumed model on val + test ...")
        _log("=" * 70)
        defense_callback.run_init_eval(config.resume_model_path, global_step=0)

    # ── train ─────────────────────────────────────────────────────────────
    _log("\n" + "=" * 70)
    _log("Starting offline TurnGate training ...")
    _log("=" * 70 + "\n")

    resume_ckpt = config.resume_model_path if config.resume else None
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # ── save final model ──────────────────────────────────────────────────
    _log("\nSaving final model ...")
    final_path = output_dir / "final_model"
    trainer.save_model(str(final_path))

    if wandb_enabled and is_main:
        import wandb

        wandb.finish()

    _log(f"\nTraining complete. Model saved to: {final_path}")
    return str(final_path)


# ---------------------------------------------------------------------- CLI


def parse_args():
    p = argparse.ArgumentParser(description="Train TurnGate Defender (turn-level GAE)")

    # Model
    p.add_argument("--base-model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--training-type", choices=["lora", "full"], default="lora")
    p.add_argument("--hf-token", type=str, default=None)

    # LoRA
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    # Data
    p.add_argument("--prepared-train-data", type=str, required=True)
    p.add_argument("--prepared-val-data", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="checkpoints/turngate_defender")

    # TurnGate
    p.add_argument("--kl-coeff", type=float, default=0.04)
    p.add_argument("--use-is-ratio", action="store_true")
    p.add_argument("--clip-epsilon", type=float, default=0.2)
    p.add_argument(
        "--clip-higher",
        action="store_true",
        help="DAPO-style clip higher: asymmetric clipping with larger upper bound.",
    )
    p.add_argument(
        "--clip-epsilon-high",
        type=float,
        default=0.28,
        help="Upper clip epsilon (used when --clip-higher). ratio max = 1+ε_high.",
    )
    p.add_argument("--gae-gamma", type=float, default=1.0)
    p.add_argument("--gae-lambda", type=float, default=1.0)

    # Training
    p.add_argument("--num-epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)

    # Distributed
    p.add_argument("--deepspeed", type=str, default=None)
    p.add_argument("--fsdp", type=str, default=None)
    p.add_argument("--fsdp-config", type=str, default=None)
    p.add_argument("--ddp-timeout", type=int, default=7200)
    p.add_argument("--local_rank", type=int, default=-1)

    # Resume
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume-model-path", type=str, default=None)
    p.add_argument("--resume-init-eval", action="store_true")

    # Defense evaluation during training
    p.add_argument("--eval-data-dir", type=str, default=None)
    p.add_argument("--eval-gpu", type=int, default=None)
    p.add_argument("--eval-steps", type=int, default=None)
    p.add_argument("--eval-epochs-only", action="store_true")
    p.add_argument("--defense-batch-size", type=int, default=500)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8)

    # Checkpoint saving
    p.add_argument("--save-steps", type=int, default=None)

    # W&B
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = TurnGateDefenderConfig(
        base_model=args.base_model,
        training_type=args.training_type,
        hf_token=args.hf_token,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        prepared_train_data=args.prepared_train_data,
        prepared_val_data=args.prepared_val_data,
        output_dir=args.output_dir,
        kl_coeff=args.kl_coeff,
        use_is_ratio=args.use_is_ratio,
        clip_epsilon=args.clip_epsilon,
        clip_higher=args.clip_higher,
        clip_epsilon_high=args.clip_epsilon_high,
        gae_gamma=args.gae_gamma,
        gae_lambda=args.gae_lambda,
        num_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        deepspeed=args.deepspeed,
        fsdp=args.fsdp,
        fsdp_config=args.fsdp_config,
        ddp_timeout=args.ddp_timeout,
        resume=args.resume,
        resume_model_path=args.resume_model_path,
        resume_init_eval=args.resume_init_eval,
        eval_data_dir=args.eval_data_dir,
        eval_gpu=args.eval_gpu,
        eval_steps=args.eval_steps,
        eval_epochs_only=args.eval_epochs_only,
        defense_batch_size=args.defense_batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        save_steps=args.save_steps,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )

    train_turngate_defender(config)
