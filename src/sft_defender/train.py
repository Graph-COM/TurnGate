"""
Training script for SFT Defender.

Supports both LoRA and full fine-tuning.
Uses next token prediction with cross-entropy loss on "harmful"/"benign" tokens.
"""

import os
import sys
import json
import math
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from torch.utils.data import Dataset

# Import from src.sft_defender (absolute import)
try:
    from src.sft_defender.config import SFTTrainingConfig
    from src.sft_defender.data_preparation import SFTDataset
except ImportError:
    # Fallback to relative import (when run as module)
    from .config import SFTTrainingConfig
    from .data_preparation import SFTDataset


def _log(msg: str):
    """Print only on the main process (rank 0) in distributed training."""
    if int(os.environ.get("LOCAL_RANK", -1)) in (-1, 0):
        print(msg)


# Debug mode: set SFT_DEBUG=1 to enable verbose logging
_SFT_DEBUG = os.environ.get("SFT_DEBUG", "0") == "1"


class SFTTrainer(Trainer):
    """
    Custom Trainer that focuses loss on the completion tokens ("harmful" or "benign").

    We extract logits for only the target tokens and compute cross-entropy loss.
    """

    def __init__(self, harmful_token_id: int, benign_token_id: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.harmful_token_id = harmful_token_id
        self.benign_token_id = benign_token_id
        self._debug_step_count = 0  # Track compute_loss calls for debug logging

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        Custom loss computation.

        We focus on predicting the completion token ("0" or "1")
        rather than the entire sequence.

        Args:
            model: The model being trained
            inputs: Input batch
            return_outputs: Whether to return model outputs along with loss
            num_items_in_batch: Number of items in batch (for newer transformers versions)
        """
        # Extract labels and metadata (pop all custom fields before model(**inputs))
        labels = inputs.pop("labels")
        target_ids = inputs.pop(
            "target_ids", None
        )  # tensor [batch_size]: 1=harmful, 0=benign
        prompt_lengths = inputs.pop("prompt_lengths", None)  # tensor [batch_size]
        inputs.pop("turn_ids", None)  # Only used by RewardShapingSFTTrainer
        inputs.pop("total_turns", None)
        inputs.pop("source_types", None)

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits

        # Compute loss on completion tokens
        # For each sample, we want the model to predict "0" (harmful) or "1" (benign)
        # at the position after the prompt
        batch_size = logits.shape[0]
        # Collect losses for each sample (ensures proper gradient flow)
        losses = []

        # Debug: log first 2 steps in detail
        should_debug = _SFT_DEBUG and self._debug_step_count < 2
        if should_debug:
            _log(f"\n[SFT_DEBUG] === compute_loss step #{self._debug_step_count} ===")
            _log(f"  Batch size: {batch_size}")
            _log(f"  Logits shape: {logits.shape}")
            _log(f"  prompt_lengths: {prompt_lengths}")
            _log(f"  target_ids: {target_ids}")
            _log(f"  harmful_token_id (token '0'): {self.harmful_token_id}")
            _log(f"  benign_token_id  (token '1'): {self.benign_token_id}")

        for i in range(batch_size):
            if prompt_lengths is not None and target_ids is not None:
                # Get the position where completion starts
                prompt_len = prompt_lengths[i].item()

                # Get logits at the last prompt position (predicting first completion token)
                # Shape: [vocab_size]
                position_logits = logits[i, prompt_len - 1, :]

                # Extract logits for "0" (harmful) and "1" (benign) tokens
                # Note: We order them as [benign, harmful] for target indexing (0=benign, 1=harmful)
                target_logits = torch.stack(
                    [
                        position_logits[
                            self.benign_token_id
                        ],  # Token "1" (benign) at index 0
                        position_logits[
                            self.harmful_token_id
                        ],  # Token "0" (harmful) at index 1
                    ]
                )

                # target_ids[i] is already the target index: 0=benign, 1=harmful
                target_tensor = (
                    target_ids[i : i + 1].to(device=target_logits.device).long()
                )

                # Compute cross-entropy loss
                sample_loss = F.cross_entropy(target_logits.unsqueeze(0), target_tensor)
                losses.append(sample_loss)

                # Debug: log per-sample details for first 2 steps
                if should_debug and i < 4:
                    probs = F.softmax(target_logits, dim=0)
                    predicted_label = (
                        "harmful" if target_logits[1] > target_logits[0] else "benign"
                    )
                    true_label = "harmful" if target_tensor.item() == 1 else "benign"
                    _log(
                        f"  Sample[{i}]: prompt_len={prompt_len}, "
                        f"logits=[benign:{target_logits[0].item():.3f}, harmful:{target_logits[1].item():.3f}], "
                        f"probs=[benign:{probs[0].item():.3f}, harmful:{probs[1].item():.3f}], "
                        f"predicted={predicted_label}, true={true_label}, loss={sample_loss.item():.4f}"
                    )

        # Compute mean loss (maintains gradient graph)
        if losses:
            loss = torch.stack(losses).mean()
        else:
            # Fallback if no valid samples (shouldn't happen)
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

        if should_debug:
            _log(f"  Mean loss: {loss.item():.4f}")

        self._debug_step_count += 1

        return (loss, outputs) if return_outputs else loss


class RewardShapingSFTTrainer(SFTTrainer):
    """
    Reward Shaping SFT Trainer: applies dynamic per-sample weights to the loss
    based on the model's current prediction and sample metadata.

    Two modes controlled by `balanced_mode`:

    (a) balanced_mode=False (DEFAULT, original 5-scenario reward shaping):
      - Accurate Block  (+r_tp):        weight = block_weight       (default 1.0)
      - Early Block     (-r_early(t)):  weight = early_block_weight * (1 + gap * gap_penalty_scale)
      - Miss at Closure (-r_miss):      weight = miss_block_weight  (default 3.0)
      - Correct Pass    (+r_bp):        weight = pass_weight        (default 0.3)
      - False Positive  (-r_fp):        weight = false_block_weight (default 1.0)

    (b) balanced_mode=True (simplified class-balancing, ignores prediction):
      - true_label == 1 (should BLOCK): weight = block_weight
      - true_label == 0 (should PASS):  weight = pass_weight
      (Intended usage: set pass_weight / block_weight to the normalized
       inverse of PASS:BLOCK sample counts so both classes contribute equally.)
    """

    def __init__(
        self,
        harmful_token_id: int,
        benign_token_id: int,
        block_weight: float = 1.0,
        miss_block_weight: float = 3.0,
        early_block_weight: float = 0.3,
        gap_penalty_scale: float = 0.5,
        false_block_weight: float = 1.0,
        pass_weight: float = 0.3,
        balanced_mode: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(harmful_token_id, benign_token_id, *args, **kwargs)
        self.block_weight = block_weight
        self.miss_block_weight = miss_block_weight
        self.early_block_weight = early_block_weight
        self.gap_penalty_scale = gap_penalty_scale
        self.false_block_weight = false_block_weight
        self.pass_weight = pass_weight
        self.balanced_mode = balanced_mode

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # Extract labels and metadata
        labels = inputs.pop("labels")
        target_ids = inputs.pop("target_ids", None)
        prompt_lengths = inputs.pop("prompt_lengths", None)
        turn_ids = inputs.pop("turn_ids", None)
        total_turns = inputs.pop("total_turns", None)
        source_types = inputs.pop("source_types", None)

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits

        batch_size = logits.shape[0]
        losses = []

        should_debug = _SFT_DEBUG and self._debug_step_count < 2
        if should_debug:
            _log(f"\n[RS_DEBUG] === compute_loss step #{self._debug_step_count} ===")
            _log(f"  Batch size: {batch_size}")
            _log(f"  target_ids: {target_ids}")
            _log(f"  source_types: {source_types}")
            _log(f"  turn_ids: {turn_ids}")
            _log(f"  total_turns: {total_turns}")

        for i in range(batch_size):
            if prompt_lengths is None or target_ids is None:
                continue

            prompt_len = prompt_lengths[i].item()
            position_logits = logits[i, prompt_len - 1, :]

            target_logits = torch.stack(
                [
                    position_logits[self.benign_token_id],  # index 0: benign
                    position_logits[self.harmful_token_id],  # index 1: harmful
                ]
            )

            target_tensor = target_ids[i : i + 1].to(device=target_logits.device).long()

            # Unweighted cross-entropy loss
            sample_loss = F.cross_entropy(
                target_logits.unsqueeze(0),
                target_tensor,
            )

            # Determine model prediction and compute dynamic weight
            predicted = torch.argmax(target_logits).item()  # 0=benign, 1=harmful
            true_label = target_tensor.item()  # 0=benign, 1=harmful
            is_benign_conv = (
                (source_types[i].item() == 1) if source_types is not None else False
            )

            if self.balanced_mode:
                # Balanced class-weighting: ignore prediction, only look at true_label
                if true_label == 1:
                    weight = self.block_weight
                    scenario = "balanced_block(label=1)"
                else:
                    weight = self.pass_weight
                    scenario = "balanced_pass(label=0)"
            elif is_benign_conv:
                # Benign conversation: all labels are benign (true_label=0)
                if predicted == 0:
                    weight = self.pass_weight  # Correct Pass
                    scenario = "correct_pass(benign)"
                else:
                    weight = self.false_block_weight  # False Positive
                    scenario = "false_positive"
            else:
                # Harmful conversation
                if true_label == 0 and predicted == 0:
                    weight = self.pass_weight  # Correct Pass
                    scenario = "correct_pass(harmful)"
                elif true_label == 1 and predicted == 1:
                    weight = self.block_weight  # Accurate Block
                    scenario = "accurate_block"
                elif true_label == 0 and predicted == 1:
                    # Early Block: gap = (total_turns - 1) - turn_id
                    if turn_ids is not None and total_turns is not None:
                        raw_gap = (total_turns[i].item() - 1) - turn_ids[i].item()
                        if raw_gap < 0:
                            _log("Warning: early block gap is negative")
                        gap = max(0, raw_gap)
                    else:
                        gap = 1
                    weight = self.early_block_weight * (
                        1.0 + gap * self.gap_penalty_scale
                    )
                    scenario = f"early_block(gap={gap})"
                else:  # true_label == 1 and predicted == 0
                    weight = self.miss_block_weight  # Miss at Closure
                    scenario = "miss_block"

            weighted_loss = weight * sample_loss
            losses.append(weighted_loss)

            if should_debug and i < 4:
                probs = F.softmax(target_logits, dim=0)
                _log(
                    f"  Sample[{i}]: scenario={scenario}, weight={weight:.1f}, "
                    f"probs=[benign:{probs[0].item():.3f}, harmful:{probs[1].item():.3f}], "
                    f"loss={sample_loss.item():.4f}, weighted={weighted_loss.item():.4f}"
                )

        if losses:
            loss = torch.stack(losses).mean()
        else:
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

        if should_debug:
            _log(f"  Mean weighted loss: {loss.item():.4f}")

        self._debug_step_count += 1
        return (loss, outputs) if return_outputs else loss


class SFTDatasetForTraining(Dataset):
    """
    Dataset wrapper that properly formats data for the custom trainer.
    """

    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 2048):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

        self.tokenizer = tokenizer
        self.max_length = max_length
        self._debug_logged_count = 0  # Track how many samples we've debug-logged

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prompt = sample["prompt"]
        completion = sample["completion"]

        # Apply chat template to match inference format (VLLMClient._format_prompt)
        # Without this, training sees raw text but inference sees chat-formatted text
        # Disable thinking mode (Qwen3) to ensure model outputs "0"/"1" directly
        try:
            messages = [{"role": "user", "content": prompt}]
            try:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                # Fallback for tokenizers that don't support enable_thinking
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        except Exception:
            formatted_prompt = prompt

        # Tokenize formatted prompt and full text separately to know where completion starts
        # Use add_special_tokens=False because chat template already includes special tokens
        prompt_encoding = self.tokenizer(
            formatted_prompt,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )

        full_text = formatted_prompt + completion
        full_encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            add_special_tokens=False,
        )

        # Convert string label to target index for cross-entropy:
        # completion "0" (harmful) → target=1, completion "1" (benign) → target=0
        # This matches target_logits ordering: [benign_logit, harmful_logit]
        label_id = 1 if completion == "0" else 0

        prompt_len = len(prompt_encoding["input_ids"])
        full_len = len(full_encoding["input_ids"])

        # Debug: log first 3 samples to verify tokenization and chat template
        if _SFT_DEBUG and self._debug_logged_count < 3:
            self._debug_logged_count += 1
            _log(f"\n[SFT_DEBUG] === Training Sample #{idx} ===")
            _log(f"  Completion: {repr(completion)} → target_id: {label_id}")
            _log(f"  Has <think> in formatted prompt: {'<think>' in formatted_prompt}")
            _log(
                f"  Formatted prompt tail (last 120 chars): {repr(formatted_prompt[-120:])}"
            )
            _log(f"  Full text tail (last 30 chars): {repr(full_text[-30:])}")
            _log(f"  Prompt token length: {prompt_len}")
            _log(
                f"  Full token length (before padding): {sum(full_encoding['attention_mask'])}"
            )
            _log(f"  Full token length (with padding): {full_len}")
            # Verify: token at prompt_len position should be the completion token
            if prompt_len < full_len:
                completion_token_id = full_encoding["input_ids"][prompt_len]
                completion_token_str = self.tokenizer.decode([completion_token_id])
                _log(
                    f"  Token at prompt_len position [{prompt_len}]: id={completion_token_id}, decoded={repr(completion_token_str)}"
                )
                _log(f"  Expected completion token: {repr(completion)}")
                if completion_token_str.strip() != completion:
                    _log(
                        f"  ⚠ WARNING: Completion token mismatch! Training may learn wrong signal!"
                    )
            else:
                _log(
                    f"  ⚠ WARNING: prompt_len ({prompt_len}) >= full_len ({full_len}), prompt was truncated!"
                )
            _log(f"  Padding side: {self.tokenizer.padding_side}")

        # Extract metadata for reward shaping
        metadata = sample.get("metadata", {})
        turn_id = metadata.get("turn_id", 0)
        total_turns = metadata.get("total_turns", 1)
        # source_type: 0=harmful conversation, 1=benign conversation
        source_type_int = (
            0 if metadata.get("source_type", "harmful") == "harmful" else 1
        )

        return {
            "input_ids": torch.tensor(full_encoding["input_ids"]),
            "attention_mask": torch.tensor(full_encoding["attention_mask"]),
            "labels": torch.tensor(
                full_encoding["input_ids"]
            ),  # For standard LM loss (if needed)
            "prompt_lengths": prompt_len,
            "target_ids": label_id,  # int: 1=harmful, 0=benign (stacked to tensor by collator)
            "turn_ids": turn_id,  # int: 0-indexed turn position
            "total_turns": total_turns,  # int: total user turns in conversation
            "source_types": source_type_int,  # int: 0=harmful conv, 1=benign conv
        }


def load_model_and_tokenizer(config: SFTTrainingConfig):
    """Load base model and tokenizer.

    For resume with full FT, loads model from resume_model_path.
    For resume with LoRA, still loads base model here (adapter loaded in setup_lora).
    """
    # Determine model path: resume full FT loads from checkpoint, otherwise from base_model
    if config.resume and config.resume_model_path and config.training_type == "full":
        model_path = config.resume_model_path
        _log(f"[Resume] Loading model from checkpoint: {model_path}")
    else:
        model_path = config.base_model
        _log(f"Loading model and tokenizer from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        token=config.hf_token,
    )

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Ensure right padding for training (left padding breaks prompt_len indexing in compute_loss)
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=(
            torch.bfloat16
            if config.bf16
            else torch.float16 if config.fp16 else torch.float32
        ),
        token=config.hf_token,
    )

    # Enable gradient checkpointing if specified
    # Skip if DeepSpeed is active — Trainer handles it via TrainingArguments
    if config.gradient_checkpointing and not config.deepspeed:
        model.gradient_checkpointing_enable()

    return model, tokenizer


def setup_lora(model, config: SFTTrainingConfig):
    """Setup LoRA for parameter-efficient fine-tuning.

    For resume, loads an existing adapter from resume_model_path.
    Otherwise, creates a fresh LoRA adapter.
    """
    if config.resume and config.resume_model_path:
        _log(f"[Resume] Loading LoRA adapter from {config.resume_model_path}...")
        model = PeftModel.from_pretrained(model, config.resume_model_path)
        # Make adapter trainable again (PeftModel.from_pretrained loads in eval mode)
        model.train()
    else:
        _log("Setting up LoRA...")
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias=config.lora_bias,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()
    return model


def train_sft_defender(config: SFTTrainingConfig) -> str:
    """
    Main training function.

    Args:
        config: SFTTrainingConfig instance

    Returns:
        Path to the saved model checkpoint
    """
    # Set random seed
    torch.manual_seed(config.seed)

    # Distributed training setup
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_main_process = local_rank in (-1, 0)

    # Create output directory and save config (mkdir is idempotent with exist_ok=True)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if is_main_process:
        with open(output_dir / "training_config.json", "w") as f:
            json.dump(config.__dict__, f, indent=2, default=str)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)

    # Setup LoRA if specified
    if config.training_type == "lora":
        model = setup_lora(model, config)
    else:
        _log("Using full parameter fine-tuning...")
        _log(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Get token IDs for "0" (harmful) and "1" (benign)
    # Using digits instead of words ensures single-token encoding
    harmful_tokens = tokenizer.encode("0", add_special_tokens=False)
    benign_tokens = tokenizer.encode("1", add_special_tokens=False)

    # Validate that these are single tokens
    if len(harmful_tokens) != 1:
        raise ValueError(
            f"'0' must be a single token, but tokenizes to {len(harmful_tokens)} tokens: {harmful_tokens}"
        )
    if len(benign_tokens) != 1:
        raise ValueError(
            f"'1' must be a single token, but tokenizes to {len(benign_tokens)} tokens: {benign_tokens}"
        )

    harmful_token_id = harmful_tokens[0]  # Token ID for "0"
    benign_token_id = benign_tokens[0]  # Token ID for "1"
    _log(f"Token IDs - 0 (harmful): {harmful_token_id}, 1 (benign): {benign_token_id}")

    # Load dataset
    _log(f"Loading training data from {config.train_data_path}...")
    train_dataset = SFTDatasetForTraining(
        config.train_data_path,
        tokenizer,
        max_length=config.max_seq_length,
    )
    _log(f"Training samples: {len(train_dataset)}")

    # Compute steps_per_epoch and eval_steps for defense evaluation
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    effective_batch = (
        config.per_device_train_batch_size
        * config.gradient_accumulation_steps
        * world_size
    )
    steps_per_epoch = math.ceil(len(train_dataset) / effective_batch)
    eval_steps = config.eval_steps or max(1, steps_per_epoch // 10)
    _log(f"Steps per epoch: {steps_per_epoch}, eval_steps: {eval_steps}")

    # Initialize W&B (before TrainingArguments so HF Trainer picks it up)
    wandb_enabled = config.wandb_project is not None
    if wandb_enabled and is_main_process:
        import wandb

        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config={
                k: str(v) if isinstance(v, list) else v
                for k, v in config.__dict__.items()
            },
        )
        # Define custom step metric for defense eval so it doesn't conflict
        # with HF Trainer's step counter (which causes "step less than current" warnings)
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
        _log(f"W&B initialized: project={config.wandb_project}, run={wandb.run.name}")

    # Training arguments
    training_args_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": config.num_epochs,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "max_grad_norm": config.max_grad_norm,
        "logging_steps": 1,  # Log train loss every step for W&B
        "save_strategy": "no",  # We save manually via DefenseEvalCallback
        "fp16": config.fp16,
        "bf16": config.bf16,
        "optim": config.optim,
        "lr_scheduler_type": config.lr_scheduler_type,
        "report_to": "wandb" if wandb_enabled else "none",
        "seed": config.seed,
        "gradient_checkpointing": config.gradient_checkpointing,
        "remove_unused_columns": False,  # Preserve custom fields (target_ids, prompt_lengths)
        "ddp_find_unused_parameters": config.ddp_find_unused_parameters,
        "ddp_timeout": config.ddp_timeout,
    }

    # Distributed training configuration
    if config.deepspeed:
        training_args_kwargs["deepspeed"] = config.deepspeed
    if config.fsdp:
        training_args_kwargs["fsdp"] = config.fsdp
    if config.fsdp_config:
        training_args_kwargs["fsdp_config"] = config.fsdp_config

    training_args = TrainingArguments(**training_args_kwargs)

    # Data collator - use default to avoid conflicts with custom loss
    data_collator = default_data_collator

    # Initialize custom trainer
    if config.reward_shaping:
        _log(f"\nUsing Reward Shaping SFT Trainer:")
        _log(f"  balanced_mode={config.balanced_mode}")
        _log(f"  block_weight={config.block_weight}")
        _log(f"  pass_weight={config.pass_weight}")
        if not config.balanced_mode:
            _log(f"  miss_block_weight={config.miss_block_weight}")
            _log(f"  early_block_weight={config.early_block_weight}")
            _log(f"  gap_penalty_scale={config.gap_penalty_scale}")
            _log(f"  false_block_weight={config.false_block_weight}")
        trainer = RewardShapingSFTTrainer(
            harmful_token_id=harmful_token_id,
            benign_token_id=benign_token_id,
            block_weight=config.block_weight,
            miss_block_weight=config.miss_block_weight,
            early_block_weight=config.early_block_weight,
            gap_penalty_scale=config.gap_penalty_scale,
            false_block_weight=config.false_block_weight,
            pass_weight=config.pass_weight,
            balanced_mode=config.balanced_mode,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            processing_class=tokenizer,
        )
    else:
        trainer = SFTTrainer(
            harmful_token_id=harmful_token_id,
            benign_token_id=benign_token_id,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            processing_class=tokenizer,
        )

    # Register defense evaluation callback on ALL ranks if eval_data_dir is provided.
    # The callback is DDP-safe: all ranks participate in checkpoint saving and barriers,
    # only rank 0 runs the vLLM subprocess and W&B logging.
    defense_callback = None
    if config.eval_data_dir:
        from src.sft_defender.eval_callback import DefenseEvalCallback

        # Determine eval GPU: default to last visible GPU
        num_gpus = torch.cuda.device_count()
        eval_gpu_index = (
            config.eval_gpu if config.eval_gpu is not None else (num_gpus - 1)
        )
        _log(f"\nDefense evaluation setup:")
        _log(f"  Eval GPU (cuda device index): {eval_gpu_index}")
        _log(f"  Eval data dir: {config.eval_data_dir}")
        _log(f"  Defense batch size: {config.defense_batch_size}")
        _log(f"  Eval every {eval_steps} steps + each epoch end")

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

    # Resume init-eval: evaluate the resumed model before training starts
    if config.resume_init_eval and defense_callback is not None:
        _log("\n" + "=" * 70)
        _log("[Resume Init Eval] Evaluating resumed model on val + test...")
        _log("=" * 70)
        defense_callback.run_init_eval(config.resume_model_path, global_step=0)

    # Train
    _log("\n" + "=" * 70)
    _log("Starting training...")
    _log("=" * 70 + "\n")

    if config.resume and config.resume_model_path:
        _log(f"Resuming training state from: {config.resume_model_path}")
        trainer.train(resume_from_checkpoint=config.resume_model_path)
    else:
        trainer.train()

    # Save final model (trainer.save_model is DDP/DeepSpeed-safe, also saves tokenizer)
    _log("\nSaving final model...")
    final_model_path = output_dir / "final_model"
    trainer.save_model(str(final_model_path))

    # Finish W&B
    if wandb_enabled and is_main_process:
        import wandb

        wandb.finish()

    _log(f"\nTraining completed! Model saved to: {final_model_path}")

    return str(final_model_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train SFT Defender")

    # Training type
    parser.add_argument(
        "--training-type",
        type=str,
        choices=["lora", "full"],
        default="lora",
        help="Training type: lora or full fine-tuning",
    )

    # Model
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Base model name or path",
    )

    # LoRA parameters
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")

    # Training parameters
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/sft_defender",
        help="Output directory",
    )
    parser.add_argument(
        "--train-data", type=str, required=True, help="Training data JSONL path"
    )
    parser.add_argument(
        "--val-data", type=str, default=None, help="Validation data JSONL path"
    )
    parser.add_argument(
        "--num-epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Per-device batch size"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=2048, help="Maximum sequence length"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face token")

    # Distributed training
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="Path to DeepSpeed config JSON (e.g., src/sft_defender/ds_zero2.json)",
    )
    parser.add_argument(
        "--fsdp",
        type=str,
        default=None,
        help="FSDP strategy: 'full_shard', 'shard_grad_op', 'offload'",
    )
    parser.add_argument(
        "--fsdp-config",
        type=str,
        default=None,
        help="Path to FSDP config JSON",
    )
    parser.add_argument(
        "--ddp-timeout",
        type=int,
        default=7200,
        help="DDP timeout in seconds (default: 7200 = 2h, covers long vLLM eval)",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (set automatically by torchrun)",
    )

    # Resume training
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from a saved checkpoint",
    )
    parser.add_argument(
        "--resume-model-path",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume from (e.g., checkpoints/sft_defender_full/epoch_2)",
    )
    parser.add_argument(
        "--resume-init-eval",
        action="store_true",
        help="Run val+test defense eval on the resumed model before training starts",
    )

    # Reward Shaping
    parser.add_argument(
        "--balanced-mode",
        action="store_true",
        default=False,
        help="Use balanced class-weighting mode (only block_weight/pass_weight, ignores prediction). "
        "Intended for balancing pass vs block turn counts.",
    )
    parser.add_argument(
        "--reward-shaping",
        action="store_true",
        help="Enable reward shaping weighted loss",
    )
    parser.add_argument(
        "--block-weight",
        type=float,
        default=1.0,
        help="Accurate Block weight (r_tp, baseline)",
    )
    parser.add_argument(
        "--miss-block-weight",
        type=float,
        default=3.0,
        help="Miss at Closure weight (r_miss)",
    )
    parser.add_argument(
        "--early-block-weight",
        type=float,
        default=0.3,
        help="Early Block base weight (r_early)",
    )
    parser.add_argument(
        "--gap-penalty-scale",
        type=float,
        default=0.5,
        help="Early Block gap scaling factor",
    )
    parser.add_argument(
        "--false-block-weight",
        type=float,
        default=1.0,
        help="False Positive weight (r_fp)",
    )
    parser.add_argument(
        "--pass-weight", type=float, default=0.3, help="Correct Pass weight (r_bp)"
    )

    # W&B
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project name (enables W&B logging)",
    )
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")

    # Defense evaluation during training
    parser.add_argument(
        "--eval-data-dir",
        type=str,
        default=None,
        help="Dir with pre-generated eval conversations (pkl files). Enables defense evaluation.",
    )
    parser.add_argument(
        "--eval-gpu",
        type=int,
        default=None,
        help="GPU index for vLLM evaluation (default: last visible GPU)",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=None,
        help="Defense eval every N steps (default: steps_per_epoch // 10)",
    )
    parser.add_argument(
        "--defense-batch-size",
        type=int,
        default=500,
        help="Batch size for Evaluator.evaluate() during defense eval",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="vLLM GPU memory utilization for evaluation",
    )

    args = parser.parse_args()

    # Create config
    config = SFTTrainingConfig(
        training_type=args.training_type,
        base_model=args.base_model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        output_dir=args.output_dir,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        num_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        hf_token=args.hf_token,
        deepspeed=args.deepspeed,
        fsdp=args.fsdp,
        fsdp_config=args.fsdp_config,
        ddp_timeout=args.ddp_timeout,
        resume=args.resume,
        resume_model_path=args.resume_model_path,
        resume_init_eval=args.resume_init_eval,
        reward_shaping=args.reward_shaping,
        balanced_mode=args.balanced_mode,
        block_weight=args.block_weight,
        miss_block_weight=args.miss_block_weight,
        early_block_weight=args.early_block_weight,
        gap_penalty_scale=args.gap_penalty_scale,
        false_block_weight=args.false_block_weight,
        pass_weight=args.pass_weight,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        eval_data_dir=args.eval_data_dir,
        eval_gpu=args.eval_gpu,
        eval_steps=args.eval_steps,
        defense_batch_size=args.defense_batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Train
    train_sft_defender(config)
