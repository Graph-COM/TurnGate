# SFT Defender

Supervised Fine-Tuning (SFT) based defense method for Intent Defender framework.

## Overview

SFT Defender uses a fine-tuned language model to classify user queries as harmful or benign in multi-turn conversations. In a harmful conversation with k turns:
- **Early turns (1 to k-1)**: Labeled as "benign" (conversation hasn't reached harmful stage)
- **Final turn (k)**: Labeled as "harmful" (harmful intent is revealed)

This teaches the model to **not over-react** to early innocent questions, but recognize when the conversation crosses into harmful territory.

> **Note**: All commands below should be run from the project root directory (`Intent-Defender/`).
>
> The training script automatically appends `_lora` or `_full` to `--output-dir` based on `--training-type`. For example, `--output-dir checkpoints/sft_defender` produces `checkpoints/sft_defender_lora/` or `checkpoints/sft_defender_full/`.

---

## Step 1: Prepare Training Data

Uses the pre-split dataset from `dataset/gpt52-gen/` (containing `benign_{train|valid|test}.jsonl` and `harmful_{train|valid|test}.jsonl`). The same data is used for both Naive SFT and Reward Shaping SFT.

```bash
python src/sft_defender/data_preparation.py \
    --dataset-dir dataset/gpt52-gen \
    --output-dir data/sft \
    --seed 42
```

This processes all three splits (train/valid/test) and generates per-split output files:
- `data/sft/harmful_{train|valid|test}.jsonl` — SFT samples from harmful conversations
- `data/sft/benign_{train|valid|test}.jsonl` — SFT samples from benign conversations
- `data/sft/{train|valid|test}.jsonl` — combined and shuffled

For each conversation with k turns, this generates k training samples. In harmful conversations, early turns are labeled benign and the final turn is labeled harmful. In benign conversations, all turns are labeled benign. Each sample includes `source_type` metadata (used by Reward Shaping).

---

## Step 2: Train

### LoRA

#### Single GPU

```bash
python src/sft_defender/train.py \
    --training-type lora \
    --base-model Qwen/Qwen3-4B-Instruct-2507 \
    --train-data data/sft/train.jsonl \
    --val-data data/sft/valid.jsonl \
    --output-dir checkpoints/sft_defender \
    --lora-r 8 \
    --lora-alpha 16 \
    --batch-size 4 \
    --gradient-accumulation-steps 8 \
    --num-epochs 3 \
    --learning-rate 2e-5
```

#### Multi-GPU (DDP)

```bash
torchrun --nproc_per_node=NUM_GPUS src/sft_defender/train.py \
    --training-type lora \
    --base-model Qwen/Qwen3-4B-Instruct-2507 \
    --train-data data/sft/train.jsonl \
    --val-data data/sft/valid.jsonl \
    --output-dir checkpoints/sft_defender \
    --lora-r 8 \
    --lora-alpha 16 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --num-epochs 3 \
    --learning-rate 2e-5
```

### Full Fine-Tuning

#### Single GPU

```bash
python src/sft_defender/train.py \
    --training-type full \
    --base-model Qwen/Qwen3-4B-Instruct-2507 \
    --train-data data/sft/train.jsonl \
    --val-data data/sft/valid.jsonl \
    --output-dir checkpoints/sft_defender \
    --batch-size 2 \
    --gradient-accumulation-steps 16 \
    --num-epochs 3 \
    --learning-rate 1e-5
```

#### Multi-GPU (DeepSpeed ZeRO-2)

Shards optimizer states and gradients across GPUs, reducing memory ~3x:

```bash
torchrun --nproc_per_node=2 src/sft_defender/train.py \
    --training-type full \
    --base-model Qwen/Qwen3-4B-Instruct-2507 \
    --train-data data/sft/train.jsonl \
    --val-data data/sft/valid.jsonl \
    --output-dir checkpoints/sft_defender \
    --deepspeed src/sft_defender/ds_zero2.json \
    --batch-size 2 \
    --gradient-accumulation-steps 4 \
    --num-epochs 3 \
    --learning-rate 1e-5 \
  --wandb-project       intent-defender \
  --wandb-run-name      sft-4b-filter
```

---

## Step 3: Evaluate

### LoRA

```bash
python src/main.py \
    --defender sft \
    --sft-type lora \
    --sft-base-model Qwen/Qwen3-4B-Instruct-2507 \
    --sft-lora-path checkpoints/sft_defender_lora/final_model \
    --dataset-dir dataset/gpt52-gen \
    --dataset-split test \
    --mixing-strategy random \
    --min-insert-benign 0 \
    --max-insert-benign 2 \
    --min-insert-harmful 0 \
    --max-insert-harmful 3 \
    --batch-size 500
```

### Full Fine-Tuning

```bash
python src/main.py \
    --defender sft \
    --sft-type full \
    --sft-checkpoint checkpoints/sft_defender_full/final_model \
    --dataset-dir dataset/gpt52-gen \
    --dataset-split test \
    --mixing-strategy random \
    --min-insert-benign 0 \
    --max-insert-benign 2 \
    --min-insert-harmful 0 \
    --max-insert-harmful 3 \
    --batch-size 500
```
