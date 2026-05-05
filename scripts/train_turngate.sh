#!/bin/bash
# Script for training TurnGate Defender (GAE-Augmented Policy Optimization)
# Using the model from anonymous as the starting point

set -e  # Exit on error

# START RUN TIME
START_TIME=$(date +%s)

echo "=========================================="
echo "TurnGate Defender Training (GAE-Augmented RL)"
echo "Started at $(date)"
echo "=========================================="

# Configuration
BASE_MODEL="anonymous_weight_path"
DATASET_DIR="dataset/gpt52-gen_filter"
TurnGate_DATA_DIR="data/turngate_defender"
EVAL_DATA_DIR="data/eval_convs"
OUTPUT_DIR="checkpoints/turngate_optimized"
TRAINING_TYPE="full"  # Using full fine-tuning
WANDB_PROJECT="intent-defender"

# GPU allocation: use 3 GPUs total, last one for vLLM evaluation
export CUDA_VISIBLE_DEVICES=0,1,2
TRAIN_GPUS=2

# Learning Rate: Lower than SFT's 1e-5 (set to 1e-6)
LEARNING_RATE=1e-6

# Reward weights (optimized for turn-level reward shaping)
BLOCK_WEIGHT=3.0
MISS_BLOCK_WEIGHT=3.0
EARLY_BLOCK_WEIGHT=1.5
FALSE_BLOCK_WEIGHT=8.0
PASS_WEIGHT=4.0

# Map to negative penalties for data_preparation
ACC_BLOCK_REWARD=$BLOCK_WEIGHT
CORR_PASS_REWARD=$PASS_WEIGHT
MISS_PENALTY=-$MISS_BLOCK_WEIGHT
FALSE_BLOCK_PENALTY=-$FALSE_BLOCK_WEIGHT
EARLY_BLOCK_BASE=-$EARLY_BLOCK_WEIGHT

# Step 1: Prepare TurnGate training data
echo ""
echo "Step 1/4: Preparing TurnGate training data (GAE-based)..."
echo "=========================================="
mkdir -p $TurnGate_DATA_DIR
python src/turngate_defender/data_preparation.py \
    --harmful-rollout-path $DATASET_DIR/harmful_train.jsonl \
    --benign-rollout-path $DATASET_DIR/benign_train.jsonl \
    --model $BASE_MODEL \
    --output-path $TurnGate_DATA_DIR/turngate_train.jsonl \
    --batch-size 128 \
    --use-vllm \
    --tp-size 2 \
    --gae-gamma 1.0 \
    --gae-lambda 1.0 \
    --accurate-block-reward $ACC_BLOCK_REWARD \
    --correct-pass-reward $CORR_PASS_REWARD \
    --miss-penalty $MISS_PENALTY \
    --false-block-penalty $FALSE_BLOCK_PENALTY \
    --early-block-base $EARLY_BLOCK_BASE

# Step 2: Pre-generate evaluation conversations
echo ""
echo "Step 2/4: Pre-generating evaluation conversations..."
echo "=========================================="
python src/sft_defender/prepare_eval_data.py \
    --dataset-dir $DATASET_DIR \
    --output-dir $EVAL_DATA_DIR \
    --skip-mixed \
    --seed 42

# Step 3: Train the model with TurnGate
echo ""
echo "Step 3/4: Training TurnGate Defender ($TRAINING_TYPE) with GAE-Augmented RL..."
echo "=========================================="

WANDB_ARGS="--wandb-project $WANDB_PROJECT --wandb-run-name turngate-gae-optimized"

torchrun --nproc_per_node=$TRAIN_GPUS src/turngate_defender/train.py \
    --training-type $TRAINING_TYPE \
    --base-model $BASE_MODEL \
    --prepared-train-data $TurnGate_DATA_DIR/turngate_train.jsonl \
    --output-dir $OUTPUT_DIR \
    --batch-size 2 \
    --gradient-accumulation 16 \
    --num-epochs 3 \
    --learning-rate $LEARNING_RATE \
    --max-seq-length 2048 \
    --use-is-ratio \
    --clip-higher \
    --gae-gamma 1.0 \
    --gae-lambda 1.0 \
    --eval-data-dir $EVAL_DATA_DIR \
    --defense-batch-size 500 \
    --seed 42 \
    $WANDB_ARGS

# Step 4: Final standalone evaluation
echo ""
echo "Step 4/4: Final evaluation on test set..."
echo "=========================================="
# output_dir is appended with _full or _lora in config.py
FINAL_MODEL_PATH="${OUTPUT_DIR}_full/final_model"

# Note: TurnGate evaluation uses the same RL evaluator logic
python src/main.py \
    --defender rl \
    --rl-type full \
    --rl-checkpoint $FINAL_MODEL_PATH \
    --dataset-dir $DATASET_DIR \
    --dataset-split test \
    --skip-mixed \
    --batch-size 500

# END RUN TIME
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
H=$((DURATION / 3600))
M=$(((DURATION % 3600) / 60))
S=$((DURATION % 60))

echo ""
echo "=========================================="
echo "TurnGate Training and evaluation completed!"
echo "Ended at $(date)"
printf "Total time: %d:%02d:%02d\n" $H $M $S
echo "=========================================="
