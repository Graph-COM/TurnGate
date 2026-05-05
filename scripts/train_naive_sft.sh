#!/bin/bash
# Script for training Naive SFT Defender
# Uniform per-turn cross-entropy loss (no class reweighting).

set -e

echo "=========================================="
echo "Naive SFT Defender Training"
echo "=========================================="

# Configuration
BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"
DATASET_DIR="dataset/gpt52-gen_filter"
SFT_DATA_DIR="data/sft"
EVAL_DATA_DIR="data/eval_convs"
OUTPUT_DIR="checkpoints/naive_sft"
TRAINING_TYPE="full"
WANDB_PROJECT="intent-defender"

# GPU allocation: use 3 GPUs total, last one for vLLM evaluation
export CUDA_VISIBLE_DEVICES=0,1,2
TRAIN_GPUS=2

# Step 1: Prepare SFT training data
echo ""
echo "Step 1/4: Preparing SFT training data..."
echo "=========================================="
python src/sft_defender/data_preparation.py \
    --dataset-dir $DATASET_DIR \
    --output-dir $SFT_DATA_DIR \
    --seed 42

# Step 2: Pre-generate evaluation conversations
echo ""
echo "Step 2/4: Pre-generating evaluation conversations..."
echo "=========================================="
python src/sft_defender/prepare_eval_data.py \
    --dataset-dir $DATASET_DIR \
    --output-dir $EVAL_DATA_DIR \
    --skip-mixed \
    --seed 42

# Step 3: Train the model with uniform loss
echo ""
echo "Step 3/4: Training Naive SFT Defender ($TRAINING_TYPE)..."
echo "=========================================="

WANDB_ARGS="--wandb-project $WANDB_PROJECT --wandb-run-name naive-sft"

torchrun --nproc_per_node=$TRAIN_GPUS --master_port=29500 src/sft_defender/train.py \
    --training-type $TRAINING_TYPE \
    --base-model $BASE_MODEL \
    --train-data $SFT_DATA_DIR/train.jsonl \
    --output-dir $OUTPUT_DIR \
    --batch-size 2 \
    --gradient-accumulation-steps 16 \
    --num-epochs 6 \
    --learning-rate 1e-5 \
    --max-seq-length 2048 \
    --eval-data-dir $EVAL_DATA_DIR \
    --defense-batch-size 500 \
    --seed 42 \
    $WANDB_ARGS

# Step 4: Final standalone evaluation
echo ""
echo "Step 4/4: Final evaluation on test set..."
echo "=========================================="
python src/main.py \
    --defender sft \
    --sft-type full \
    --sft-checkpoint ${OUTPUT_DIR}_full/final_model \
    --dataset-dir $DATASET_DIR \
    --dataset-split test \
    --skip-mixed \
    --batch-size 500

echo ""
echo "=========================================="
echo "Naive SFT training and evaluation completed!"
echo "=========================================="
