#!/bin/bash
# Script to evaluate a specific checkpoint (SFT/TurnGate) on the test set.
# This script automatically detects the type (SFT/RL) and format (Full/LoRA).

set -e

# Default values
BASE_MODEL_DEFAULT="Qwen/Qwen3-4B-Instruct-2507"
DATASET_DIR_DEFAULT="dataset/gpt52-gen_filter"
SPLIT_DEFAULT="test"

CHECKPOINT_DIR=$1
BASE_MODEL=${2:-$BASE_MODEL_DEFAULT}
DATASET_DIR=${3:-$DATASET_DIR_DEFAULT}
SPLIT=${4:-$SPLIT_DEFAULT}
TRAIN_TYPE_OVERRIDE=$5
DEFENDER_TYPE_OVERRIDE=$6

if [ -z "$CHECKPOINT_DIR" ]; then
    echo "Usage: $0 <checkpoint_path_or_hf_repo> [base_model] [dataset_dir] [split] [train_type] [defender_type]"
    echo ""
    echo "Examples:"
    echo "  $0 checkpoints/sft_reward_shaping_full/final_model"
    echo "  $0 checkpoints/turngate_optimized_full/final_model"
    echo "  $0 checkpoints/rl_defender_lora/checkpoint-500 meta-llama/Llama-3-8B-Instruct"
    echo "  $0 your-org/your-model meta-llama/Llama-3-8B-Instruct dataset/gpt52-gen_filter test full rl"
    exit 1
fi

# Local path or HF repo ID
CHECKPOINT_SOURCE="huggingface"
CHECKPOINT_REF="$CHECKPOINT_DIR"
if [ -e "$CHECKPOINT_DIR" ]; then
    CHECKPOINT_SOURCE="local"
    CHECKPOINT_REF=$(realpath "$CHECKPOINT_DIR")
fi

# Detect training type (LoRA vs Full)
# Local: detect from adapter_config.json
# HF repo: infer from name, or use override arg 5
TRAIN_TYPE="full"
if [ "$CHECKPOINT_SOURCE" == "local" ]; then
    if [ -f "$CHECKPOINT_REF/adapter_config.json" ]; then
        TRAIN_TYPE="lora"
    fi
else
    if [[ "$CHECKPOINT_REF" == *"lora"* ]] || [[ "$CHECKPOINT_REF" == *"adapter"* ]]; then
        TRAIN_TYPE="lora"
    fi
fi
if [ -n "$TRAIN_TYPE_OVERRIDE" ]; then
    TRAIN_TYPE="$TRAIN_TYPE_OVERRIDE"
fi

# Detect defender type (RL/TurnGate vs SFT)
# We look for 'rl' or 'turngate' in the path/repo ID.
DEFENDER_TYPE="sft"
if [[ "$CHECKPOINT_REF" == *"rl"* ]] || [[ "$CHECKPOINT_REF" == *"turngate"* ]]; then
    DEFENDER_TYPE="rl"
fi
if [ -n "$DEFENDER_TYPE_OVERRIDE" ]; then
    DEFENDER_TYPE="$DEFENDER_TYPE_OVERRIDE"
fi

echo "=========================================="
echo "Checkpoint Evaluation Script"
echo "=========================================="
echo "Checkpoint Source: $CHECKPOINT_SOURCE"
echo "Detected Defender: $DEFENDER_TYPE"
echo "Detected Format:   $TRAIN_TYPE"
echo "Checkpoint Ref:    $CHECKPOINT_REF"
echo "Base Model:        $BASE_MODEL"
echo "Dataset Dir:       $DATASET_DIR"
echo "Split:             $SPLIT"
echo "=========================================="

# Common evaluation arguments
COMMON_ARGS="--dataset-dir $DATASET_DIR --dataset-split $SPLIT --batch-size 1000 --skip-mixed"

if [ "$DEFENDER_TYPE" == "sft" ]; then
    if [ "$TRAIN_TYPE" == "lora" ]; then
        echo "Running SFT LoRA evaluation..."
        python src/main.py \
            --defender sft \
            --sft-type lora \
            --sft-base-model "$BASE_MODEL" \
            --sft-lora-path "$CHECKPOINT_REF" \
            $COMMON_ARGS
    else
        echo "Running SFT Full-FT evaluation..."
        python src/main.py \
            --defender sft \
            --sft-type full \
            --sft-checkpoint "$CHECKPOINT_REF" \
            $COMMON_ARGS
    fi
else
    # RL or TurnGate
    if [ "$TRAIN_TYPE" == "lora" ]; then
        echo "Running RL/TurnGate LoRA evaluation..."
        python src/main.py \
            --defender rl \
            --rl-type lora \
            --rl-base-model "$BASE_MODEL" \
            --rl-lora-path "$CHECKPOINT_REF" \
            $COMMON_ARGS
    else
        echo "Running RL/TurnGate Full-FT evaluation..."
        python src/main.py \
            --defender rl \
            --rl-type full \
            --rl-checkpoint "$CHECKPOINT_REF" \
            $COMMON_ARGS
    fi
fi

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "=========================================="
