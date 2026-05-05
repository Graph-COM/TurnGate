#!/bin/bash

# Configuration
DATASET_DIR="dataset/gpt52-gen_filter"
BATCH_SIZE=500
HF_TOKEN="${HF_TOKEN:-}"

# Optional limit for testing (set to empty string for full run)
LIMIT=""

# Output directory (main.py handles this, but good to ensure it exists)
mkdir -p results

# We will store results for both valid and test for training-free methods,
# and only for test for trained methods (unless valid is also desired).

# List of training-free methods with their respective model configurations
# Format: "METHOD_NAME | DEFENDER_ARG | MODEL_TYPE | MODEL_ARG | EXTRA_ARGS"
TRAINING_FREE_METHODS=(
    # "Naive_LLM_4B | naive_llm | open | Qwen/Qwen3-4B-Instruct-2507 | "
    "Naive_LLM_5.2 | naive_llm | closed | gpt-5.2 | " # Commented out for now to run only open versions
    # "Sequential_Monitor_4B | sequential_monitor | open | Qwen/Qwen3-4B-Instruct-2507 | "
    # "Prompt_Intent_Anal_4B | intention_analysis | open | Qwen/Qwen3-4B-Instruct-2507 | "
    # "Prompt_Intent_Anal_5.2 | intention_analysis | closed | gpt-5.2 | " # Commented out for now to run only open versions
    # "Llama_Guard_3_8B | llama_guard | open | meta-llama/Llama-Guard-3-8B | "
    # "Qwen_Guard_8B | qwen_guard | open | Qwen/Qwen3Guard-Gen-8B | "
    # "Synthesis_Llama_Guard_4B_8B | synthesis_llama_guard | open |  | --synthesis-model Qwen/Qwen3-4B-Instruct-2507 --guard-model meta-llama/Llama-Guard-3-8B"
    # "Synthesis_Qwen_Guard_4B_8B | synthesis_guard | open |  | --synthesis-model Qwen/Qwen3-4B-Instruct-2507 --guard-model Qwen/Qwen3Guard-Gen-8B"
)

echo "=========================================="
echo "Running Training-Free Baselines (Valid & Test)"
echo "=========================================="

for METHOD_INFO in "${TRAINING_FREE_METHODS[@]}"; do
    IFS="|" read -r NAME DEFENDER M_TYPE MODEL EXTRA <<< "$METHOD_INFO"

    # Trim whitespaces
    NAME=$(echo $NAME | xargs)
    DEFENDER=$(echo $DEFENDER | xargs)
    M_TYPE=$(echo $M_TYPE | xargs)
    MODEL=$(echo $MODEL | xargs)
    EXTRA=$(echo $EXTRA | xargs)

    for SPLIT in "test"; do
        echo -e "\n[Evaluating] $NAME on $SPLIT split..."

        CMD="python src/main.py \
            --defender $DEFENDER \
            --model-type $M_TYPE \
            --dataset-dir $DATASET_DIR \
            --dataset-split $SPLIT \
            --batch-size $BATCH_SIZE \
            --skip-mixed"

        if [ ! -z "$MODEL" ]; then
            CMD="$CMD --model $MODEL"
        fi

        if [ ! -z "$HF_TOKEN" ]; then
            CMD="$CMD --hf-token $HF_TOKEN"
        fi

        if [ ! -z "$LIMIT" ]; then
            CMD="$CMD --limit $LIMIT"
        fi

        if [ ! -z "$EXTRA" ]; then
            CMD="$CMD $EXTRA"
        fi

        echo "Running: $CMD"
        eval $CMD
    done
done

echo -e "\nAll evaluations complete! Results are stored in the results/ directory."
