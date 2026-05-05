"""
Standalone defense evaluation script.

Launched as a subprocess during training to evaluate defense performance
on a dedicated GPU via vLLM. This script:
1. Loads a saved checkpoint as a vLLM model
2. Loads pre-generated conversations from a pickle file
3. Runs Evaluator.evaluate() with SFTDefender
4. Writes results to a JSON file

Usage (called by eval_callback.py, not manually):
    CUDA_VISIBLE_DEVICES=2 python src/sft_defender/eval_defense.py \
        --checkpoint checkpoints/sft_defender_full/tmp_eval \
        --eval-conversations data/eval_convs/val_conversations.pkl \
        --defense-batch-size 500 \
        --output /tmp/eval_result.json \
        --training-type full \
        --base-model Qwen/Qwen3-4B-Instruct-2507 \
        --gpu-memory-utilization 0.8
"""

import os
import sys
import json
import pickle
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


def flatten_metrics(organized: dict, prefix: str) -> dict:
    """Flatten the nested Evaluator metrics dict into W&B-compatible flat keys.

    Example:
        {"CONVERSATION_LEVEL": {"harmful": {"block_rate": 0.8}}}
        -> {"prefix/harmful_block_rate": 0.8}
    """
    flat = {}
    for level_key in ("CONVERSATION_LEVEL", "TURN_LEVEL"):
        level_data = organized.get(level_key, {})
        for category, metrics in level_data.items():
            for metric_name, value in metrics.items():
                if value is not None:
                    flat[f"{prefix}/{category}_{metric_name}"] = value
    # OVERALL metrics (F1, precision, recall, etc.)
    for metric_name, value in organized.get("OVERALL", {}).items():
        if value is not None:
            flat[f"{prefix}/overall_{metric_name}"] = value
    return flat


def run_evaluation(
    checkpoint_path: str,
    eval_conversations_path: str,
    defense_batch_size: int = 500,
    output_path: str = "/tmp/eval_result.json",
    training_type: str = "full",
    base_model: str = None,
    hf_token: str = None,
    gpu_memory_utilization: float = 0.8,
    enforce_eager: bool = True,
    mda_mode: bool = False,
):
    """Run defense evaluation using vLLM and Evaluator."""
    from src.intention_analysis import VLLMClient
    from src.defender import SFTDefender
    from src.evaluator import Evaluator

    # Load pre-generated conversations
    print(f"Loading eval conversations from {eval_conversations_path}...")
    with open(eval_conversations_path, "rb") as f:
        conversations = pickle.load(f)
    print(f"  Loaded {len(conversations)} conversations")

    # Initialize vLLM client
    if training_type == "lora":
        # LoRA: base model + adapter
        model_path = base_model
        print(
            f"Loading vLLM with base model {model_path} + LoRA adapter {checkpoint_path}..."
        )
        llm_client = VLLMClient(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            hf_token=hf_token,
            enforce_eager=enforce_eager,
            enable_lora=True,
            lora_modules={"sft": checkpoint_path},
            max_lora_rank=64,
        )
    else:
        # Full fine-tuning: checkpoint is the full model
        model_path = checkpoint_path
        print(f"Loading vLLM with checkpoint {model_path}...")
        llm_client = VLLMClient(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            hf_token=hf_token,
            enforce_eager=enforce_eager,
        )

    # Create defender and evaluator
    defender = SFTDefender(llm_client=llm_client, mda_mode=mda_mode)
    evaluator = Evaluator(defender)

    # Run evaluation
    print(f"Running defense evaluation (batch_size={defense_batch_size})...")
    results = evaluator.evaluate(
        conversations,
        show_progress=True,
        desc="Defense Eval",
        batch_size=defense_batch_size,
    )

    # Write results
    print(f"Writing results to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("Defense evaluation complete.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Defense evaluation subprocess")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument(
        "--eval-conversations",
        type=str,
        required=True,
        help="Path to pickled conversations",
    )
    parser.add_argument("--defense-batch-size", type=int, default=500)
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    parser.add_argument(
        "--training-type",
        type=str,
        default="full",
        choices=["lora", "full"],
    )
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=True,
        help="Use eager mode for vLLM (avoids CUDA graph issues in subprocess)",
    )
    parser.add_argument(
        "--mda-mode",
        action="store_true",
        default=False,
        help="Use MDA input-only prompts (no assistant response)",
    )

    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=args.checkpoint,
        eval_conversations_path=args.eval_conversations,
        defense_batch_size=args.defense_batch_size,
        output_path=args.output,
        training_type=args.training_type,
        base_model=args.base_model,
        hf_token=args.hf_token,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        mda_mode=args.mda_mode,
    )
