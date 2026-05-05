import argparse
import os
import sys
import json
from datetime import datetime

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import ConversationMixer
from src.dataset_utils import resolve_dataset_paths
from src.defender import (
    DummyDefender,
    RandomDefender,
    NaiveLLMDefender,
    IntentionAnalysisDefender,
    SFTDefender,
    RLDefender,
)
from src.guard_defender import LlamaGuardDefender, QwenGuardDefender
from src.synthesis_defender import SynthesisGuardDefender, SynthesisLlamaGuardDefender
from src.evaluator import Evaluator, compute_fci_metrics


def print_results(results, title="Results"):
    """Print evaluation results in a clean hierarchical format.

    Args:
        results: The organized metrics dict from evaluator
        title: Title for the output section
    """
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print("=" * 70)
    print(json.dumps(results, indent=2))
    print("=" * 70 + "\n")
    return


def parse_args():
    parser = argparse.ArgumentParser(description="Intent Defender Framework evaluation")
    parser.add_argument(
        "--defender",
        choices=[
            "dummy",
            "random",
            "naive_llm",
            "intention_analysis",
            "sft",
            "sequential_monitor",
            "rl",
            "llama_guard",
            "qwen_guard",
            "synthesis_guard",
            "synthesis_llama_guard",
        ],
        default=os.environ.get("DEFENDER", "dummy"),
        help=(
            "Defender to use: dummy (keyword), random, naive_llm (direct LLM classify), "
            "intention_analysis (Stage1+Stage2), sft (supervised fine-tuned), "
            "sequential_monitor (Lightweight Sequential Monitor), "
            "rl (offline GRPO-trained, see src/rl_defender/), "
            "llama_guard (Llama Guard 3), qwen_guard (Qwen3Guard), "
            "synthesis_guard (Qwen3 4B synthesis + Qwen Guard), "
            "synthesis_llama_guard (Qwen3 4B synthesis + Llama Guard 3)"
        ),
    )
    parser.add_argument(
        "--monitor-prompt-type",
        type=str,
        default="BINARY_INTENTION_EVAL_ORIGINAL",
        help="Prompt type for sequential_monitor defender. See PROMPT_DICT in src/subtask_eval/prompts.py",
    )
    parser.add_argument(
        "--monitor-threshold",
        type=float,
        default=0.5,
        help="Threshold for sequential_monitor defender",
    )
    parser.add_argument(
        "--block-prob",
        type=float,
        default=0.1,
        help="Block probability for random defender",
    )
    parser.add_argument(
        "--model-type",
        choices=["open", "closed"],
        default=os.environ.get("MODEL_TYPE", "closed"),
        help="For intention_analysis: open (vLLM) or closed (OpenAI API). Default: closed",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name. Open: default Qwen/Qwen3-4B-Instruct-2507. Closed: default gpt-4o-mini.",
    )
    parser.add_argument(
        "--synthesis-model",
        type=str,
        default=None,
        help="Synthesis model for synthesis_guard and synthesis_llama_guard defenders. Default: Qwen/Qwen3-4B-Instruct-2507",
    )
    parser.add_argument(
        "--guard-model",
        type=str,
        default=None,
        help="Guard model for synthesis_guard and synthesis_llama_guard defenders. For synthesis_guard: Qwen/Qwen3Guard-Gen-8B. For synthesis_llama_guard: meta-llama/Llama-Guard-3-8B",
    )
    parser.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "1")),
        help="vLLM tensor parallel size (GPUs) when model-type=open. Default: 1",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.8")),
        help="vLLM GPU memory utilization when model-type=open. Default: 0.8",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("BATCH_SIZE", "500")),
        help="Batch size for evaluation. Default: 500",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Dataset directory containing benign_train.jsonl and harmful_train.jsonl",
    )
    parser.add_argument(
        "--conversations-pkl",
        type=str,
        default=None,
        help="Path to a conversations pickle file (e.g., from mda_data_utils --mode eval). "
        "Bypasses --dataset-dir loading entirely.",
    )
    parser.add_argument(
        "--mda-mode",
        action="store_true",
        help="Use MDA-specific prompts (query-only, no assistant responses). "
        "Only affects prompt templates in naive_llm, intention_analysis, sft, rl defenders.",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        choices=["train", "valid", "test"],
        help="Dataset split to load when using --dataset-dir",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of samples from each dataset",
    )  # this is for debugging
    parser.add_argument(
        "--mixing-strategy",
        choices=["random", "similarity"],
        default="random",
        help="Strategy for mixing datasets: random or similarity-based",
    )
    parser.add_argument(
        "--skip-mixed",
        action="store_true",
        dest="skip_mixed",
        help="Skip mixed benign/harmful datasets and evaluate only raw benign/harmful (DEFAULT: False)",
    )
    parser.set_defaults(skip_mixed=False)
    # Insertion control - Percentage mode
    parser.add_argument(
        "--insert-ratio",
        type=float,
        default=None,
        help="Percentage mode: insert (ratio * original_turns) turns. Overrides min/max if set.",
    )
    # Insertion control - Range mode for mixed benign
    parser.add_argument(
        "--min-insert-benign",
        type=int,
        default=0,
        help="Range mode (mixed benign): minimum harmful turns to insert per conversation",
    )
    parser.add_argument(
        "--max-insert-benign",
        type=int,
        default=2,
        help="Range mode (mixed benign): maximum harmful turns to insert per conversation",
    )
    # Insertion control - Range mode for mixed harmful
    parser.add_argument(
        "--min-insert-harmful",
        type=int,
        default=0,
        help="Range mode (mixed harmful): minimum benign pairs to insert before each harmful turn",
    )
    parser.add_argument(
        "--max-insert-harmful",
        type=int,
        default=2,
        help="Range mode (mixed harmful): maximum benign pairs to insert before each harmful turn",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face token for private models. Can also be set via HF_TOKEN environment variable.",
    )
    # SFT-specific parameters
    parser.add_argument(
        "--sft-type",
        choices=["lora", "full"],
        default="lora",
        help="SFT training type: lora (LoRA adapter) or full (full fine-tuning)",
    )
    parser.add_argument(
        "--sft-base-model",
        type=str,
        default=None,
        help="Base model for LoRA (only used when sft-type=lora). If not set, uses --model",
    )
    parser.add_argument(
        "--sft-lora-path",
        type=str,
        default=None,
        help="Path to LoRA adapter weights (only used when sft-type=lora)",
    )
    parser.add_argument(
        "--sft-checkpoint",
        type=str,
        default=None,
        help="Path to full fine-tuned checkpoint (only used when sft-type=full)",
    )
    # RL Defender parameters (only used when defender=rl)
    parser.add_argument(
        "--rl-type",
        choices=["lora", "full"],
        default="lora",
        help="RL defender checkpoint type: lora (LoRA adapter) or full (full checkpoint). Default: lora",
    )
    parser.add_argument(
        "--rl-base-model",
        type=str,
        default=None,
        help="Base model for RL LoRA (only when rl-type=lora). Falls back to --model if unset.",
    )
    parser.add_argument(
        "--rl-lora-path",
        type=str,
        default=None,
        help="Path to RL LoRA adapter directory (only when rl-type=lora).",
    )
    parser.add_argument(
        "--rl-checkpoint",
        type=str,
        default=None,
        help="Path to full RL checkpoint directory (only when rl-type=full).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("Setting up Intent Defender Framework...")

    # 1. Load Data
    if args.conversations_pkl:
        # Direct pkl loading (e.g., MDA data converted via mda_data_utils)
        import pickle

        print(f"Loading conversations from {args.conversations_pkl}...")
        with open(args.conversations_pkl, "rb") as f:
            all_conversations_loaded = pickle.load(f)
        if args.limit:
            all_conversations_loaded = all_conversations_loaded[: args.limit]
        n_harmful = sum(1 for c in all_conversations_loaded if c.conv_type == "harmful")
        n_benign = sum(1 for c in all_conversations_loaded if c.conv_type == "benign")
        print(
            f"  Loaded {len(all_conversations_loaded)} conversations ({n_harmful} harmful, {n_benign} benign)"
        )
        # Skip mixing — use loaded conversations directly
        benign_convs = [c for c in all_conversations_loaded if c.conv_type == "benign"]
        harmful_convs = [
            c for c in all_conversations_loaded if c.conv_type == "harmful"
        ]
        mixed_benign_convs = []
        mixed_harmful_convs = []
    else:
        if not args.dataset_dir:
            raise ValueError("Either --dataset-dir or --conversations-pkl is required")
        # Paths
        benign_path, harmful_path = resolve_dataset_paths(
            args.dataset_dir, args.dataset_split, "", ""
        )

        # Load Data & Mix
        print(f"Loading data from {benign_path} and {harmful_path}...")
        mixer = ConversationMixer(benign_path, harmful_path)

        if args.limit:
            print(f"Limiting datasets to top {args.limit} samples...")
            mixer.benign_data = mixer.benign_data[: args.limit]
            mixer.harmful_data = mixer.harmful_data[: args.limit]

        print("Generating datasets...")
        benign_convs = mixer.get_benign_conversations()
        harmful_convs = mixer.get_harmful_conversations()

        if args.skip_mixed:
            print("  Skipping mixed datasets (raw benign/harmful only)")
            mixed_benign_convs = []
            mixed_harmful_convs = []
        else:
            # Generate both types of mixed datasets
            print(f"  Using mixing strategy: {args.mixing_strategy}")
            if args.insert_ratio is not None:
                print(f"  Using percentage mode: insert_ratio={args.insert_ratio}")
                mixed_benign_convs = mixer.generate_mixed_benign(
                    insert_ratio=args.insert_ratio,
                    strategy=args.mixing_strategy,
                )
                mixed_harmful_convs = mixer.generate_mixed_harmful(
                    insert_ratio=args.insert_ratio,
                    strategy=args.mixing_strategy,
                )
            else:
                print(
                    f"  Using range mode: benign=[{args.min_insert_benign}, {args.max_insert_benign}], harmful=[{args.min_insert_harmful}, {args.max_insert_harmful}]"
                )
                mixed_benign_convs = mixer.generate_mixed_benign(
                    min_insert=args.min_insert_benign,
                    max_insert=args.max_insert_benign,
                    strategy=args.mixing_strategy,
                )
                mixed_harmful_convs = mixer.generate_mixed_harmful(
                    min_insert_per_turn=args.min_insert_harmful,
                    max_insert_per_turn=args.max_insert_harmful,
                    strategy=args.mixing_strategy,
                )

    print(f"Stats:")
    print(f"  Benign Conversations: {len(benign_convs)}")
    print(f"  Harmful Conversations: {len(harmful_convs)}")
    print(f"  Mixed Benign Conversations: {len(mixed_benign_convs)}")
    print(f"  Mixed Harmful Conversations: {len(mixed_harmful_convs)}")

    # 2. Initialize Defender
    print(f"\nInitializing Defender: {args.defender}...")
    if args.defender == "dummy":
        defender = DummyDefender(
            keywords=["security", "jamming", "spoofing", "fumes", "toxic", "poison"]
        )
    elif args.defender == "random":
        defender = RandomDefender(block_prob=args.block_prob)
    elif args.defender == "sequential_monitor":
        from src.intention_analysis import VLLMClient, OpenAILLMClient
        from src.defender import SequentialMonitorDefender

        model_type = args.model_type
        model = args.model or (
            os.environ.get("MODEL_OPEN", "Qwen/Qwen3-4B-Instruct-2507")
            if model_type == "open"
            else os.environ.get("MODEL_CLOSED", "gpt-4o-mini")
        )
        print(f"  Model type: {model_type}")
        print(f"  Model: {model}")
        print(f"  Prompt type: {args.monitor_prompt_type}")
        print(f"  Threshold: {args.monitor_threshold}")

        if model_type == "open":
            llm_client = VLLMClient(
                model=model,
                tensor_parallel_size=args.vllm_tensor_parallel_size,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                hf_token=args.hf_token,
            )
        else:
            llm_client = OpenAILLMClient(model=model)

        defender = SequentialMonitorDefender(
            llm_client=llm_client,
            prompt_type=args.monitor_prompt_type,
            threshold=args.monitor_threshold,
        )
    elif args.defender == "naive_llm":
        from src.intention_analysis import VLLMClient, OpenAILLMClient

        model_type = args.model_type
        model = args.model or (
            os.environ.get("MODEL_OPEN", "Qwen/Qwen3-4B-Instruct-2507")
            if model_type == "open"
            else os.environ.get("MODEL_CLOSED", "gpt-4o-mini")
        )
        print(f"  Model type: {model_type}")
        print(
            f"  Model: {model} (direct harmful/benign classification, no intention analysis)"
        )
        if model_type == "open":
            llm_client = VLLMClient(
                model=model,
                tensor_parallel_size=args.vllm_tensor_parallel_size,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                hf_token=args.hf_token,
            )
        else:
            llm_client = OpenAILLMClient(model=model)
        defender = NaiveLLMDefender(llm_client=llm_client, mda_mode=args.mda_mode)
    elif args.defender == "sft":
        from src.intention_analysis import VLLMClient

        print(f"  SFT type: {args.sft_type}")
        if args.sft_type == "lora":
            # LoRA: base model + adapter
            base_model = (
                args.sft_base_model
                or args.model
                or os.environ.get("MODEL_OPEN", "Qwen/Qwen3-4B-Instruct-2507")
            )
            lora_path = args.sft_lora_path
            if not lora_path:
                raise ValueError(
                    "--sft-lora-path is required when using sft defender with lora"
                )
            print(f"  Base model: {base_model}")
            print(f"  LoRA adapter: {lora_path}")
            llm_client = VLLMClient(
                model=base_model,
                tensor_parallel_size=args.vllm_tensor_parallel_size,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                hf_token=args.hf_token,
                enable_lora=True,
                lora_modules={"sft": lora_path},
                max_lora_rank=64,
            )
        else:  # full fine-tuning
            checkpoint_path = args.sft_checkpoint
            if not checkpoint_path:
                raise ValueError(
                    "--sft-checkpoint is required when using sft defender with full fine-tuning"
                )
            print(f"  SFT checkpoint: {checkpoint_path}")
            llm_client = VLLMClient(
                model=checkpoint_path,
                tensor_parallel_size=args.vllm_tensor_parallel_size,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                hf_token=args.hf_token,
            )
        defender = SFTDefender(llm_client=llm_client, mda_mode=args.mda_mode)
    elif args.defender == "rl":
        from src.intention_analysis import VLLMClient

        print(f"  RL type: {args.rl_type}")
        if args.rl_type == "lora":
            base_model = (
                args.rl_base_model
                or args.model
                or os.environ.get("MODEL_OPEN", "Qwen/Qwen3-4B-Instruct-2507")
            )
            lora_path = args.rl_lora_path
            if not lora_path:
                raise ValueError(
                    "--rl-lora-path is required when using rl defender with lora"
                )
            print(f"  Base model: {base_model}")
            print(f"  LoRA adapter: {lora_path}")
            llm_client = VLLMClient(
                model=base_model,
                tensor_parallel_size=args.vllm_tensor_parallel_size,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                hf_token=args.hf_token,
                enable_lora=True,
                lora_modules={"rl": lora_path},
                max_lora_rank=64,
            )
        else:  # full checkpoint
            checkpoint_path = args.rl_checkpoint
            if not checkpoint_path:
                raise ValueError(
                    "--rl-checkpoint is required when using rl defender with full"
                )
            print(f"  RL checkpoint: {checkpoint_path}")
            llm_client = VLLMClient(
                model=checkpoint_path,
                tensor_parallel_size=args.vllm_tensor_parallel_size,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                hf_token=args.hf_token,
            )
        defender = RLDefender(llm_client=llm_client, mda_mode=args.mda_mode)
    elif args.defender == "llama_guard":
        from src.intention_analysis import VLLMClient

        if args.model_type == "closed":
            raise ValueError("llama_guard defender requires --model-type open")

        model = args.model or os.environ.get(
            "MODEL_LLAMA_GUARD", "meta-llama/Llama-Guard-3-8B"
        )
        print(f"  Model type: open")
        print(f"  Model: {model} (Llama Guard 3)")
        llm_client = VLLMClient(
            model=model,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            hf_token=args.hf_token,
        )
        defender = LlamaGuardDefender(llm_client=llm_client)
    elif args.defender == "qwen_guard":
        from src.intention_analysis import VLLMClient

        if args.model_type == "closed":
            raise ValueError("qwen_guard defender requires --model-type open")

        model = args.model or os.environ.get(
            "MODEL_QWEN_GUARD", "Qwen/Qwen3Guard-Gen-8B"
        )
        print(f"  Model type: open")
        print(f"  Model: {model} (Qwen Guard)")
        llm_client = VLLMClient(
            model=model,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            hf_token=args.hf_token,
        )
        defender = QwenGuardDefender(llm_client=llm_client)
    elif args.defender == "synthesis_guard":
        from src.intention_analysis import VLLMClient

        if args.model_type == "closed":
            raise ValueError("synthesis_guard defender requires --model-type open")

        synthesis_model = (
            args.synthesis_model
            or args.model
            or os.environ.get("MODEL_SYNTHESIS", "Qwen/Qwen3-4B-Instruct-2507")
        )
        guard_model = args.guard_model or os.environ.get(
            "MODEL_SYNTHESIS_GUARD", "Qwen/Qwen3Guard-Gen-8B"
        )
        synthesis_vram_util = min(
            args.vllm_gpu_memory_utilization,
            float(os.environ.get("VLLM_SYNTHESIS_GPU_MEMORY_UTILIZATION", "0.25")),
        )
        guard_vram_util = min(
            args.vllm_gpu_memory_utilization,
            float(os.environ.get("VLLM_GUARD_GPU_MEMORY_UTILIZATION", "0.45")),
        )
        print(f"  Model type: open")
        print(f"  Synthesis model: {synthesis_model} (Qwen3 4B)")
        print(f"  Guard model: {guard_model}")
        print(
            f"  vLLM gpu_memory_utilization (synthesis/guard): {synthesis_vram_util}/{guard_vram_util}"
        )

        synthesis_client = VLLMClient(
            model=synthesis_model,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            gpu_memory_utilization=synthesis_vram_util,
            hf_token=args.hf_token,
        )
        guard_client = VLLMClient(
            model=guard_model,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            gpu_memory_utilization=guard_vram_util,
            hf_token=args.hf_token,
        )
        defender = SynthesisGuardDefender(
            synthesis_client=synthesis_client, guard_client=guard_client
        )
    elif args.defender == "synthesis_llama_guard":
        from src.intention_analysis import VLLMClient

        if args.model_type == "closed":
            raise ValueError(
                "synthesis_llama_guard defender requires --model-type open"
            )

        synthesis_model = (
            args.synthesis_model
            or args.model
            or os.environ.get("MODEL_SYNTHESIS", "Qwen/Qwen3-4B-Instruct-2507")
        )
        guard_model = args.guard_model or os.environ.get(
            "MODEL_SYNTHESIS_LLAMA_GUARD", "meta-llama/Llama-Guard-3-8B"
        )
        synthesis_vram_util = min(
            args.vllm_gpu_memory_utilization,
            float(os.environ.get("VLLM_SYNTHESIS_GPU_MEMORY_UTILIZATION", "0.25")),
        )
        guard_vram_util = min(
            args.vllm_gpu_memory_utilization,
            float(os.environ.get("VLLM_GUARD_GPU_MEMORY_UTILIZATION", "0.45")),
        )
        print(f"  Model type: open")
        print(f"  Synthesis model: {synthesis_model} (Qwen3 4B)")
        print(f"  Guard model: {guard_model}")
        print(
            f"  vLLM gpu_memory_utilization (synthesis/guard): {synthesis_vram_util}/{guard_vram_util}"
        )

        synthesis_client = VLLMClient(
            model=synthesis_model,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            gpu_memory_utilization=synthesis_vram_util,
            hf_token=args.hf_token,
        )
        guard_client = VLLMClient(
            model=guard_model,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            gpu_memory_utilization=guard_vram_util,
            hf_token=args.hf_token,
        )
        defender = SynthesisLlamaGuardDefender(
            synthesis_client=synthesis_client, guard_client=guard_client
        )
    else:  # intention_analysis
        from src.intention_analysis import OpenAILLMClient, VLLMClient

        model_type = args.model_type
        model = args.model or (
            os.environ.get("MODEL_OPEN", "Qwen/Qwen3-4B-Instruct-2507")
            if model_type == "open"
            else os.environ.get("MODEL_CLOSED", "gpt-4o-mini")
        )
        print(f"  Model type: {model_type}")
        print(f"  Model: {model}")
        if model_type == "open":
            print(f"  vLLM tensor_parallel_size: {args.vllm_tensor_parallel_size}")
            defender = IntentionAnalysisDefender(
                llm_client=VLLMClient(
                    model=model,
                    tensor_parallel_size=args.vllm_tensor_parallel_size,
                    gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                    hf_token=args.hf_token,
                ),
                mda_mode=args.mda_mode,
            )
        else:
            defender = IntentionAnalysisDefender(
                llm_client=OpenAILLMClient(model=model),
                mda_mode=args.mda_mode,
            )

    # 3. Evaluate
    evaluator = Evaluator(defender)

    # Setup checkpoint saving
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate config string for filename
    config_str = f"{args.defender}"
    if args.defender in ["naive_llm", "intention_analysis"]:
        model_name = (
            args.model or ("qwen" if args.model_type == "open" else "gpt4omini")
        ).replace("/", "_")
        config_str += f"_{args.model_type}_{model_name}"
    elif args.defender == "sft":
        config_str += f"_{args.sft_type}"
        if args.sft_type == "lora":
            lora_name = os.path.basename(args.sft_lora_path or "lora")
            config_str += f"_{lora_name}"
        else:
            checkpoint_name = os.path.basename(args.sft_checkpoint or "full")
            config_str += f"_{checkpoint_name}"
    elif args.defender == "rl":
        config_str += f"_{args.rl_type}"
        if args.rl_type == "lora":
            lora_name = os.path.basename(args.rl_lora_path or "lora")
            config_str += f"_{lora_name}"
        else:
            checkpoint_name = os.path.basename(args.rl_checkpoint or "full")
            config_str += f"_{checkpoint_name}"
    if args.mda_mode:
        config_str += "_mda"
    if args.conversations_pkl:
        pkl_name = os.path.splitext(os.path.basename(args.conversations_pkl))[0]
        config_str += f"_pkl_{pkl_name}"
    elif args.skip_mixed:
        config_str += "_nomixed"
    else:
        config_str += f"_{args.mixing_strategy}"
        if args.insert_ratio is not None:
            config_str += f"_ratio{args.insert_ratio}"
        else:
            config_str += f"_benign{args.min_insert_benign}-{args.max_insert_benign}_harmful{args.min_insert_harmful}-{args.max_insert_harmful}"

    def save_checkpoint(conv_count, checkpoint_results, dataset_name):
        """Save intermediate checkpoint."""
        checkpoint_filename = (
            f"{timestamp}_{config_str}_checkpoint_{dataset_name}_{conv_count}.json"
        )
        checkpoint_filepath = os.path.join(results_dir, checkpoint_filename)

        checkpoint_data = {
            "timestamp": timestamp,
            "checkpoint": {
                "dataset": dataset_name,
                "conversations_processed": conv_count,
            },
            "config": {
                "defender": args.defender,
                "model_type": (
                    args.model_type
                    if args.defender in ["naive_llm", "intention_analysis"]
                    else None
                ),
                "model": args.model,
                # SFT-specific config
                "sft_type": args.sft_type if args.defender == "sft" else None,
                "sft_base_model": (
                    args.sft_base_model if args.defender == "sft" else None
                ),
                "sft_lora_path": args.sft_lora_path if args.defender == "sft" else None,
                "sft_checkpoint": (
                    args.sft_checkpoint if args.defender == "sft" else None
                ),
                # RL-specific config
                "rl_type": args.rl_type if args.defender == "rl" else None,
                "rl_base_model": args.rl_base_model if args.defender == "rl" else None,
                "rl_lora_path": args.rl_lora_path if args.defender == "rl" else None,
                "rl_checkpoint": args.rl_checkpoint if args.defender == "rl" else None,
                # Data mixing config
                "skip_mixed": args.skip_mixed,
                "mixing_strategy": args.mixing_strategy,
                "insert_ratio": args.insert_ratio,
                "min_insert_benign": args.min_insert_benign,
                "max_insert_benign": args.max_insert_benign,
                "min_insert_harmful": args.min_insert_harmful,
                "max_insert_harmful": args.max_insert_harmful,
                "dataset_dir": args.dataset_dir,
                "dataset_split": args.dataset_split,
                "limit": args.limit,
            },
            "results": checkpoint_results,
        }

        with open(checkpoint_filepath, "w") as f:
            json.dump(checkpoint_data, f, indent=2)
        print(f"  Checkpoint saved: {checkpoint_filename}")

    # Combine ALL conversations into a single evaluate() call for maximum batching.
    # The evaluator's sliding window (batch_size) keeps the LLM batch full by continuously
    # refilling from the full pool, rather than shrinking as each per-category set finishes.
    all_conversations = (
        benign_convs + harmful_convs + mixed_benign_convs + mixed_harmful_convs
    )
    print(
        f"\nEvaluating all {len(all_conversations)} conversations together (batch_size={args.batch_size})..."
    )

    combined_results = evaluator.evaluate(
        all_conversations,
        desc="Evaluating",
        checkpoint_callback=lambda count, results: save_checkpoint(
            count, results, "all"
        ),
        checkpoint_interval=500,
        batch_size=args.batch_size,
    )

    # MEMORY OPTIMIZATION: For synthesis defenders, finalize synthesis phase
    # This unloads the synthesis model to free VRAM before saving results
    if hasattr(defender, "finalize_synthesis"):
        defender.finalize_synthesis()

    # Print combined results
    print_results(combined_results, "EVALUATION RESULTS - ALL CASES")

    # Compute FCI metrics from per-sample scores
    fci_metrics = compute_fci_metrics(combined_results, prefix="combined")
    if fci_metrics:
        print(f"\n{'='*70}")
        print(f"{'FCI METRICS':^70}")
        print("=" * 70)
        print(json.dumps(fci_metrics, indent=2))
        print("=" * 70 + "\n")
        combined_results.update(fci_metrics)

    # Save final results to file (timestamp and config_str already set)
    filename = f"{timestamp}_{config_str}_final.json"
    filepath = os.path.join(results_dir, filename)

    # Save with full config metadata
    output_data = {
        "timestamp": timestamp,
        "config": {
            "defender": args.defender,
            "model_type": (
                args.model_type
                if args.defender in ["naive_llm", "intention_analysis"]
                else None
            ),
            "model": args.model,
            # SFT-specific config
            "sft_type": args.sft_type if args.defender == "sft" else None,
            "sft_base_model": args.sft_base_model if args.defender == "sft" else None,
            "sft_lora_path": args.sft_lora_path if args.defender == "sft" else None,
            "sft_checkpoint": args.sft_checkpoint if args.defender == "sft" else None,
            # RL-specific config
            "rl_type": args.rl_type if args.defender == "rl" else None,
            "rl_base_model": args.rl_base_model if args.defender == "rl" else None,
            "rl_lora_path": args.rl_lora_path if args.defender == "rl" else None,
            "rl_checkpoint": args.rl_checkpoint if args.defender == "rl" else None,
            # Data mixing config
            "skip_mixed": args.skip_mixed,
            "mixing_strategy": args.mixing_strategy,
            "insert_ratio": args.insert_ratio,
            "min_insert_benign": args.min_insert_benign,
            "max_insert_benign": args.max_insert_benign,
            "min_insert_harmful": args.min_insert_harmful,
            "max_insert_harmful": args.max_insert_harmful,
            "dataset_dir": args.dataset_dir,
            "dataset_split": args.dataset_split,
            "limit": args.limit,
        },
        "results": combined_results,
    }

    with open(filepath, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Results saved to: {filepath}")


if __name__ == "__main__":
    main()
