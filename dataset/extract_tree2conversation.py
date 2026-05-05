import json
import os
import argparse
import glob
from typing import List, Dict, Any, Optional
from collections import defaultdict


def extract_trajectories_from_file(
    file_path: str, prefer_success: bool = True
) -> List[Dict[str, Any]]:
    """
    Reads the intermediate result file and extracts the best trajectory for EACH rollout.
    Rollouts are detected by changes in rollout_id or resets in iteration count.
    """
    records = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

    if not records:
        return []

    # Group records by rollout_id
    # If rollout_id is missing, we try to detect it via iteration resets
    rollout_groups = defaultdict(list)
    current_rollout_id = None
    
    # First pass: identify explicit rollout_id if available
    has_explicit_rollout_id = any("rollout_id" in r for r in records)
    
    if has_explicit_rollout_id:
        for r in records:
            rid = r.get("rollout_id", 0)
            rollout_groups[rid].append(r)
    else:
        # Detect rollout boundaries via iteration resets
        rid_counter = 1
        prev_iter = -1
        for r in records:
            curr_iter = r.get("iteration", 0)
            if curr_iter < prev_iter:
                rid_counter += 1
            rollout_groups[rid_counter].append(r)
            prev_iter = curr_iter

    extracted_trajectories = []
    
    for rid, group in rollout_groups.items():
        best_record = None
        
        # Logic to select the best record within the rollout
        if prefer_success:
            # Prioritize successful jailbreaks
            for record in reversed(group):
                if record.get("success", False) or record.get("asr_classification") == "D":
                    best_record = record
                    break
        
        if not best_record:
            # Fallback: find the latest record with a best_node
            for record in reversed(group):
                if record.get("best_node"):
                    best_record = record
                    break
        
        if not best_record:
            continue

        data = best_record
        best_node = data.get("best_node")
        
        # Reconstruct conversation
        conversation = []
        subgoals = best_node.get("subgoals", [])
        responses = best_node.get("responses", [])

        # If subgoals/responses are missing in best_node, try to reconstruct from tree_structure
        if (not subgoals or not responses) and "tree_structure" in data and "trajectory" in best_node:
            tree = data.get("tree_structure", {})
            trajectory = best_node.get("trajectory", [])
            subgoals = []
            responses = []
            for node_id in trajectory:
                if node_id in tree:
                    node = tree[node_id]
                    action = node.get("action", "").strip()
                    observation = node.get("observation", "").strip()
                    if action:
                        subgoals.append(action)
                        responses.append(observation)

        for i, (q, a) in enumerate(zip(subgoals, responses)):
            turn_id = i + 1
            conversation.append({
                "turn_id": turn_id,
                "role": "user",
                "content": q,
                "hidden_rationale": "Decomposed subgoal for information gathering.",
            })
            conversation.append({
                "turn_id": turn_id,
                "role": "assistant",
                "content": a
            })

        entry = {
            "sample_index": data.get("sample_index"),
            "rollout_id": rid,
            "dataset_key": data.get("dataset_key"),
            "is_jailbreak_success": data.get("success", False) or data.get("asr_classification") == "D",
            "asr_classification": data.get("asr_classification"),
            "conversation": conversation,
            "synthesized_knowledge": best_node.get("synthesized_response", ""),
            "metadata": {
                "max_depth": data.get("max_depth_reached"),
                "iteration": data.get("iteration"),
            },
        }
        extracted_trajectories.append(entry)

    return extracted_trajectories


def load_intent_map(csv_files: List[str]) -> Dict[str, str]:
    """Load original queries from CSV files to map sample_index back to meta_intent."""
    import csv
    mapping = {}
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            continue
        with open(csv_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # We assume order in CSV matches sample_index if no explicit index column exists
            for idx, row in enumerate(reader):
                if "prompt" in row:
                    mapping[idx] = row["prompt"]
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Extract CKA trajectories into a multi-turn dataset.")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory containing inter_result_sample_*.json files")
    parser.add_argument("--output-file", type=str, default="cka_multiturn_dataset.jsonl", help="Output JSONL file path")
    parser.add_argument("--only-success", action="store_true", help="Only include successful jailbreaks")
    parser.add_argument("--benign-csv", type=str, default="dataset/benign_prompts.csv")
    parser.add_argument("--harmful-csv", type=str, default="dataset/harmful_prompts.csv")

    args = parser.parse_args()

    files = glob.glob(os.path.join(args.results_dir, "inter_result_sample_*.json"))
    try:
        files.sort(key=lambda x: int(os.path.basename(x).split("_sample_")[1].split(".")[0]))
    except:
        files.sort()

    if not files:
        print(f"No intermediate result files found in {args.results_dir}")
        return

    intent_map_benign = load_intent_map([args.benign_csv])
    intent_map_harmful = load_intent_map([args.harmful_csv])

    dataset = []
    for file_path in files:
        trajectories = extract_trajectories_from_file(file_path, prefer_success=True)
        
        # Try to determine if it's benign or harmful from filename or content
        is_harmful = "harmful" in file_path.lower() or "harmful" in trajectories[0].get("dataset_key", "").lower() if trajectories else True
        intent_map = intent_map_harmful if is_harmful else intent_map_benign

        for entry in trajectories:
            if args.only_success and not entry["is_jailbreak_success"]:
                continue
            
            idx = entry.get("sample_index")
            entry["meta_intent"] = intent_map.get(idx, "Unknown")
            dataset.append(entry)

    with open(args.output_file, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Successfully saved {len(dataset)} conversations to {args.output_file}")


if __name__ == "__main__":
    main()
