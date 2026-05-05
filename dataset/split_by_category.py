import json
import csv
import os
import random


def load_prompt_categories(csv_files):
    prompt_to_category = {}
    for csv_file in csv_files:
        with open(csv_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt = row["prompt"].strip()
                # Use 'category' for benign, 'harmful_category' or similar if needed.
                # Actually, benign_prompts.csv has 'category', harmful_prompts.csv also has 'category'?
                # Let's check the CSVs later if needed, but the original code used row['category'].strip()
                category = row["category"].strip()
                prompt_to_category[prompt] = category
    return prompt_to_category


def process_datasets(input_dir, prompt_to_category):
    all_samples = []
    jsonl_files = [f for f in os.listdir(input_dir) if f.endswith(".jsonl")]

    for filename in jsonl_files:
        file_path = os.path.join(input_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    meta_intent = sample.get("meta_intent", "").strip()
                    if meta_intent in prompt_to_category:
                        sample["category"] = prompt_to_category[meta_intent]
                        dataset_key = sample.get("dataset_key", "")
                        if "benign" in dataset_key:
                            sample["is_benign"] = True
                        elif "harmful" in dataset_key:
                            sample["is_benign"] = False
                        else:
                            sample["is_benign"] = "benign" in filename
                        all_samples.append(sample)
    return all_samples


def save_jsonl(samples, output_dir, filename):
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"Saved {len(samples)} samples to {path}")


def generate_split(train_samples, test_samples, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Group by meta_intent for the training set to avoid leakage between train/valid
    def group_by_intent(samples):
        grouped = {}
        for s in samples:
            intent = s["meta_intent"]
            if intent not in grouped:
                grouped[intent] = []
            grouped[intent].append(s)
        return list(grouped.values())

    train_intents = group_by_intent(train_samples)
    random.seed(42)
    random.shuffle(train_intents)

    split_idx = int(0.9 * len(train_intents))
    train_pool = [s for group in train_intents[:split_idx] for s in group]
    valid_pool = [s for group in train_intents[split_idx:] for s in group]
    test_pool = test_samples

    save_jsonl(
        [s for s in train_pool if s["is_benign"]], output_dir, "benign_train.jsonl"
    )
    save_jsonl(
        [s for s in train_pool if not s["is_benign"]], output_dir, "harmful_train.jsonl"
    )

    save_jsonl(
        [s for s in valid_pool if s["is_benign"]], output_dir, "benign_valid.jsonl"
    )
    save_jsonl(
        [s for s in valid_pool if not s["is_benign"]], output_dir, "harmful_valid.jsonl"
    )

    save_jsonl(
        [s for s in test_pool if s["is_benign"]], output_dir, "benign_test.jsonl"
    )
    save_jsonl(
        [s for s in test_pool if not s["is_benign"]], output_dir, "harmful_test.jsonl"
    )


if __name__ == "__main__":
    csv_files = ["dataset/benign_prompts.csv", "dataset/harmful_prompts.csv"]
    input_dir = "dataset/gpt52-gen_filter/"

    prompt_to_category = load_prompt_categories(csv_files)
    all_samples = process_datasets(input_dir, prompt_to_category)

    chemistry = [s for s in all_samples if s["category"].lower() == "chemistry"]
    cybersecurity = [s for s in all_samples if s["category"].lower() == "cybersecurity"]

    print(f"Total Chemistry samples: {len(chemistry)}")
    print(f"Total Cybersecurity samples: {len(cybersecurity)}")

    print("\nGenerating Chemistry Split (Train: Chem, Test: Cyber)...")
    generate_split(chemistry, cybersecurity, "dataset/chemistry_split_filter")

    print("\nGenerating Cybersecurity Split (Train: Cyber, Test: Chem)...")
    generate_split(cybersecurity, chemistry, "dataset/cybersecurity_split_filter")
