"""
Dataset Preprocessing Script for Multi-turn RL Defense Project

This script filters harmful and benign prompts from the WildJailbreak dataset
using Gemini API for category classification.

Categories:
- chemistry: Chemistry-related harmful/benign prompts
- cybersecurity: Cybersecurity-related harmful/benign prompts
"""

import json
import os
import time
import pandas as pd
from datasets import load_dataset
from google import genai  # pip install google-genai
from tqdm import tqdm
from typing import List, Dict

# ============== Configuration ==============
HF_TOKEN = os.getenv("HF_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

# Categories to filter
CATEGORIES = ["chemistry", "cybersecurity"]

# Number of samples needed per category
SAMPLES_PER_CATEGORY = 500  # Adjust as needed

# Batch size for Gemini API calls
BATCH_SIZE = 50

# Output file paths
OUTPUT_HARMFUL_CSV = "harmful_prompts.csv"
OUTPUT_BENIGN_CSV = "benign_prompts.csv"

# ============== Setup Gemini ==============
# Using the newer google-genai SDK with Client approach
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.5-flash"  # Model name for API calls


# ============== Filtering Prompt Template ==============
FILTER_PROMPT_TEMPLATE = """You are an expert classifier tasked with categorizing prompts into specific domains.

Your task is to analyze the following batch of prompts and identify which ones belong to these categories:
1. **chemistry**: Prompts related to chemical substances, reactions, synthesis, laboratory procedures, chemical compounds, drugs, explosives, toxins, or any chemistry-related topics.
2. **cybersecurity**: Prompts related to hacking, malware, network attacks, system vulnerabilities, data breaches, phishing, social engineering, unauthorized access, or any cybersecurity/computer security topics.

For each prompt, determine if it belongs to one of these categories. A prompt may belong to neither category if it's unrelated to both domains.

**Instructions:**
- Analyze each prompt carefully based on its content and intent
- Return ONLY a JSON array with the classification results
- Each element should have: "index" (0-based), "category" (one of: "chemistry", "cybersecurity", "none")
- Be inclusive: if a prompt is even partially related to chemistry or cybersecurity, classify it accordingly
- Focus on the TOPIC, not the harmfulness

**Prompts to classify:**
{prompts_text}

**Response format (return ONLY valid JSON, no markdown):**
[
  {{"index": 0, "category": "chemistry"}},
  {{"index": 1, "category": "none"}},
  {{"index": 2, "category": "cybersecurity"}},
  ...
]
"""


def load_wildjailbreak_dataset():
    """Load the WildJailbreak dataset from HuggingFace."""
    print("=" * 60)
    print("Loading WildJailbreak dataset from HuggingFace...")
    print("=" * 60)

    # IMPORTANT: Use delimiter="\t" and keep_default_na=False as specified in official HuggingFace repo
    # This prevents PyArrow from misinterpreting strings as other types (e.g., doubles)
    dataset = load_dataset(
        "allenai/wildjailbreak",
        "train",
        delimiter="\t",
        keep_default_na=False,
        token=HF_TOKEN,
    )

    return dataset


def explore_dataset(dataset):
    """Explore and print dataset structure and samples."""
    print("\n" + "=" * 60)
    print("Dataset Structure and Statistics")
    print("=" * 60)

    print(f"\nDataset keys: {dataset.keys()}")

    for split_name in dataset.keys():
        split_data = dataset[split_name]
        print(f"\n--- Split: {split_name} ---")
        print(f"Number of samples: {len(split_data)}")
        print(f"Features: {split_data.features}")

        # Show column names
        print(f"Columns: {split_data.column_names}")

        # Show first sample
        if len(split_data) > 0:
            print(f"\nFirst sample:")
            for key, value in split_data[0].items():
                if isinstance(value, str) and len(value) > 200:
                    print(f"  {key}: {value[:200]}...")
                else:
                    print(f"  {key}: {value}")

    return dataset


def get_data_by_type(dataset, data_type: str) -> List[Dict]:
    """
    Extract prompts by data_type from the dataset.

    Args:
        dataset: The loaded HuggingFace dataset
        data_type: One of 'vanilla_harmful', 'vanilla_benign', 'adversarial_harmful', 'adversarial_benign'

    Returns:
        List of dictionaries with prompt information
    """
    print(f"\nExtracting {data_type} prompts...")

    # The dataset has a 'train' split
    train_data = dataset["train"]

    # Filter by data_type
    filtered_data = []
    for i, item in enumerate(train_data):
        if item.get("data_type") == data_type:
            # For vanilla types, the prompt is in 'vanilla' column
            # For adversarial types, the prompt is in 'adversarial' column
            if "vanilla" in data_type:
                prompt = item.get("vanilla", "")
            else:
                prompt = item.get("adversarial", "")

            if prompt:  # Only add if prompt is not empty
                filtered_data.append(
                    {"prompt": prompt, "original_index": i, "data_type": data_type}
                )

    print(f"Found {len(filtered_data)} {data_type} prompts")
    return filtered_data


def classify_prompts_batch(prompts: List[str], batch_index: int = 0) -> List[Dict]:
    """
    Classify a batch of prompts using Gemini API.

    Args:
        prompts: List of prompt strings to classify
        batch_index: Index of the batch for logging

    Returns:
        List of classification results with index and category
    """
    # Format prompts for the API
    prompts_text = ""
    for i, prompt in enumerate(prompts):
        # Truncate very long prompts
        truncated = prompt[:500] + "..." if len(prompt) > 500 else prompt
        prompts_text += f"\n[{i}] {truncated}\n"

    full_prompt = FILTER_PROMPT_TEMPLATE.format(prompts_text=prompts_text)

    try:
        # Using the new Client-based API (same as your BlackBoxModel)
        response = gemini_client.models.generate_content(
            model=MODEL_NAME,
            contents=full_prompt,
        )

        # Extract text from response
        response_text = ""
        if hasattr(response, "text") and response.text:
            response_text = response.text.strip()
        elif hasattr(response, "candidates") and response.candidates:
            cand = response.candidates[0]
            if (
                hasattr(cand, "content")
                and cand.content
                and getattr(cand.content, "parts", None)
            ):
                parts = getattr(cand.content, "parts", None)
                for p in parts:
                    if hasattr(p, "text") and p.text:
                        response_text = p.text.strip()
                        break

        # Clean up response - remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        response_text = response_text.strip()

        # Parse JSON response
        results = json.loads(response_text)
        return results

    except json.JSONDecodeError as e:
        print(f"  Warning: JSON parsing error in batch {batch_index}: {e}")
        print(f"  Response: {response_text[:200]}...")
        return []
    except Exception as e:
        print(f"  Warning: Error in batch {batch_index}: {e}")
        return []


def filter_prompts_by_category(
    prompts_data: List[Dict],
    target_categories: List[str],
    samples_per_category: int,
    source_name: str,
) -> Dict[str, List[Dict]]:
    """
    Filter prompts by category using Gemini API.

    Args:
        prompts_data: List of prompt dictionaries
        target_categories: List of categories to filter for
        samples_per_category: Number of samples needed per category
        source_name: Name of the data source for logging

    Returns:
        Dictionary mapping category to list of filtered prompts
    """
    print(f"\n" + "=" * 60)
    print(f"Filtering {source_name} prompts by category")
    print(f"Target categories: {target_categories}")
    print(f"Samples needed per category: {samples_per_category}")
    print("=" * 60)

    # Initialize result containers
    filtered_by_category = {cat: [] for cat in target_categories}

    # Track progress
    total_prompts = len(prompts_data)
    processed = 0

    # Check if we already have enough samples
    def have_enough_samples():
        return all(
            len(filtered_by_category[cat]) >= samples_per_category
            for cat in target_categories
        )

    # Process in batches
    prompts_list = [p["prompt"] for p in prompts_data]

    with tqdm(total=total_prompts, desc="Processing prompts") as pbar:
        for batch_start in range(0, total_prompts, BATCH_SIZE):
            if have_enough_samples():
                print("\nReached target sample count for all categories!")
                break

            batch_end = min(batch_start + BATCH_SIZE, total_prompts)
            batch_prompts = prompts_list[batch_start:batch_end]
            batch_data = prompts_data[batch_start:batch_end]

            # Classify batch
            results = classify_prompts_batch(batch_prompts, batch_start // BATCH_SIZE)

            # Process results
            for result in results:
                idx = result.get("index", -1)
                category = result.get("category", "none")

                if 0 <= idx < len(batch_data) and category in target_categories:
                    if len(filtered_by_category[category]) < samples_per_category:
                        filtered_by_category[category].append(
                            {
                                "prompt": batch_data[idx]["prompt"],
                                "source": source_name,
                                "category": category,
                            }
                        )

            processed += len(batch_prompts)
            pbar.update(len(batch_prompts))

            # Print progress
            counts = {cat: len(filtered_by_category[cat]) for cat in target_categories}
            pbar.set_postfix(counts)

            # Rate limiting - avoid hitting API limits
            time.sleep(1)

    # Print final counts
    print("\nFiltering complete!")
    for cat in target_categories:
        print(f"  {cat}: {len(filtered_by_category[cat])} samples")

    return filtered_by_category


def save_to_csv(filtered_data: Dict[str, List[Dict]], output_path: str):
    """
    Save filtered prompts to CSV file.

    Args:
        filtered_data: Dictionary mapping category to list of prompts
        output_path: Path to save the CSV file
    """
    # Combine all categories into one list
    all_prompts = []
    for category, prompts in filtered_data.items():
        all_prompts.extend(prompts)

    # Create DataFrame
    df = pd.DataFrame(all_prompts, columns=["prompt", "source", "category"])

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} prompts to {output_path}")

    # Print summary
    print("\nDataset summary:")
    print(df["category"].value_counts())

    return df


def print_sample_results(harmful_df: pd.DataFrame, benign_df: pd.DataFrame):
    """Print sample results from both datasets."""
    print("\n" + "=" * 60)
    print("Sample Results Preview")
    print("=" * 60)

    print("\n--- Sample Harmful Prompts ---")
    for cat in CATEGORIES:
        samples = harmful_df[harmful_df["category"] == cat].head(2)
        print(f"\n[{cat.upper()}]:")
        for _, row in samples.iterrows():
            prompt_preview = (
                row["prompt"][:150] + "..."
                if len(row["prompt"]) > 150
                else row["prompt"]
            )
            print(f"  - {prompt_preview}")

    print("\n--- Sample Benign Prompts ---")
    for cat in CATEGORIES:
        samples = benign_df[benign_df["category"] == cat].head(2)
        print(f"\n[{cat.upper()}]:")
        for _, row in samples.iterrows():
            prompt_preview = (
                row["prompt"][:150] + "..."
                if len(row["prompt"]) > 150
                else row["prompt"]
            )
            print(f"  - {prompt_preview}")


def main():
    """Main function to run the dataset preprocessing pipeline."""
    print("\n" + "=" * 60)
    print("Multi-turn RL Defense - Dataset Preprocessing")
    print("=" * 60)

    # Step 1: Load dataset
    dataset = load_wildjailbreak_dataset()

    # Step 2: Explore dataset structure
    explore_dataset(dataset)

    # Step 3: Extract vanilla harmful prompts
    print("\n" + "-" * 40)
    print("Processing HARMFUL prompts (vanilla_harmful)")
    print("-" * 40)
    harmful_prompts = get_data_by_type(dataset, "vanilla_harmful")

    # Step 4: Filter harmful prompts by category
    filtered_harmful = filter_prompts_by_category(
        harmful_prompts,
        CATEGORIES,
        SAMPLES_PER_CATEGORY,
        source_name="wildjailbreak_vanilla_harmful",
    )

    # Step 5: Save harmful prompts
    harmful_df = save_to_csv(filtered_harmful, OUTPUT_HARMFUL_CSV)

    # Step 6: Extract vanilla benign prompts
    print("\n" + "-" * 40)
    print("Processing BENIGN prompts (vanilla_benign)")
    print("-" * 40)
    benign_prompts = get_data_by_type(dataset, "vanilla_benign")

    # Step 7: Filter benign prompts by category
    filtered_benign = filter_prompts_by_category(
        benign_prompts,
        CATEGORIES,
        SAMPLES_PER_CATEGORY,
        source_name="wildjailbreak_vanilla_benign",
    )

    # Step 8: Save benign prompts
    benign_df = save_to_csv(filtered_benign, OUTPUT_BENIGN_CSV)

    # Step 9: Print sample results
    print_sample_results(harmful_df, benign_df)

    print("\n" + "=" * 60)
    print("Dataset preprocessing complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - Harmful prompts: {OUTPUT_HARMFUL_CSV}")
    print(f"  - Benign prompts: {OUTPUT_BENIGN_CSV}")


if __name__ == "__main__":
    main()
