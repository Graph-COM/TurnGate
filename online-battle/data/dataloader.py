import pandas as pd
import os
from typing import Dict, List, Any, Optional
from collections import Counter


class DataLoader:
    """
    DataLoader class for loading and managing different jailbreak datasets.

    This class provides a unified interface to load various jailbreak datasets
    and perform statistical analysis on them.
    """

    def __init__(self, data_dir: str = "data/datasets"):
        """
        Initialize the DataLoader.

        Args:
            data_dir (str): Base directory path where datasets are stored
        """
        self.data_dir = data_dir
        self.loaded_data = {}
        self.dataset_stats = {}

    def load(
        self, dataset_name: str, dataset_size: str = "large", **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Unified function to load different datasets.

        Args:
            dataset_name (str): Name of the dataset to load (e.g., 'strongreject', 'ckjail',
                           'jbb_behaviors_cka', 'strongreject_cka')
            dataset_size (str): Size of the dataset ('large' or 'small') for datasets that support it
            **kwargs: Additional arguments for specific dataset loaders

        Returns:
            List[Dict[str, Any]]: List of loaded data samples

        Raises:
            ValueError: If dataset_name is not supported
        """
        name = dataset_name.lower()
        if name == "strongreject":
            data = self._load_strongreject(dataset_size, **kwargs)
            return data
        elif name == "ckjail":
            # ckjail has a single CSV; dataset_size is ignored
            data = self._load_ckjail(**kwargs)
            return data
        elif name == "jbb_behaviors_cka" or name == "jbb-behaviors-cka":
            # Load JBB-Behaviors CKA dataset (Subset of JBB-Behaviors)
            data = self._load_jbb_behaviors_cka(**kwargs)
            return data
        elif name == "strongreject_cka" or name == "strongreject-cka":
            # Load StrongReject CKA dataset (Subset of StrongReject)
            data = self._load_strongreject_cka(**kwargs)
            return data
        elif name == "harmbench_cka" or name == "harmbench-cka" or name == "harmbench":
            # Load HarmBench CKA dataset
            data = self._load_harmbench_cka(**kwargs)
            return data
        elif name in (
            "gpt52gen_cka",
            "gpt52gen-cka",
            "gpt52gen_cka_test",
            "gpt52gen-cka-test",
        ):
            # Load GPT52-gen CKA dataset (from Intent-Defender)
            # "gpt52gen_cka" or "gpt52gen_cka_test" → test split only
            data = self._load_gpt52gen_cka(split="test", **kwargs)
            return data
        elif name in ("gpt52gen_cka_all", "gpt52gen-cka-all"):
            # Load all GPT52-gen splits (train+valid+test)
            data = self._load_gpt52gen_cka(split="all", **kwargs)
            return data
        elif name in (
            "gpt52gen_filter_test",
            "gpt52gen-filter-test",
            "gpt52gen_filter_cka_test",
        ):
            # Load GPT52-gen filtered test dataset (35 queries)
            data = self._load_gpt52gen_cka(split="filter_test", **kwargs)
            return data
        elif name in (
            "gpt52gen_filter_all",
            "gpt52gen-filter-all",
            "gpt52gen_filter_cka_all",
        ):
            # Load GPT52-gen filtered all splits (215 queries)
            data = self._load_gpt52gen_cka(split="filter_all", **kwargs)
            return data
        elif name in ("cybersecurity_filter", "cybersecurity_filter_cka"):
            data = self._load_generic_cka(
                "cybersecurity_filter_cka_dataset.csv", "cybersecurity_filter", **kwargs
            )
            return data
        elif name in ("chemistry_filter", "chemistry_filter_cka"):
            data = self._load_generic_cka(
                "chemistry_filter_cka_dataset.csv", "chemistry_filter", **kwargs
            )
            return data
        else:
            raise ValueError(
                f"Dataset '{dataset_name}' is not supported yet. "
                f"Supported datasets: ['strongreject', 'ckjail', 'jbb_behaviors_cka', "
                f"'strongreject_cka', 'harmbench_cka', 'gpt52gen_cka', 'gpt52gen_cka_all', "
                f"'gpt52gen_filter_test', 'gpt52gen_filter_all', "
                f"'cybersecurity_filter', 'chemistry_filter']"
            )

    def _load_jbb_behaviors_cka(self) -> List[Dict[str, Any]]:
        """
        Load JBB-Behaviors CKA dataset.

        This dataset is located in the cka-attack subdirectory and follows
        the same CSV format as CKJail dataset.

        Expects CSV at <data_dir>/cka-attack/jbb-behaviors_cka_dataset.csv with columns:
        - category
        - source
        - forbidden_prompt

        Returns:
            List[Dict[str, Any]]: List of data samples with keys:
                                - 'query': forbidden_prompt content
                                - 'source': source information
                                - 'category': category classification

        Raises:
            FileNotFoundError: If the dataset file doesn't exist
            ValueError: If required columns are missing
        """
        file_path = os.path.join(
            self.data_dir, "cka-attack", "jbb-behaviors_cka_dataset.csv"
        )

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise Exception(f"Error reading CSV file {file_path}: {str(e)}")

        # Validate required columns
        required_columns = ["category", "source", "forbidden_prompt"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Convert DataFrame to list of dictionaries
        data_list = []
        for _, row in df.iterrows():
            sample = {
                "query": row["forbidden_prompt"],
                "source": row["source"],
                "category": row["category"],
            }
            # Preserve target_str if it exists in the dataset
            if "target_str" in df.columns:
                sample["target_str"] = row["target_str"]
            data_list.append(sample)

        # Store loaded data for statistics
        dataset_key = "jbb_behaviors_cka"
        self.loaded_data[dataset_key] = data_list
        self._compute_statistics(dataset_key, data_list)

        print(
            f"Successfully loaded {len(data_list)} samples from JBB-Behaviors CKA dataset"
        )
        return data_list

    def _load_strongreject_cka(self) -> List[Dict[str, Any]]:
        """
        Load StrongReject CKA dataset.

        This dataset is located in the cka-attack subdirectory and follows
        the same CSV format as CKJail dataset.

        Expects CSV at <data_dir>/cka-attack/strongreject_cka_dataset.csv with columns:
        - category
        - source
        - forbidden_prompt

        Returns:
            List[Dict[str, Any]]: List of data samples with keys:
                                - 'query': forbidden_prompt content
                                - 'source': source information
                                - 'category': category classification

        Raises:
            FileNotFoundError: If the dataset file doesn't exist
            ValueError: If required columns are missing
        """
        file_path = os.path.join(
            self.data_dir, "cka-attack", "strongreject_cka_dataset.csv"
        )

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise Exception(f"Error reading CSV file {file_path}: {str(e)}")

        # Validate required columns
        required_columns = ["category", "source", "forbidden_prompt"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Convert DataFrame to list of dictionaries
        data_list = []
        for _, row in df.iterrows():
            sample = {
                "query": row["forbidden_prompt"],
                "source": row["source"],
                "category": row["category"],
            }
            # Preserve target_str if it exists in the dataset
            if "target_str" in df.columns:
                sample["target_str"] = row["target_str"]
            data_list.append(sample)

        # Store loaded data for statistics
        dataset_key = "strongreject_cka"
        self.loaded_data[dataset_key] = data_list
        self._compute_statistics(dataset_key, data_list)

        print(
            f"Successfully loaded {len(data_list)} samples from StrongReject CKA dataset"
        )
        return data_list

    def _load_generic_cka(
        self, filename: str, dataset_key: str, **kwargs
    ) -> List[Dict[str, Any]]:
        """Load a generic CKA dataset CSV from cka-attack directory."""
        file_path = os.path.join(self.data_dir, "cka-attack", filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        df = pd.read_csv(file_path)
        required_columns = ["category", "source", "forbidden_prompt"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        data_list = []
        for _, row in df.iterrows():
            sample = {
                "query": row["forbidden_prompt"],
                "source": row["source"],
                "category": row["category"],
            }
            if "target_str" in df.columns:
                sample["target_str"] = row["target_str"]
            data_list.append(sample)
        self.loaded_data[dataset_key] = data_list
        self._compute_statistics(dataset_key, data_list)
        print(
            f"Successfully loaded {len(data_list)} samples from {dataset_key} dataset"
        )
        return data_list

    def _load_harmbench_cka(self) -> List[Dict[str, Any]]:
        """
        Load HarmBench CKA dataset.

        This dataset is located in the cka-attack subdirectory and follows
        the same CSV format as other CKA datasets.

        Expects CSV at <data_dir>/cka-attack/harmbench_cka_dataset.csv with columns:
        - category
        - source
        - forbidden_prompt
        - target_str (optional, will be included if present)

        Returns:
            List[Dict[str, Any]]: List of data samples with keys:
                                - 'query': forbidden_prompt content
                                - 'source': source information
                                - 'category': category classification
                                - 'target_str': (optional) target string if present in dataset

        Raises:
            FileNotFoundError: If the dataset file doesn't exist
            ValueError: If required columns are missing
        """
        file_path = os.path.join(
            self.data_dir, "cka-attack", "harmbench_cka_dataset.csv"
        )

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise Exception(f"Error reading CSV file {file_path}: {str(e)}")

        # Validate required columns
        required_columns = ["category", "source", "forbidden_prompt"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Convert DataFrame to list of dictionaries
        data_list = []
        for _, row in df.iterrows():
            sample = {
                "query": row["forbidden_prompt"],
                "source": row["source"],
                "category": row["category"],
            }
            # Preserve target_str if it exists in the dataset
            if "target_str" in df.columns:
                sample["target_str"] = row["target_str"]
            data_list.append(sample)

        # Store loaded data for statistics
        dataset_key = "harmbench_cka"
        self.loaded_data[dataset_key] = data_list
        self._compute_statistics(dataset_key, data_list)

        print(
            f"Successfully loaded {len(data_list)} samples from HarmBench CKA dataset"
        )
        return data_list

    def _load_gpt52gen_cka(self, split: str = "test") -> List[Dict[str, Any]]:
        """
        Load GPT52-gen CKA dataset (from Intent-Defender).

        Expects CSV at <data_dir>/cka-attack/gpt52gen_{split}_cka_dataset.csv
        with columns: category, source, forbidden_prompt.

        Args:
            split: "test" (34 queries) or "all" (214 queries)

        Returns:
            List[Dict[str, Any]]: List of data samples
        """
        file_path = os.path.join(
            self.data_dir, "cka-attack", f"gpt52gen_{split}_cka_dataset.csv"
        )

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise Exception(f"Error reading CSV file {file_path}: {str(e)}")

        required_columns = ["category", "source", "forbidden_prompt"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        data_list = []
        for _, row in df.iterrows():
            sample = {
                "query": row["forbidden_prompt"],
                "source": row["source"],
                "category": row["category"],
            }
            if "target_str" in df.columns:
                sample["target_str"] = row["target_str"]
            data_list.append(sample)

        dataset_key = f"gpt52gen_cka_{split}"
        self.loaded_data[dataset_key] = data_list
        self._compute_statistics(dataset_key, data_list)

        print(
            f"Successfully loaded {len(data_list)} samples from GPT52-gen CKA dataset ({split})"
        )
        return data_list

    def _load_strongreject(self, dataset_size: str = "large") -> List[Dict[str, Any]]:
        """
        Load StrongReject dataset.

        Args:
            dataset_size (str): Size of dataset to load ('large' or 'small')
                              'large' loads strongreject_dataset.csv
                              'small' loads strongreject_small_dataset.csv

        Returns:
            List[Dict[str, Any]]: List of data samples with keys:
                                 - 'query': forbidden_prompt content
                                 - 'source': source information
                                 - 'category': category classification

        Raises:
            FileNotFoundError: If the specified dataset file doesn't exist
            ValueError: If dataset_size is not 'large' or 'small'
        """
        # Validate dataset_size parameter
        if dataset_size not in ["large", "small"]:
            raise ValueError("dataset_size must be 'large' or 'small'")

        # Determine filename based on dataset size
        if dataset_size == "large":
            filename = "strongreject_dataset.csv"
        else:  # dataset_size == "small"
            filename = "strongreject_small_dataset.csv"

        # Construct full file path
        file_path = os.path.join(self.data_dir, "strongreject", filename)

        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Load CSV file
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise Exception(f"Error reading CSV file {file_path}: {str(e)}")

        # Validate required columns
        required_columns = ["category", "source", "forbidden_prompt"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Convert DataFrame to list of dictionaries with renamed keys
        data_list = []
        for _, row in df.iterrows():
            sample = {
                "query": row["forbidden_prompt"],
                "source": row["source"],
                "category": row["category"],
            }
            # If the dataset has a target_str column, preserve it for the PAIR method
            if "target_str" in df.columns:
                sample["target_str"] = row["target_str"]
            data_list.append(sample)

        # Store loaded data for statistics
        dataset_key = f"strongreject_{dataset_size}"
        self.loaded_data[dataset_key] = data_list

        # Compute and store statistics
        self._compute_statistics(dataset_key, data_list)

        print(
            f"Successfully loaded {len(data_list)} samples from StrongReject ({dataset_size}) dataset"
        )

        return data_list

    def _load_ckjail(self) -> List[Dict[str, Any]]:
        """
        Load CKJail dataset.

        Expects a single CSV at <data_dir>/ckjail/ckjail_dataset.csv with columns:
          - category
          - source
          - forbidden_prompt

        Returns:
            List[Dict[str, Any]]: List of data samples with keys:
                                 - 'query'
                                 - 'source'
                                 - 'category'
        """
        file_path = os.path.join(self.data_dir, "ckjail", "ckjail_dataset.csv")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise Exception(f"Error reading CSV file {file_path}: {str(e)}")

        required_columns = ["category", "source", "forbidden_prompt"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        data_list = []
        for _, row in df.iterrows():
            sample = {
                "query": row["forbidden_prompt"],
                "source": row["source"],
                "category": row["category"],
            }
            if "target_str" in df.columns:
                sample["target_str"] = row["target_str"]
            data_list.append(sample)

        dataset_key = "ckjail"
        self.loaded_data[dataset_key] = data_list
        self._compute_statistics(dataset_key, data_list)

        print(f"Successfully loaded {len(data_list)} samples from CKJail dataset")
        return data_list

    def _compute_statistics(self, dataset_key: str, data: List[Dict[str, Any]]) -> None:
        """
        Compute statistics for the loaded dataset.

        Args:
            dataset_key (str): Key to identify the dataset
            data (List[Dict[str, Any]]): Loaded data samples
        """
        if not data:
            self.dataset_stats[dataset_key] = {}
            return

        # Basic statistics
        total_samples = len(data)

        # Category distribution
        categories = [sample["category"] for sample in data]
        category_counts = Counter(categories)

        # Source distribution
        sources = [sample["source"] for sample in data]
        source_counts = Counter(sources)

        # Query length statistics
        query_lengths = [len(sample["query"]) for sample in data]
        avg_query_length = sum(query_lengths) / len(query_lengths)
        min_query_length = min(query_lengths)
        max_query_length = max(query_lengths)

        # Store statistics
        self.dataset_stats[dataset_key] = {
            "total_samples": total_samples,
            "category_distribution": dict(category_counts),
            "source_distribution": dict(source_counts),
            "query_length_stats": {
                "average": round(avg_query_length, 2),
                "minimum": min_query_length,
                "maximum": max_query_length,
            },
        }

    def print_statistics(self, dataset_key: Optional[str] = None) -> None:
        """
        Print statistics for loaded datasets.

        Args:
            dataset_key (str, optional): Specific dataset to print stats for.
                                       If None, prints stats for all loaded datasets.
        """
        if dataset_key:
            # Print statistics for specific dataset
            if dataset_key not in self.dataset_stats:
                print(f"No statistics available for dataset: {dataset_key}")
                return

            self._print_dataset_stats(dataset_key, self.dataset_stats[dataset_key])
        else:
            # Print statistics for all loaded datasets
            if not self.dataset_stats:
                print("No datasets loaded yet.")
                return

            for key, stats in self.dataset_stats.items():
                self._print_dataset_stats(key, stats)
                print("-" * 60)  # Separator between datasets

    def _print_dataset_stats(self, dataset_key: str, stats: Dict[str, Any]) -> None:
        """
        Print statistics for a single dataset.

        Args:
            dataset_key (str): Dataset identifier
            stats (Dict[str, Any]): Statistics dictionary
        """
        print(f"\n=== Dataset Statistics: {dataset_key} ===")
        print(f"Total Samples: {stats.get('total_samples', 0)}")

        # Category distribution
        print(f"\nCategory Distribution:")
        category_dist = stats.get("category_distribution", {})
        for category, count in sorted(category_dist.items()):
            percentage = (
                (count / stats["total_samples"]) * 100
                if stats.get("total_samples", 0)
                else 0.0
            )
            print(f"  {category}: {count} ({percentage:.1f}%)")

        # Source distribution
        print(f"\nSource Distribution:")
        source_dist = stats.get("source_distribution", {})
        for source, count in sorted(source_dist.items()):
            percentage = (
                (count / stats["total_samples"]) * 100
                if stats.get("total_samples", 0)
                else 0.0
            )
            print(f"  {source}: {count} ({percentage:.1f}%)")

        # Query length statistics
        query_stats = stats.get("query_length_stats", {})
        if query_stats:
            print(f"\nQuery Length Statistics:")
            print(f"  Average: {query_stats.get('average', 0)} characters")
            print(f"  Minimum: {query_stats.get('minimum', 0)} characters")
            print(f"  Maximum: {query_stats.get('maximum', 0)} characters")

    def get_loaded_datasets(self) -> List[str]:
        """
        Get list of loaded dataset keys.

        Returns:
            List[str]: List of loaded dataset identifiers
        """
        return list(self.loaded_data.keys())

    def get_dataset(self, dataset_key: str) -> List[Dict[str, Any]]:
        """
        Get loaded dataset by key.

        Args:
            dataset_key (str): Dataset identifier

        Returns:
            List[Dict[str, Any]]: Dataset samples

        Raises:
            KeyError: If dataset_key is not found
        """
        if dataset_key not in self.loaded_data:
            raise KeyError(
                f"Dataset '{dataset_key}' not loaded. "
                f"Available datasets: {list(self.loaded_data.keys())}"
            )

        return self.loaded_data[dataset_key]


# # Test code
# if __name__ == "__main__":
#     print("=== Testing DataLoader ===")

#     # Initialize DataLoader
#     dataloader = DataLoader(data_dir="data/datasets")

#     # 1) Load StrongReject small
#     print("\n1. Loading StrongReject small dataset...")
#     try:
#         small_data = dataloader.load("strongreject", dataset_size="small")
#         print(f"Loaded {len(small_data)} samples")

#         # Print first few samples
#         print("\nFirst 3 samples:")
#         for i, sample in enumerate(small_data[:3]):
#             print(f"Sample {i+1}:")
#             print(f"  Query: {sample['query'][:100]}...")  # First 100 chars
#             print(f"  Source: {sample['source']}")
#             print(f"  Category: {sample['category']}")
#             print()
#     except Exception as e:
#         print(f"Error loading small dataset: {e}")

#     # 2) Load StrongReject large
#     print("\n2. Loading StrongReject large dataset...")
#     try:
#         large_data = dataloader.load("strongreject", dataset_size="large")
#         print(f"Loaded {len(large_data)} samples")
#     except Exception as e:
#         print(f"Error loading large dataset: {e}")

#     # 3) Load CKJail dataset
#     print("\n3. Loading CKJail dataset...")
#     try:
#         ckjail_data = dataloader.load("ckjail")
#         print(f"Loaded {len(ckjail_data)} samples")
#     except Exception as e:
#         print(f"Error loading CKJail dataset: {e}")

#     # 3.5) Load JBB-Behaviors CKA dataset
#     print("\n3.5. Loading JBB-Behaviors CKA dataset...")
#     try:
#         jbb_cka_data = dataloader.load("jbb_behaviors_cka")
#         print(f"Loaded {len(jbb_cka_data)} samples")
#     except Exception as e:
#         print(f"Error loading JBB-Behaviors CKA dataset: {e}")

#     # 3.6) Load StrongReject CKA dataset
#     print("\n3.6. Loading StrongReject CKA dataset...")
#     try:
#         strongreject_cka_data = dataloader.load("strongreject_cka")
#         print(f"Loaded {len(strongreject_cka_data)} samples")
#     except Exception as e:
#         print(f"Error loading StrongReject CKA dataset: {e}")

#     # 3.7) Load HarmBench CKA dataset
#     print("\n3.7. Loading HarmBench CKA dataset...")
#     try:
#         harmbench_cka_data = dataloader.load("harmbench_cka")
#         print(f"Loaded {len(harmbench_cka_data)} samples")

#         # Print first sample to verify structure
#         if harmbench_cka_data:
#             print("\nFirst sample from HarmBench:")
#             sample = harmbench_cka_data[0]
#             print(f"  Query: {sample['query'][:80]}...")
#             print(f"  Source: {sample['source']}")
#             print(f"  Category: {sample['category']}")
#             if 'target_str' in sample:
#                 print(f"  Target_str: Present")
#             else:
#                 print(f"  Target_str: Not present (will be added later)")
#     except Exception as e:
#         print(f"Error loading HarmBench CKA dataset: {e}")

#     # 4) Print statistics
#     print("\n4. Dataset Statistics:")
#     dataloader.print_statistics()

#     # 5) Plot distributions for each loaded dataset (save to <project>/data/plot)
#     print("\n5. Saving distributions (category & source) to image files...")

#     try:
#         import matplotlib
#         matplotlib.use("Agg")  # Use a non-interactive backend (safe for headless environments)
#         import matplotlib.pyplot as plt
#         import re

#         # Resolve "<data>/plot" where "<data>" is the parent of dataloader.data_dir
#         # If dataloader.data_dir == "data/datasets", this becomes "data/plot".
#         data_root = os.path.dirname(os.path.normpath(dataloader.data_dir))
#         plot_dir = os.path.join(data_root, "plot")
#         os.makedirs(plot_dir, exist_ok=True)

#         def _sanitize(name: str) -> str:
#             """Sanitize a string so it is safe to use in filenames."""
#             name = re.sub(r"[^\w\-]+", "_", name.strip())
#             return name.strip("_") or "untitled"

#         def save_distribution_plot(dist: Dict[str, int], title: str, xlabel: str, outfile: str):
#             """
#             Save a simple bar chart for a distribution dictionary to a PNG file.

#             Args:
#                 dist: Mapping of label -> count.
#                 title: Plot title.
#                 xlabel: X-axis label.
#                 outfile: Full output path for the PNG file.
#             """
#             if not dist:
#                 print(f"[plot] No data for {title}; skipping.")
#                 return

#             # Sort by count (desc) so the bars are ordered by frequency
#             items = sorted(dist.items(), key=lambda x: x[1], reverse=True)
#             labels = [k for k, _ in items]
#             values = [v for _, v in items]

#             plt.figure()
#             plt.bar(labels, values)  # Do not specify colors per your constraints
#             plt.title(title)
#             plt.xlabel(xlabel)
#             plt.ylabel("Count")
#             plt.xticks(rotation=45, ha="right")
#             plt.tight_layout()
#             plt.savefig(outfile, dpi=150)
#             plt.close()
#             print(f"[plot] Saved: {outfile}")

#         # Generate and save two plots per loaded dataset: category and source distributions
#         for ds_key, stats in dataloader.dataset_stats.items():
#             cat_dist = stats.get("category_distribution", {})
#             src_dist = stats.get("source_distribution", {})

#             safe_ds = _sanitize(ds_key)
#             cat_out = os.path.join(plot_dir, f"{safe_ds}_category_distribution.png")
#             src_out = os.path.join(plot_dir, f"{safe_ds}_source_distribution.png")

#             save_distribution_plot(
#                 cat_dist,
#                 title=f"{ds_key} - Category Distribution",
#                 xlabel="Category",
#                 outfile=cat_out
#             )
#             save_distribution_plot(
#                 src_dist,
#                 title=f"{ds_key} - Source Distribution",
#                 xlabel="Source",
#                 outfile=src_out
#             )

#     except Exception as e:
#         print(f"Plotting error: {e}")

#     # 6) Loaded datasets
#     print(f"\n6. Loaded datasets: {dataloader.get_loaded_datasets()}")

#     # 7) Error handling tests
#     print("\n7. Testing error handling...")
#     try:
#         dataloader.load("nonexistent_dataset")
#     except ValueError as e:
#         print(f"Expected error: {e}")

#     try:
#         dataloader.load("strongreject", dataset_size="invalid")
#     except ValueError as e:
#         print(f"Expected error: {e}")

#     print("\n=== Testing Complete ===")
