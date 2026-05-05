import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    logger_name: str = "JailbreakExperiment",
) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file (str, optional): Path to log file
        logger_name (str): Name of the logger

    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear any existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # More detailed for file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def create_output_dirs(base_output_dir: str) -> None:
    """
    Create output directory structure.

    Args:
        base_output_dir (str): Base output directory path
    """
    # Create main directories
    directories = [
        base_output_dir,
        os.path.join(base_output_dir, "evaluation"),
        os.path.join(base_output_dir, "logs"),
        os.path.join(base_output_dir, "figures"),
        os.path.join(base_output_dir, "raw_results"),
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def save_results(results: Dict[str, Any], output_file: str) -> None:
    """
    Save experimental results to JSON file.

    Args:
        results (Dict[str, Any]): Results dictionary
        output_file (str): Path to output file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Add metadata
    results_with_metadata = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "version": "1.0",
            "total_datasets": len(results),
        },
        "results": results,
    }

    # Save to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_with_metadata, f, indent=2, ensure_ascii=False)


def load_results(input_file: str) -> Dict[str, Any]:
    """
    Load experimental results from JSON file.

    Args:
        input_file (str): Path to input file

    Returns:
        Dict[str, Any]: Loaded results dictionary
    """
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Return just the results if it has metadata wrapper
    if "results" in data and "metadata" in data:
        return data["results"]
    else:
        return data


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate experiment configuration.

    Args:
        config (Dict[str, Any]): Configuration dictionary

    Returns:
        bool: True if configuration is valid
    """
    required_sections = ["experiment", "data", "model", "methods"]

    for section in required_sections:
        if section not in config:
            print(f"Missing required configuration section: {section}")
            return False

    # Validate experiment section
    experiment = config["experiment"]
    if "name" not in experiment:
        print("Missing experiment name")
        return False

    # Validate data section
    data = config["data"]
    if "datasets" not in data or not data["datasets"]:
        print("No datasets specified")
        return False

    # Validate model section
    model = config["model"]
    if "type" not in model:
        print("Missing model type")
        return False

    model_type = model["type"].lower()
    if model_type not in ["whitebox", "blackbox"]:
        print(f"Invalid model type: {model_type}")
        return False

    # Check if model config exists for the specified type
    if model_type not in model:
        print(f"Missing configuration for model type: {model_type}")
        return False

    return True


def format_time(seconds: float) -> str:
    """
    Format time duration in human-readable format.

    Args:
        seconds (float): Time in seconds

    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to specified length.

    Args:
        text (str): Text to truncate
        max_length (int): Maximum length
        suffix (str): Suffix to add if truncated

    Returns:
        str: Truncated text
    """
    if len(text) <= max_length:
        return text
    else:
        return text[: max_length - len(suffix)] + suffix


def calculate_success_rate(results: List[Dict[str, Any]]) -> float:
    """
    Calculate success rate from results.

    Args:
        results (List[Dict[str, Any]]): List of result dictionaries

    Returns:
        float: Success rate (0.0 to 1.0)
    """
    if not results:
        return 0.0

    successful = sum(1 for result in results if result.get("success", False))
    return successful / len(results)


def print_summary_stats(results: Dict[str, Any]) -> None:
    """
    Print summary statistics for experimental results.

    Args:
        results (Dict[str, Any]): Results dictionary
    """
    print("\n=== Experimental Results Summary ===")

    for dataset_key, dataset_results in results.items():
        if dataset_key == "metadata":  # Skip metadata
            continue

        print(f"\nDataset: {dataset_key}")
        print("-" * 40)

        for method_name, method_results in dataset_results.items():
            if method_name == "evaluation":  # Skip evaluation results for now
                continue

            if isinstance(method_results, dict):
                if "error" in method_results:
                    print(f"  {method_name}: ERROR - {method_results['error']}")
                elif "num_samples" in method_results:
                    print(
                        f"  {method_name}: {method_results['num_samples']} samples processed"
                    )
                else:
                    print(f"  {method_name}: Completed")


# Test code
if __name__ == "__main__":
    print("=== Testing Utils ===")

    # Test logging setup
    logger = setup_logging("DEBUG", "test.log", "TestLogger")
    logger.info("Test logging message")

    # Test directory creation
    test_dir = "test_output"
    create_output_dirs(test_dir)
    print(f"Created directories under: {test_dir}")

    # Test results saving/loading
    test_results = {
        "dataset1": {
            "method1": {"success": True, "response": "test response"},
            "method2": {"success": False, "error": "test error"},
        }
    }

    results_file = os.path.join(test_dir, "test_results.json")
    save_results(test_results, results_file)
    print(f"Saved results to: {results_file}")

    loaded_results = load_results(results_file)
    print(f"Loaded results: {loaded_results == test_results}")

    # Test config validation
    test_config = {
        "experiment": {"name": "test"},
        "data": {"datasets": [{"name": "test_dataset"}]},
        "model": {"type": "whitebox", "whitebox": {"name": "test_model"}},
        "methods": {"baselines": []},
    }

    print(f"Config validation: {validate_config(test_config)}")

    # Test utility functions
    print(f"Format time (150s): {format_time(150)}")
    print(
        f"Truncate text: {truncate_text('This is a very long text that should be truncated', 20)}"
    )

    test_method_results = [{"success": True}, {"success": False}, {"success": True}]
    print(f"Success rate: {calculate_success_rate(test_method_results):.2%}")

    # Clean up
    import shutil

    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    if os.path.exists("test.log"):
        os.remove("test.log")

    print("=== Testing Complete ===")
