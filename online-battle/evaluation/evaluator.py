import os
import logging
import json
from typing import Dict, Any
from evaluation.evaluator_factory import create_metric


class Evaluator:
    """
    Minimal evaluator: load configured metrics via the factory,
    then hand the full list of method results to each metric's .evaluate().
    """

    def __init__(self, config: Dict[str, Any], output_dir: str):
        self.config = config or {}
        self.output_dir = output_dir
        self.logger = logging.getLogger("Evaluator")
        os.makedirs(output_dir, exist_ok=True)

        # Build metrics from config
        metric_names = self.config.get("metrics", [])
        if not metric_names:
            raise ValueError("No metrics specified under evaluation.metrics")

        # Get judge models configuration
        judge_models = self.config.get("judge_models", {})

        self.metrics = {}
        for mname in metric_names:
            metric_cfg = {}

            # Pass appropriate judge model config to each metric
            if mname == "harmful_rate":
                # Pass Llama Guard config
                llama_guard_cfg = judge_models.get("llama_guard", {})
                metric_cfg = {
                    "judge_model": llama_guard_cfg.get(
                        "name", "meta-llama/Llama-Guard-3-8B"
                    ),
                    "device": llama_guard_cfg.get("device_map", "auto"),
                    "hf_token": llama_guard_cfg.get("hf_token"),
                    "use_vllm": llama_guard_cfg.get("use_vllm", False),
                    "batch_size": llama_guard_cfg.get("batch_size", 1),
                    "vllm_kwargs": llama_guard_cfg.get("vllm_kwargs", {}),
                    "max_new_tokens": 12,  # Keep short for SAFE/UNSAFE
                    "max_length": llama_guard_cfg.get(
                        "max_length", 2048
                    ),  # Add max_length config
                }
            elif mname == "attack_success_rate":
                # Pass entire judge_models config (contains llama_judge)
                metric_cfg = judge_models

            metric = create_metric(mname, metric_cfg)
            self.metrics[mname] = metric

        self.logger.info(
            "Evaluator initialized with metrics: %s", ", ".join(self.metrics.keys())
        )

    def get_available_metrics(self):
        return list(self.metrics.keys())

    def evaluate_dataset_results(
        self, dataset_key: str, dataset_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        self.logger.info(f"Evaluating results for dataset: {dataset_key}")

        evaluation_results: Dict[str, Any] = {}

        for method_name, method_payload in dataset_results.items():
            if method_name == "evaluation":
                continue

            # Handle both dict and list formats
            if isinstance(method_payload, list):
                # Direct list format: results are the list itself
                results = method_payload
                top_err = None
            else:
                # Dict format: extract error and results
                top_err = method_payload.get("error")
                results = method_payload.get("results", [])

            if top_err:
                self.logger.warning(
                    f"Skipping {method_name} due to top-level error: {top_err}"
                )
                continue
            n = len(results)
            if n == 0:
                self.logger.warning(
                    f"{method_name} has 0 results → no evaluation will run"
                )
                evaluation_results[method_name] = {"error": "No results to evaluate"}
                continue

            self.logger.info(f"Evaluating method: {method_name} ; results={n}")

            method_eval = {}
            for metric_name, metric in self.metrics.items():
                try:
                    metric_out = metric.evaluate(results)
                    method_eval[metric_name] = metric_out
                    # ---- print metric & result in logs ----
                    self.logger.info(
                        f"[{dataset_key}::{method_name}] metric={metric_name} result="
                        + json.dumps(metric_out, ensure_ascii=False)
                    )

                    # Optimization: release model memory immediately after each metric evaluation
                    self._teardown_metric_model(metric_name, metric)

                except Exception as e:
                    self.logger.error(
                        f"Metric {metric_name} failed on {method_name}: {e}"
                    )
                    method_eval[metric_name] = {"error": str(e)}

            evaluation_results[method_name] = method_eval

        return evaluation_results

    def _teardown_metric_model(self, metric_name: str, metric) -> None:
        """
        Release metric model resources to free GPU memory after evaluation.
        Similar to _teardown_whitebox_resources but for judge models.
        """
        try:
            print(f"[DEBUG] Releasing {metric_name} model to free GPU memory")

            # Call metric's teardown method (if available)
            if hasattr(metric, "teardown") and callable(getattr(metric, "teardown")):
                try:
                    metric.teardown()
                    print(f"[DEBUG] {metric_name} teardown method called successfully")
                except Exception as e:
                    print(f"[DEBUG] {metric_name} teardown method failed: {e}")

            # Manually release model resources
            if hasattr(metric, "judge_model") and hasattr(metric.judge_model, "model"):
                # harmful_rate: model is stored in judge_model
                if (
                    hasattr(metric.judge_model, "model")
                    and metric.judge_model.model is not None
                ):
                    metric.judge_model.model = None
                if (
                    hasattr(metric.judge_model, "tokenizer")
                    and metric.judge_model.tokenizer is not None
                ):
                    metric.judge_model.tokenizer = None
                if (
                    hasattr(metric.judge_model, "vllm_model")
                    and metric.judge_model.vllm_model is not None
                ):
                    metric.judge_model.vllm_model = None
                print(f"[DEBUG] {metric_name} judge_model resources released")

            elif hasattr(metric, "model") and hasattr(metric, "tokenizer"):
                # attack_success_rate: model is stored directly in metric
                if hasattr(metric, "model") and metric.model is not None:
                    metric.model = None
                if hasattr(metric, "tokenizer") and metric.tokenizer is not None:
                    metric.tokenizer = None
                print(f"[DEBUG] {metric_name} model resources released")

            # Force garbage collection
            import gc
            import torch

            gc.collect()
            torch.cuda.empty_cache()
            print(f"[DEBUG] {metric_name} model released, GPU memory freed")

        except Exception as e:
            print(f"[DEBUG] Error releasing {metric_name} model: {e}")

    def save_evaluation_results(
        self, results: Dict[str, Any], filename: str = "evaluation_results.json"
    ):
        """Optional persistence helper."""
        from utils.utils import save_results

        path = os.path.join(self.output_dir, filename)
        save_results(results, path)
        self.logger.info(f"Evaluation results saved to: {path}")
