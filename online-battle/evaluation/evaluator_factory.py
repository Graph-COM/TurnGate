from typing import Dict, Any

# Import metric implementations
from evaluation.metrics.harmful_rate import HarmfulRateMetric
from evaluation.metrics.attack_success_rate import AttackSuccessRateMetric


def create_metric(metric_name: str, config: Dict[str, Any] | None = None):
    """
    Factory that returns a metric instance by name.
    Each metric implements a uniform interface: .name and .evaluate(results) -> dict

    Args:
        metric_name: Name of the metric to create
        config: Configuration dictionary for the metric

    Returns:
        Metric instance
    """
    name = (metric_name or "").lower()

    if name == "harmful_rate":
        return HarmfulRateMetric(config or {})

    if name == "attack_success_rate":
        return AttackSuccessRateMetric(config or {})

    raise ValueError(
        f"Unsupported metric: {metric_name}. " f"Supported: {list_available_metrics()}"
    )


def list_available_metrics() -> list:
    """List all available metrics."""
    return ["harmful_rate", "attack_success_rate"]
