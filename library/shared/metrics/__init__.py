"""Metrics: task accuracy (per-domain), distribution metrics, generation metrics.

`TaskMetric` is the unified interface the old `*AccuracyCalculator` classes lacked.
Distribution/generation helpers are pure functions so they can be called inside the
eval pass (compute-once principle) and unit-tested without a model.
"""

from .base import MetricResult, TaskMetric
from .task import ClassificationMetric, MathMetric, PostGenerationMetric

__all__ = [
    "MetricResult",
    "TaskMetric",
    "ClassificationMetric",
    "MathMetric",
    "PostGenerationMetric",
]
