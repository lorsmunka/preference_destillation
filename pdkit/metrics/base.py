"""Unified task-metric interface.

The old calculators (`ClassificationAccuracyCalculator`, `MathAccuracyCalculator`,
`PostGenerationAccuracyCalculator`) shared an implicit interface but no base type, and
`get_confusion_matrices()` returned three incompatible shapes. `TaskMetric` formalizes it
and `MetricResult` gives one shape with optional fields.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MetricResult:
    """Result of a task metric over an eval pass.

    `accuracy` is the headline number (task accuracy). `validity_rate` is the fraction of
    examples whose output was structurally valid (None for free-form domains that have no
    structural notion). `confusion` is per-category {true: {pred: count}} when meaningful.
    """

    name: str
    accuracy: float
    correct: int
    total: int
    examples: int
    validity_rate: Optional[float] = None
    confusion: Optional[Dict[str, Dict[str, Dict[str, int]]]] = None
    extra: Dict[str, float] = field(default_factory=dict)


class TaskMetric(ABC):
    """Accumulate per-example predictions, then report a `MetricResult`."""

    name: str = "task"

    @abstractmethod
    def update(self, predicted_tokens: List[str], ground_truth_response: str) -> None:
        ...

    @abstractmethod
    def result(self) -> MetricResult:
        ...
