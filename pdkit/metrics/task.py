"""Per-domain task metrics (ported from shared/*_accuracy.py, behind one interface).

Fixes a real bug in the old classification calculator: when the student produced
unparseable JSON it returned (0, 0) and the example was *dropped from the denominator*,
overstating task accuracy. Here an invalid structured output counts its ground-truth
fields as wrong, and the structural validity rate is reported separately.
"""

import json
import re
from collections import Counter
from typing import Dict, List, Optional

from .base import MetricResult, TaskMetric

CATEGORIES = ["tone", "sentiment", "safety", "toxicity"]

CATEGORY_VALUES = {
    "tone": ["aggressive", "rude", "neutral", "polite", "friendly"],
    "sentiment": ["negative", "neutral", "positive"],
    "safety": ["harmful", "safe"],
    "toxicity": ["toxic", "respectful"],
}

INVALID_LABEL = "<invalid>"


def _tokens_to_text(tokens: List[str]) -> str:
    return "".join(tokens).replace("▁", " ")


class ClassificationMetric(TaskMetric):
    name = "task_accuracy"

    def __init__(self) -> None:
        self.correct = 0
        self.total = 0
        self.examples = 0
        self.invalid = 0
        self.confusion = {category: Counter() for category in CATEGORIES}

    def update(self, predicted_tokens: List[str], ground_truth_response: str) -> None:
        predicted = self._parse(_tokens_to_text(predicted_tokens))
        ground_truth = self._parse(ground_truth_response)

        if ground_truth is None:
            return  # cannot score against an unparseable reference

        self.examples += 1

        if predicted is None:
            # Structural failure: count every ground-truth field as wrong (was silently dropped).
            self.invalid += 1
            for category in CATEGORIES:
                if category in ground_truth:
                    self.total += 1
                    self.confusion[category][(ground_truth[category], INVALID_LABEL)] += 1
            return

        for category in CATEGORIES:
            if category in ground_truth:
                self.total += 1
                predicted_value = predicted.get(category, "")
                self.confusion[category][(ground_truth[category], predicted_value)] += 1
                if predicted_value == ground_truth[category]:
                    self.correct += 1

    def result(self) -> MetricResult:
        accuracy = self.correct / self.total if self.total else 0.0
        validity_rate = (self.examples - self.invalid) / self.examples if self.examples else None
        return MetricResult(
            name=self.name,
            accuracy=accuracy,
            correct=self.correct,
            total=self.total,
            examples=self.examples,
            validity_rate=validity_rate,
            confusion=self._confusion_matrices(),
        )

    def _parse(self, text: str) -> Optional[Dict[str, str]]:
        text = text.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None
        result = {}
        for category in CATEGORIES:
            if category in data:
                result[category] = str(data[category]).lower()
        return result or None

    def _confusion_matrices(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        matrices: Dict[str, Dict[str, Dict[str, int]]] = {}
        for category in CATEGORIES:
            values = CATEGORY_VALUES[category]
            matrix = {}
            for true_value in values:
                row = {pred: self.confusion[category].get((true_value, pred), 0) for pred in values}
                row[INVALID_LABEL] = self.confusion[category].get((true_value, INVALID_LABEL), 0)
                matrix[true_value] = row
            matrices[category] = matrix
        return matrices


class MathMetric(TaskMetric):
    name = "task_accuracy"

    def __init__(self) -> None:
        self.correct = 0
        self.examples = 0
        self.invalid = 0

    def update(self, predicted_tokens: List[str], ground_truth_response: str) -> None:
        predicted_solution = self._extract_solution(_tokens_to_text(predicted_tokens))
        ground_truth_solution = self._extract_solution(ground_truth_response)

        if ground_truth_solution is None:
            return

        self.examples += 1
        if predicted_solution is None:
            self.invalid += 1
            return
        if predicted_solution == ground_truth_solution:
            self.correct += 1

    def result(self) -> MetricResult:
        accuracy = self.correct / self.examples if self.examples else 0.0
        validity_rate = (self.examples - self.invalid) / self.examples if self.examples else None
        return MetricResult(
            name=self.name,
            accuracy=accuracy,
            correct=self.correct,
            total=self.examples,
            examples=self.examples,
            validity_rate=validity_rate,
        )

    def _extract_solution(self, text: str) -> Optional[str]:
        match = re.search(r"Solution:\s*([^;\n]+)", text)
        return match.group(1).strip() if match else None


class PostGenerationMetric(TaskMetric):
    """Free-form: task accuracy == structural completion rate (does `<end>` appear)."""

    name = "task_accuracy"

    def __init__(self, end_marker: str = "<end>") -> None:
        self.end_marker = end_marker
        self.complete = 0
        self.examples = 0

    def update(self, predicted_tokens: List[str], ground_truth_response: str) -> None:
        self.examples += 1
        if self.end_marker in _tokens_to_text(predicted_tokens):
            self.complete += 1

    def result(self) -> MetricResult:
        accuracy = self.complete / self.examples if self.examples else 0.0
        return MetricResult(
            name=self.name,
            accuracy=accuracy,
            correct=self.complete,
            total=self.examples,
            examples=self.examples,
            validity_rate=None,  # free-form: no structural-validity notion
        )
