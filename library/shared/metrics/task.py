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

INVALID_LABEL = "<invalid>"   # the output did not parse at all
OTHER_LABEL = "<other>"       # the output parsed, but this field's value is outside its label set (or missing)

BOOLEAN_ANSWERS = ("True", "False")


def _tokens_to_text(tokens: List[str]) -> str:
    return "".join(tokens).replace("▁", " ")


def answer_confusion_matrix(pair_counts: Counter) -> Dict[str, Dict[str, int]]:
    """{true answer: {predicted answer: count}} from (true, predicted) counts: the sparse
    confusion matrix of an open-answer metric (only the pairs that occurred)."""
    matrix: Dict[str, Dict[str, int]] = {}
    for (true_answer, predicted_answer), count in sorted(pair_counts.items()):
        matrix.setdefault(true_answer, {})[predicted_answer] = count
    return matrix


def answer_type_accuracy(confusion_matrices: Dict[str, Dict[str, Dict[str, int]]]) -> Dict[str, Optional[float]]:
    """Math: task accuracy split by answer type, read off the "solution" confusion matrix:
    the true/false questions (true answer True or False) and the numeric ones, plus the
    accuracy of always giving the more common true/false value on the true/false questions.
    {} when there is no "solution" matrix (eval logs from before MathMetric kept one)."""
    matrix = (confusion_matrices or {}).get("solution")
    if not matrix:
        return {}
    correct = {"boolean": 0, "numeric": 0}
    total = {"boolean": 0, "numeric": 0}
    for true_answer, predictions in matrix.items():
        answer_type = "boolean" if true_answer in BOOLEAN_ANSWERS else "numeric"
        correct[answer_type] += predictions.get(true_answer, 0)
        total[answer_type] += sum(predictions.values())
    boolean_rows = [sum(matrix.get(answer, {}).values()) for answer in BOOLEAN_ANSWERS]
    return {
        "boolean_examples": total["boolean"],
        "boolean_accuracy": correct["boolean"] / total["boolean"] if total["boolean"] else None,
        "boolean_majority_baseline": max(boolean_rows) / total["boolean"] if total["boolean"] else None,
        "numeric_examples": total["numeric"],
        "numeric_accuracy": correct["numeric"] / total["numeric"] if total["numeric"] else None,
    }


class ClassificationMetric(TaskMetric):
    """Closed-field JSON classification. `categories` / `values` default to the reddit
    sentiment fields; a domain with extra closed fields passes its own."""
    name = "task_accuracy"

    def __init__(self, categories: Optional[List[str]] = None,
                 values: Optional[Dict[str, List[str]]] = None) -> None:
        self.categories = list(categories or CATEGORIES)
        self.values = dict(values or CATEGORY_VALUES)
        self.correct = 0
        self.total = 0
        self.examples = 0
        self.invalid = 0
        self.confusion = {category: Counter() for category in self.categories}

    def update(self, predicted_tokens: List[str], ground_truth_response: str) -> None:
        predicted = self._parse(_tokens_to_text(predicted_tokens))
        ground_truth = self._parse(ground_truth_response)

        if ground_truth is None:
            return  # cannot score against an unparseable reference

        self.examples += 1

        if predicted is None:
            # Structural failure: count every ground-truth field as wrong (was silently dropped).
            self.invalid += 1
            for category in self.categories:
                if category in ground_truth:
                    self.total += 1
                    self.confusion[category][(ground_truth[category], INVALID_LABEL)] += 1
            return

        for category in self.categories:
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
        if not isinstance(data, dict):   # valid JSON without fields: '"tone"', '5', 'null', '[1]'
            return None
        result = {}
        for category in self.categories:
            if category in data:
                result[category] = str(data[category]).lower()
        return result or None

    def _confusion_matrices(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        """Per field {true label: {predicted label: count}} over the field's label set, plus
        the <other> and <invalid> columns, so every scored prediction is counted."""
        matrices: Dict[str, Dict[str, Dict[str, int]]] = {}
        for category in self.categories:
            values = self.values[category]
            matrix = {}
            for true_value in values:
                row = {pred: self.confusion[category].get((true_value, pred), 0) for pred in values}
                row[OTHER_LABEL] = sum(count for (truth, predicted), count in self.confusion[category].items()
                                       if truth == true_value and predicted not in values and predicted != INVALID_LABEL)
                row[INVALID_LABEL] = self.confusion[category].get((true_value, INVALID_LABEL), 0)
                matrix[true_value] = row
            matrices[category] = matrix
        return matrices


class MathMetric(TaskMetric):
    """`Solution: X` must match exactly. The confusion matrix has one field, "solution"
    ({true answer: {predicted answer: count}}, only the pairs that occur), so an eval log
    carries the majority baseline (always giving the most common answer) and the split into
    true/false and numeric questions (answer_type_accuracy, also in `extra`)."""
    name = "task_accuracy"

    def __init__(self) -> None:
        self.correct = 0
        self.examples = 0
        self.invalid = 0
        self.confusion = Counter()   # (true answer, predicted answer or <invalid>) -> count

    def update(self, predicted_tokens: List[str], ground_truth_response: str) -> None:
        predicted_solution = self._extract_solution(_tokens_to_text(predicted_tokens))
        ground_truth_solution = self._extract_solution(ground_truth_response)

        if ground_truth_solution is None:
            return

        self.examples += 1
        if predicted_solution is None:
            self.invalid += 1
            self.confusion[(ground_truth_solution, INVALID_LABEL)] += 1
            return
        self.confusion[(ground_truth_solution, predicted_solution)] += 1
        if predicted_solution == ground_truth_solution:
            self.correct += 1

    def result(self) -> MetricResult:
        accuracy = self.correct / self.examples if self.examples else 0.0
        validity_rate = (self.examples - self.invalid) / self.examples if self.examples else None
        confusion = {"solution": answer_confusion_matrix(self.confusion)}
        return MetricResult(
            name=self.name,
            accuracy=accuracy,
            correct=self.correct,
            total=self.examples,
            examples=self.examples,
            validity_rate=validity_rate,
            confusion=confusion,
            extra=answer_type_accuracy(confusion),
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


class ExactAnswerMetric(TaskMetric):
    """Single-answer domains: the text after `Answer:` up to the `;` must match exactly
    (case-insensitive). An output with no parseable answer counts as wrong and invalid.
    The confusion matrix has one field, "answer" ({true: {predicted: count}}, only the
    pairs that occur), so an eval log carries the majority baseline."""

    name = "task_accuracy"

    def __init__(self) -> None:
        self.correct = 0
        self.examples = 0
        self.invalid = 0
        self.confusion = Counter()   # (true answer, predicted answer or <invalid>) -> count

    def update(self, predicted_tokens: List[str], ground_truth_response: str) -> None:
        predicted = self._extract_answer(_tokens_to_text(predicted_tokens))
        expected = self._extract_answer(ground_truth_response)
        if expected is None:
            return
        self.examples += 1
        if predicted is None:
            self.invalid += 1
            self.confusion[(expected, INVALID_LABEL)] += 1
            return
        self.confusion[(expected, predicted)] += 1
        if predicted == expected:
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
            confusion={"answer": answer_confusion_matrix(self.confusion)},
        )

    def _extract_answer(self, text: str) -> Optional[str]:
        return extract_answer(text)


def extract_answer(text: str) -> Optional[str]:
    """The lower-cased answer of an `Answer: X;` response, or None. An "Answer:" anywhere
    wins (after working lines, a leading space or an earlier ";"); without one, the text
    before the first ";" on the first line (the teacher prompt ends in "Answer:", so a
    response may start at the value itself)."""
    match = re.search(r"Answer:\s*([^;\n]+);", text)
    if match is None:
        match = re.match(r"\s*([^;\n]+);", text)
    return match.group(1).strip().lower() if match else None

