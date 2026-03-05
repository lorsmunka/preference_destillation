import re
from typing import List, Optional, Tuple


class MathAccuracyCalculator:

    def __init__(self):
        self.total_examples = 0
        self.correct_solutions = 0

    def update(self, predicted_tokens: List[str], ground_truth_response: str) -> Tuple[int, int]:
        predicted_solution = self._extract_solution(
            ''.join(predicted_tokens).replace('\u2581', ' '))
        ground_truth_solution = self._extract_solution(ground_truth_response)

        if ground_truth_solution is None:
            return 0, 0

        self.total_examples += 1
        if predicted_solution == ground_truth_solution:
            self.correct_solutions += 1
            return 1, 1

        return 0, 1

    def get_accuracy(self) -> float:
        if self.total_examples == 0:
            return 0.0
        return self.correct_solutions / self.total_examples

    def get_confusion_matrices(self) -> dict:
        return {
            "solution_accuracy": {
                "correct": self.correct_solutions,
                "total": self.total_examples,
            }
        }

    def _extract_solution(self, text: str) -> Optional[str]:
        match = re.search(r'Solution:\s*([^;\n]+)', text)
        if match:
            return match.group(1).strip()
        return None
