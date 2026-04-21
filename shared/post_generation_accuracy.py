from typing import List, Tuple


class PostGenerationAccuracyCalculator:

    def __init__(self):
        self.total_examples = 0
        self.complete_responses = 0

    def update(self, predicted_tokens: List[str], ground_truth_response: str) -> Tuple[int, int]:
        predicted_text = ''.join(predicted_tokens).replace('\u2581', ' ')

        self.total_examples += 1
        if "<end>" in predicted_text:
            self.complete_responses += 1
            return 1, 1

        return 0, 1

    def get_accuracy(self) -> float:
        if self.total_examples == 0:
            return 0.0
        return self.complete_responses / self.total_examples

    def get_confusion_matrices(self) -> dict:
        return {
            "completion_rate": {
                "complete": self.complete_responses,
                "total": self.total_examples,
            }
        }
