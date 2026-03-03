import json
from collections import Counter
from typing import Dict, List, Optional, Tuple


CATEGORIES = ['tone', 'sentiment', 'safety', 'toxicity']

CATEGORY_VALUES = {
    'tone': ['aggressive', 'rude', 'neutral', 'polite', 'friendly'],
    'sentiment': ['negative', 'neutral', 'positive'],
    'safety': ['harmful', 'safe'],
    'toxicity': ['toxic', 'respectful'],
}


class ClassificationAccuracyCalculator:

    def __init__(self):
        self.total_categories = 0
        self.correct_categories = 0
        self.confusion_matrices = {
            category: Counter() for category in CATEGORIES
        }

    def update(self, predicted_tokens: List[str], ground_truth_response: str) -> Tuple[int, int]:
        predicted = self._parse_tokens(predicted_tokens)
        ground_truth = self._parse_response(ground_truth_response)

        if predicted is None or ground_truth is None:
            return 0, 0

        correct = 0
        total = 0
        for category in CATEGORIES:
            if category in ground_truth:
                total += 1
                predicted_value = predicted.get(category, '')
                ground_truth_value = ground_truth[category]
                self.confusion_matrices[category][(ground_truth_value, predicted_value)] += 1
                if predicted_value == ground_truth_value:
                    correct += 1

        self.correct_categories += correct
        self.total_categories += total
        return correct, total

    def get_accuracy(self) -> float:
        if self.total_categories == 0:
            return 0.0
        return self.correct_categories / self.total_categories

    def get_confusion_matrices(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        result = {}
        for category in CATEGORIES:
            values = CATEGORY_VALUES[category]
            matrix = {}
            for true_value in values:
                row = {}
                for predicted_value in values:
                    row[predicted_value] = self.confusion_matrices[category].get(
                        (true_value, predicted_value), 0)
                matrix[true_value] = row
            result[category] = matrix
        return result

    def _parse_tokens(self, tokens: List[str]) -> Optional[Dict[str, str]]:
        text = ''.join(tokens).replace('\u2581', ' ')
        return self._extract_json_values(text)

    def _parse_response(self, response: str) -> Optional[Dict[str, str]]:
        return self._extract_json_values(response)

    def _extract_json_values(self, text: str) -> Optional[Dict[str, str]]:
        text = text.replace('```json', '').replace('```', '').strip()
        try:
            data = json.loads(text)
            result = {}
            for category in CATEGORIES:
                if category in data:
                    result[category] = str(data[category]).lower()
            return result if result else None
        except json.JSONDecodeError:
            return None
