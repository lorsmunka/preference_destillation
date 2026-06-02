import json

from ..metrics.task import ClassificationMetric
from .base import Domain


class RedditSentimentDomain(Domain):
    name = "reddit_comment_sentiment"
    stop_token = "}"
    max_steps = 50
    max_input_tokens = 25

    def teacher_prompt(self, text: str) -> str:
        from shared.utilities import Utilities  # single source for the prompt text (bridge)
        return Utilities.create_reddit_sentiment_prompt(text)

    def task_metric(self) -> ClassificationMetric:
        return ClassificationMetric()

    def structural_validity(self, output: str) -> bool:
        text = output.replace("```json", "").replace("```", "").strip()
        try:
            json.loads(text)
            return True
        except json.JSONDecodeError:
            return False
