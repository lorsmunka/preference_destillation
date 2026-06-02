import json
from typing import Optional

from library.shared.metrics.task import ClassificationMetric
from ..base import Domain


class RedditSentimentDomain(Domain):
    name = "reddit_comment_sentiment"
    stop_token = "}"
    max_steps = 50
    max_input_tokens = 25

    example_responses = [
        '```json\n{\n    "tone": "aggressive",\n    "sentiment": "negative",\n    "safety": "harmful",\n    "toxicity": "toxic"\n}',
        '```json\n{\n    "tone": "rude",\n    "sentiment": "neutral",\n    "safety": "safe",\n    "toxicity": "respectful"\n}',
        '```json\n{\n    "tone": "neutral",\n    "sentiment": "positive",\n    "safety": "harmful",\n    "toxicity": "toxic"\n}',
        '```json\n{\n    "tone": "polite",\n    "sentiment": "negative",\n    "safety": "safe",\n    "toxicity": "respectful"\n}',
        '```json\n{\n    "tone": "friendly",\n    "sentiment": "positive",\n    "safety": "harmful",\n    "toxicity": "toxic"\n}',
    ]

    prompt_tokens = [
        "Analyze", "sentence", "return", "evaluation", "JSON",
        "Sentence", "Provide", "exactly", "one", "value", "for", "each",
        "field", "based", "on", "the", "content",
        "tone", "sentiment", "safety", "toxicity",
        "-", "(", ")", ".", ":",
    ]

    def teacher_prompt(self, text: str) -> str:
        return f"""Analyze this sentence and return your evaluation as JSON:


Sentence: "{text}"

Provide exactly one value for each field based on the sentence content:
    - tone: aggressive, rude, neutral, polite, friendly
    - sentiment: negative, neutral, positive
    - safety: harmful, safe
    - toxicity: toxic, respectful

JSON:
"""

    def task_metric(self) -> ClassificationMetric:
        return ClassificationMetric()

    def build_corpus(self, count: Optional[int] = None) -> None:
        from .corpus import build_corpus
        build_corpus(self.corpus_path, count=count)

    def structural_validity(self, output: str) -> bool:
        text = output.replace("```json", "").replace("```", "").strip()
        try:
            json.loads(text)
            return True
        except json.JSONDecodeError:
            return False
