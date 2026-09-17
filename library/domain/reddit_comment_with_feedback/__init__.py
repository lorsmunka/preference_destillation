"""Reddit comment classification with feedback: the four sentiment fields plus a verdict.

Same comment input and the same tone / sentiment / safety / toxicity JSON as
reddit_comment_sentiment, plus "approve" (would a moderator approve the comment) and
"reason" (one free sentence). Fixed fields with one free sentence: the semi-structured
middle between the closed JSON domain and free-form post generation. Task accuracy scores
the five closed fields; the reason is free text and is not scored.
"""

from typing import List, Optional

from library.shared.metrics.task import CATEGORIES, CATEGORY_VALUES, ClassificationMetric
from ..reddit_comment_sentiment import RedditSentimentDomain

FEEDBACK_CATEGORIES = CATEGORIES + ["approve"]
FEEDBACK_VALUES = {**CATEGORY_VALUES, "approve": ["true", "false"]}


class RedditCommentWithFeedbackDomain(RedditSentimentDomain):
    name = "reddit_comment_with_feedback"
    max_steps = 90

    example_responses = [
        '```json\n{\n    "tone": "aggressive",\n    "sentiment": "negative",\n    "safety": "harmful",\n    "toxicity": "toxic",\n    "approve": false,\n    "reason": "The comment attacks another person with a slur."\n}',
        '```json\n{\n    "tone": "rude",\n    "sentiment": "neutral",\n    "safety": "safe",\n    "toxicity": "respectful",\n    "approve": true,\n    "reason": "It is blunt but does not target or harm anyone."\n}',
        '```json\n{\n    "tone": "neutral",\n    "sentiment": "positive",\n    "safety": "harmful",\n    "toxicity": "toxic",\n    "approve": false,\n    "reason": "The comment encourages a dangerous action."\n}',
        '```json\n{\n    "tone": "polite",\n    "sentiment": "negative",\n    "safety": "safe",\n    "toxicity": "respectful",\n    "approve": true,\n    "reason": "A civil disagreement that stays on topic."\n}',
        '```json\n{\n    "tone": "friendly",\n    "sentiment": "positive",\n    "safety": "safe",\n    "toxicity": "respectful",\n    "approve": true,\n    "reason": "Supportive and relevant to the discussion."\n}',
    ]

    prompt_tokens = RedditSentimentDomain.prompt_tokens + [
        "approve", "reason", "true", "false", "would", "moderator", "this",
        "comment", "why", "one", "sentence", "Give",
    ]

    def teacher_prompt(self, text: str) -> str:
        return f"""Analyze this sentence and return your evaluation as JSON:


Sentence: "{text}"

Provide exactly one value for each field based on the sentence content:
    - tone: aggressive, rude, neutral, polite, friendly
    - sentiment: negative, neutral, positive
    - safety: harmful, safe
    - toxicity: toxic, respectful
    - approve: true, false (would a moderator approve this comment?)
    - reason: one sentence explaining the approve decision

JSON:
"""

    def task_metric(self) -> ClassificationMetric:
        return ClassificationMetric(categories=FEEDBACK_CATEGORIES, values=FEEDBACK_VALUES)

    @property
    def corpus_path(self) -> str:
        """Same reddit comments as the sentiment domain — it reuses that corpus."""
        from ..registry import get_domain
        return get_domain("reddit_comment_sentiment").corpus_path

    def build_corpus(self, count: Optional[int] = None) -> None:
        from ..registry import get_domain
        get_domain("reddit_comment_sentiment").build_corpus(count=count)
