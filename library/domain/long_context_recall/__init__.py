"""Long-context recall: a page of facts about Virág, one question, one-word answer.

Unlike the other domains the true answer is generated with the text, so both the teacher
and the student can be scored against ground truth (the corpus stores it as "answer").
The answer set is closed (see corpus.ALL_ANSWERS), which is exactly what the reduced
output vocabulary is for.
"""

import re
from typing import List, Optional

from library.shared.metrics.task import ExactAnswerMetric
from ..base import Domain
from .corpus import ALL_ANSWERS, SUBJECT


class LongContextRecallDomain(Domain):
    name = "long_context_recall"
    stop_token = ";"
    max_steps = 12

    # Every possible answer, in the exact output shape, so the vocabulary's hardest section
    # contains each of them.
    example_responses = [f"Answer: {answer};" for answer in ALL_ANSWERS]

    prompt_tokens = [
        "Read", "the", "text", "and", "answer", "question", "with", "one", "word",
        "from", "text", "Text", "Question", "Answer", "Format", "exactly", "end",
        "semicolon", ":", ";", ".", "?",
    ]

    def teacher_prompt(self, text: str) -> str:
        return (
            "Read the text and answer the question with one word taken from the text. "
            "Format exactly: Answer: <word>;\n\n"
            f"Text: {text}\n\n"
            "Answer:"
        )

    def task_metric(self) -> ExactAnswerMetric:
        return ExactAnswerMetric()

    def structural_validity(self, output: str) -> bool:
        return re.search(r"Answer:\s*[^;\n]+;", "Answer:" + output) is not None

    def build_corpus(self, count: Optional[int] = None) -> None:
        from .corpus import build_corpus
        build_corpus(self.corpus_path, count=count)
