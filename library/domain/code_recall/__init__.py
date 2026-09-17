"""Code recall: a small randomly structured Python program with numbered lines, one
question, one short answer (a line number, an identifier or a literal).

The programs are grown from a grammar with random identifiers, so answers cannot be read
off a template; they have to be found in the text — the code version of long-context
recall. Ground truth is generated with the program, so teacher and student can both be
scored against it. The answer set is closed (line numbers, literals, the identifier pool),
which is what the reduced output vocabulary is for.
"""

import re
from typing import List, Optional

from library.shared.metrics.task import ExactAnswerMetric
from ..base import Domain
from .corpus import all_possible_answers


class CodeRecallDomain(Domain):
    name = "code_recall"
    stop_token = ";"
    max_steps = 10
    max_input_tokens = 1024

    # Every possible answer in the exact output shape (identifier pieces tokenize into
    # sub-words, so the vocabulary carries the pieces the answers are made of).
    example_responses = [f"Answer: {answer};" for answer in all_possible_answers()]

    prompt_tokens = [
        "Read", "the", "numbered", "Python", "program", "and", "answer", "question",
        "with", "one", "line", "number", "identifier", "or", "value", "from", "it",
        "Program", "Question", "Answer", "Format", "exactly", "end", "semicolon", ":", ";", "?", "`",
    ]

    def teacher_prompt(self, text: str) -> str:
        return (
            "Read the numbered Python program and answer the question with one line number, "
            "identifier or value taken from it. Format exactly: Answer: <value>;\n\n"
            f"Program:\n{text}\n\n"
            "Answer:"
        )

    def task_metric(self) -> ExactAnswerMetric:
        return ExactAnswerMetric()

    def structural_validity(self, output: str) -> bool:
        return re.search(r"Answer:\s*[^;\n]+;", "Answer:" + output) is not None

    def build_corpus(self, count: Optional[int] = None) -> None:
        from .corpus import build_corpus
        build_corpus(self.corpus_path, count=count)
