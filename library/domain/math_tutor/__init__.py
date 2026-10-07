"""Math tutor: grade 2-4 word problems with executable solve() functions and stateful
follow-ups. Spec: thesis_work/domain_specs.txt.

The model's side of the tutor agent: one turn in (the problem, plus earlier turns with
their answers), one answer out. The harness that holds the conversation is separate.
Ground truth is the executed solve(), so teacher and student are both scored against it.

Two output formats, picked on Gemma 4's accuracy and format adherence (plan step 5):
  working  "24 - 9 = 15\\nAnswer: 15;"  calculation lines, then the answer
  answer   "Answer: 15;"
"""

import re
from typing import Optional

from library.shared.metrics.task import ExactAnswerMetric
from ..base import Domain

PROMPTS = {
    "working": (
        "Solve the maths word problem. If there are follow-up questions, answer only the last one; "
        "earlier answers are given and correct.\n"
        "Write each calculation on its own line as: number operator number = result, using only + - * /. "
        "Then write the last line as Answer: followed by the number and a semicolon, for example:\n"
        "6 * 4 = 24\n24 - 9 = 15\nAnswer: 15;\n"
        "The semicolon is required. No words, no units, nothing after the semicolon.\n\n{text}"
    ),
    "answer": (
        "Solve the maths word problem. If there are follow-up questions, answer only the last one; "
        "earlier answers are given and correct.\n"
        "Reply with exactly one line: Answer: followed by the number and a semicolon, for example:\n"
        "Answer: 15;\n"
        "The semicolon is required. No working, no words, no units.\n\n{text}"
    ),
}

# Every token an output can contain: digits one by one, the four operators, "=", the
# line break and the answer line (Gemma tokenizes numbers digit by digit).
EXAMPLE_RESPONSES = [
    "Answer: 1234567890;",
    "24 - 9 = 15\nAnswer: 15;",
    "12 + 7 = 19\nAnswer: 19;",
    "3 * 24 = 72\n72 - 35 = 37\nAnswer: 37;",
    "56 / 7 = 8\nAnswer: 8;",
    "45 + 8 = 53\n53 - 1 = 52\n52 / 8 = 6\nAnswer: 6;",
    "35-23=12\n4*6=24\n8/2=4\n3+4=7\nAnswer: 7;",   # Gemma sometimes drops the spaces
]


class MathTutorDomain(Domain):
    name = "math_tutor"
    stop_token = ";"
    max_steps = 64

    example_responses = EXAMPLE_RESPONSES
    prompt_tokens = ["Problem", "Answer", "Follow", "-", "up", ":", ";", "="]

    def __init__(self, output_format: str = "working"):
        if output_format not in PROMPTS:
            raise ValueError(f"output_format must be one of {sorted(PROMPTS)}")
        self.output_format = output_format

    def teacher_prompt(self, text: str) -> str:
        return PROMPTS[self.output_format].format(text=text)

    def task_metric(self) -> ExactAnswerMetric:
        return ExactAnswerMetric()

    def structural_validity(self, output: str) -> bool:
        return re.search(r"Answer:\s*\d+;", output) is not None

    def answer_matches(self, response: str, ground_truth) -> Optional[bool]:
        match = re.search(r"Answer:\s*(\d+)", response)
        return match is not None and int(match.group(1)) == int(ground_truth)

    def build_corpus(self, count: Optional[int] = None) -> None:
        from .corpus import build_corpus
        build_corpus(self.corpus_path, count=count)
