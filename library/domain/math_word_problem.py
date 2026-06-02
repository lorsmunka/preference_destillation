import re

from library.shared.metrics.task import MathMetric
from .base import Domain


class MathWordProblemDomain(Domain):
    name = "math_word_problem"
    stop_token = ";"
    max_steps = 350
    max_input_tokens = 500

    def teacher_prompt(self, text: str) -> str:
        from library.shared.utilities import Utilities
        return Utilities.create_math_word_problem_prompt(text)

    def task_metric(self) -> MathMetric:
        return MathMetric()

    def structural_validity(self, output: str) -> bool:
        return bool(re.search(r"Solution:\s*[^;\n]+", output))
