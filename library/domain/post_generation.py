from library.shared.metrics.task import PostGenerationMetric
from .base import Domain


class PostGenerationDomain(Domain):
    name = "post_generation"
    stop_token = "<end>"
    max_steps = 150
    max_input_tokens = 25

    def teacher_prompt(self, text: str) -> str:
        from library.shared.utilities import Utilities
        return Utilities.create_post_generation_prompt(text)

    def task_metric(self) -> PostGenerationMetric:
        return PostGenerationMetric(end_marker=self.stop_token)

    # free-form: structural_validity stays None (inherited)
