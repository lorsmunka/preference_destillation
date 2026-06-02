import os
from functools import lru_cache
from typing import List

from library.shared.metrics.task import PostGenerationMetric
from .base import Domain


# library/domain/post_generation.py -> repo root is two levels up
_SAMPLE_POSTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "text_generation", "post_generation", "sample_posts.txt",
)


@lru_cache(maxsize=1)
def _load_sample_posts() -> List[str]:
    """Read the example posts once, on first vocab build (the file is large — keep it
    out of `import library`, which only needs the domain registry, not the examples)."""
    with open(_SAMPLE_POSTS_PATH, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


class PostGenerationDomain(Domain):
    name = "post_generation"
    stop_token = "<end>"
    max_steps = 150
    max_input_tokens = 25

    prompt_tokens = [
        "Given", "Reddit", "comment", "write", "original", "post",
        "that", "could", "have", "prompted", "this", "comment",
        "The", "post", "should", "be", "no", "longer", "than",
        "sentences", "End", "with", "Comment", "Post",
        "Example", "example",
        "<", ">", "end",
    ]

    @property
    def example_responses(self) -> List[str]:
        return _load_sample_posts()

    def teacher_prompt(self, text: str) -> str:
        return (
            f'Generate a plausible reddit post based on this comment: "{text}"\n\n'
            "No title or text formatting needed. Only reply with the body of the post. "
            "3 sentences or about 50 words. End post with <end>.\n\nPost: "
        )

    def task_metric(self) -> PostGenerationMetric:
        return PostGenerationMetric(end_marker=self.stop_token)

    # free-form: structural_validity stays None (inherited)
