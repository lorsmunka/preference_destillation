import os
from functools import lru_cache
from typing import List, Optional

from library.shared.metrics.task import PostGenerationMetric
from ..base import Domain


# Teacher-generated example posts, co-located in this domain's package.
_SAMPLE_POSTS_PATH = os.path.join(os.path.dirname(__file__), "sample_posts.txt")


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

    @property
    def corpus_path(self) -> str:
        """Post-generation is fed reddit comments — it reuses that domain's input corpus."""
        from ..registry import get_domain
        return get_domain("reddit_comment_sentiment").corpus_path

    def build_corpus(self, count: Optional[int] = None) -> None:
        """Input corpus = reddit comments, so building it means building the reddit corpus."""
        from ..registry import get_domain
        get_domain("reddit_comment_sentiment").build_corpus(count=count)

    def build_example_responses(self, count: Optional[int] = None) -> None:
        """(Re)generate sample_posts.txt — the teacher-written example posts that seed this
        domain's `example_responses` vocab section. Heavy (runs the teacher); rarely needed."""
        from .corpus import build_example_responses
        build_example_responses(_SAMPLE_POSTS_PATH, count=count)

    # free-form: structural_validity stays None (inherited)
