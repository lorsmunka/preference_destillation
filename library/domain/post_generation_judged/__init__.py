"""Post generation v2: write the post, then judge it — two JSON fields appended.

Same input as post_generation (a reddit comment), same free-form post, but the teacher
must close with a verdict on its own post: {"approve": true|false, "reason": "<one
sentence>"}. The post keeps its entropy; the verdict adds a structured, checkable tail
with some freedom in the reason sentence. Task accuracy = the student's `approve`
matches the teacher's; structural validity = the JSON parses.
"""

import json
import re
from typing import List, Optional

from library.shared.metrics.task import JudgedPostMetric
from ..post_generation import PostGenerationDomain, _load_sample_posts

JUDGMENT_EXAMPLES = [
    '{"approve": true, "reason": "The post is clear and fits the comment."}',
    '{"approve": false, "reason": "The post drifts away from what the comment replies to."}',
    '{"approve": true, "reason": "It reads like a real post and sets up the comment well."}',
    '{"approve": false, "reason": "The post is too vague to have prompted this comment."}',
]


class PostGenerationJudgedDomain(PostGenerationDomain):
    name = "post_generation_judged"
    stop_token = "}"
    max_steps = 220
    max_input_tokens = 25

    prompt_tokens = PostGenerationDomain.prompt_tokens + [
        "Then", "judge", "your", "own", "JSON", "approve", "reason", "true", "false",
        "one", "sentence", "line", "{", "}", ":", ",", '"',
    ]

    @property
    def example_responses(self) -> List[str]:
        # Posts for the free-form half, judgment blocks for the structured tail.
        return _load_sample_posts() + JUDGMENT_EXAMPLES

    def teacher_prompt(self, text: str) -> str:
        return (
            f'Generate a plausible reddit post based on this comment: "{text}"\n\n'
            "No title or text formatting needed. Only reply with the body of the post. "
            "3 sentences or about 50 words. End the post with <end>.\n"
            "Then, on a new line, judge your own post as JSON with exactly two fields: "
            '"approve" (true or false: would a moderator approve this post?) and '
            '"reason" (one sentence).\n\n'
            "Post: "
        )

    def task_metric(self) -> JudgedPostMetric:
        return JudgedPostMetric()

    def structural_validity(self, output: str) -> bool:
        return parse_judgment(output) is not None


def parse_judgment(output: str) -> Optional[dict]:
    """The JSON object after the post's <end>, or None if it is missing or malformed."""
    tail = output.split("<end>", 1)[1] if "<end>" in output else output
    match = re.search(r"\{.*\}", tail, re.DOTALL)
    if not match:
        return None
    try:
        judgment = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(judgment, dict) or not isinstance(judgment.get("approve"), bool):
        return None
    return judgment
