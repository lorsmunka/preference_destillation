"""Generation metrics — natural termination and length, from a free-generation rollout.

Today's eval forces the student to exactly the teacher's length, so self-termination is never
tested. With a free-gen rollout capped at `cap_multiple x teacher_length` these come for free:
if no stop token by the cap, the example is a termination failure; the same rollout yields the
student/teacher length ratio.
"""

from dataclasses import dataclass
from typing import List, Optional


def stop_index(tokens: List[str], stop_token: str) -> Optional[int]:
    """Index of the first token at/by which `stop_token` has appeared in the decoded text,
    or None if it never appears. Mirrors the substring check used during generation."""
    text = ""
    for index, token in enumerate(tokens):
        text += token
        if stop_token in token or stop_token in text:
            return index
    return None


def length_ratio(student_length: int, teacher_length: int) -> Optional[float]:
    if not teacher_length:
        return None
    return student_length / teacher_length


@dataclass
class TerminationStats:
    examples: int = 0
    terminated: int = 0
    length_ratio_sum: float = 0.0

    def update(self, student_tokens: List[str], stop_token: str, teacher_length: int) -> None:
        self.examples += 1
        index = stop_index(student_tokens, stop_token)
        if index is not None:
            self.terminated += 1
        ratio = length_ratio(len(student_tokens), teacher_length)
        if ratio is not None:
            self.length_ratio_sum += ratio

    @property
    def termination_rate(self) -> float:
        return self.terminated / self.examples if self.examples else 0.0

    @property
    def mean_length_ratio(self) -> float:
        return self.length_ratio_sum / self.examples if self.examples else 0.0
