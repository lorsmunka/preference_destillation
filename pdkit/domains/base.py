"""Domain ABC — everything that used to be dispatched on the domain string lives here."""

from abc import ABC, abstractmethod
from typing import Optional

from ..config import PROMPT_DELIMITER
from ..metrics.base import TaskMetric


class Domain(ABC):
    name: str
    stop_token: str
    max_steps: int
    max_input_tokens: int

    def student_prompt(self, text: str) -> str:
        """Context fed to the student at inference: just the sentence + delimiter (same for all)."""
        return text + PROMPT_DELIMITER

    @abstractmethod
    def teacher_prompt(self, text: str) -> str:
        """Full prompt fed to the teacher (domain-specific)."""

    @abstractmethod
    def task_metric(self) -> TaskMetric:
        """Fresh accumulator for this domain's end-task accuracy."""

    def structural_validity(self, output: str) -> Optional[bool]:
        """Whether a generated output is structurally valid. None for free-form domains
        that have no structural notion (overridden only by structured domains)."""
        return None

    def is_stop(self, last_token_decoded: str, generated_text: str) -> bool:
        """Stop condition during generation. Unifies the old per-domain dispatch
        (ModelHandler.is_stop / inference.has_stop_token)."""
        return last_token_decoded == self.stop_token or self.stop_token in generated_text

    def __repr__(self) -> str:
        return f"Domain({self.name!r})"
