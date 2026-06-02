"""Domain ABC — everything that used to be dispatched on the domain string lives here:
the teacher prompt, the stop condition, the end-task metric, and the domain's slice of
the reduced output vocabulary (example responses, prompt tokens, extra auxiliary tokens).
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from library.shared.config import PROMPT_DELIMITER
from library.shared.metrics.base import TaskMetric


class Domain(ABC):
    name: str
    stop_token: str
    max_steps: int
    max_input_tokens: int

    # --- reduced-vocabulary contributions (see library.shared.vocabulary) ---------
    # Example responses are tokenized into the "example" section (the hardest labels).
    example_responses: List[str] = []
    # Domain-specific tokens from the evaluation prompt -> the "prompt" section.
    prompt_tokens: List[str] = []
    # Domain extras prepended to the shared auxiliary bank (e.g. math's A..T scaffold).
    extra_auxiliary_tokens: List[str] = []

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
