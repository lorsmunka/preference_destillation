"""Domain ABC — everything that used to be dispatched on the domain string lives here:
the teacher prompt, the stop condition, the end-task metric, the domain's slice of the
reduced output vocabulary (example responses, prompt tokens, extra auxiliary tokens), and
where its input corpus comes from (`corpus_path` + `build_corpus`).

A domain is one self-contained package under library/domain/<name>/: this class in
__init__.py, plus a corpus.py that produces the {"text": ...} input jsonl. To add a
domain, copy a package, edit it, and register it in registry.py.
"""

import sys
from abc import ABC, abstractmethod
from pathlib import Path
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

    # --- input corpus -------------------------------------------------------------
    @property
    def package_dir(self) -> Path:
        """The directory of the concrete domain's package (library/domain/<name>/)."""
        return Path(sys.modules[type(self).__module__].__file__).resolve().parent

    @property
    def corpus_path(self) -> str:
        """Path to this domain's input corpus — a jsonl of {"text": ...} records, read by
        data-gen and produced by `build_corpus`. Defaults to corpus.jsonl in the package;
        a domain may override to reuse another's corpus."""
        return str(self.package_dir / "corpus.jsonl")

    def build_corpus(self, count: Optional[int] = None) -> None:
        """Produce the input corpus at `corpus_path`. Override per domain, importing any
        heavy/optional deps lazily inside the override (so `import library` stays light)."""
        raise NotImplementedError(
            f"{type(self).__name__} has no corpus builder — add build_corpus to its corpus.py"
        )

    def __repr__(self) -> str:
        return f"Domain({self.name!r})"
