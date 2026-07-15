"""Typed run/job specifications — author these instead of raw config dicts.

    from library import TrainingRun, GenerationJob
    from library.domain import MATH_WORD_PROBLEM, RedditSentimentDomain

    JOBS = [GenerationJob(domain=RedditSentimentDomain(), max_examples=8000)]
    RUNS = [TrainingRun(domain=MATH_WORD_PROBLEM, run_name="r1", hidden_dim=48)]

`domain=` takes a Domain object — a singleton (`MATH_WORD_PROBLEM`), a fresh instance
(`RedditSentimentDomain()`), or your own `Domain` subclass — or a name string. A Domain
object is auto-registered, so a brand-new custom domain works end-to-end. Defaults cover
every other knob; only `run_name` (training) and `domain` are required.

Torch-free: imports only `library.domain` + `library.shared` constants.
"""

from dataclasses import dataclass
from typing import Optional, Union

from library.domain import Domain, as_domain
from library.shared import DEFAULT_TEACHER_MODEL


@dataclass
class TrainingRun:
    """One training run. Serializes (via `as_dict`) to the dict TrainingRunner consumes
    and writes to info.json — with `domain` as its name string for paths/tooling."""
    domain: Union[str, Domain]
    run_name: str
    teacher_model: str = DEFAULT_TEACHER_MODEL
    experiment: str = "default"
    hidden_dim: int = 48
    num_layers: int = 3
    num_heads: int = 1
    dropout: float = 0.08
    epoch_count: int = 3
    batch_size: int = 32
    learning_rate: float = 0.0015
    lr_warmup_ratio: float = 0.1
    max_training_examples: Optional[int] = 8000
    training_test_ratio: float = 0.8
    auxiliary_token_percentage: float = 1.0
    kl_ratio_start: float = 0.99
    kl_ratio_end: float = 0.5
    distillation_temperature_start: float = 1.0
    distillation_temperature_end: float = 1.0
    eval_top_k: int = 20
    eval_cap_multiple: float = 2.0
    description: str = ""

    def __post_init__(self):
        self.domain = as_domain(self.domain)

    def as_dict(self) -> dict:
        data = dict(self.__dict__)
        data["domain"] = self.domain.name
        return data


@dataclass
class GenerationJob:
    """One teacher-data-generation job. `teacher_model` maps to the consumer's
    `model_name` key; `domain` serializes to its name string."""
    domain: Union[str, Domain]
    teacher_model: str = DEFAULT_TEACHER_MODEL
    max_examples: int = 8000
    batch_size: int = 32
    description: str = ""

    def __post_init__(self):
        self.domain = as_domain(self.domain)

    def as_dict(self) -> dict:
        return {
            "domain": self.domain.name,
            "model_name": self.teacher_model,
            "max_examples": self.max_examples,
            "batch_size": self.batch_size,
            "description": self.description,
        }
