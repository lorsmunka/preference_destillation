"""Typed run/job specifications — author these instead of raw config dicts.

    from library import TrainingRun, GenerationJob
    from library.domain import MATH_WORD_PROBLEM, RedditSentimentDomain

    JOBS = [GenerationJob(domain=RedditSentimentDomain(), max_examples=8000)]
    RUNS = [TrainingRun(domain=MATH_WORD_PROBLEM, run_name="r1", hidden_dim=48)]

`domain=` takes a Domain object — a singleton (`MATH_WORD_PROBLEM`), a fresh instance
(`RedditSentimentDomain()`), or your own `Domain` subclass — or a name string. A Domain
object is auto-registered, so a brand-new custom domain works end-to-end. Defaults cover
every other knob; only `run_name` (training) and `domain` are required.

A custom student network instead of the built-in Transformer: `model=` takes its class (kept in
an importable file such as library/model/custom/, so a finished run can be rebuilt later) and
`model_settings=` the keyword arguments for its __init__; hidden_dim / num_layers / num_heads /
dropout are then ignored. See library/model/custom_student.py for what the class must do.

Torch-free: imports only `library.domain` + `library.shared` constants.
"""

import importlib
import json
from dataclasses import dataclass
from typing import Optional, Union

from library.domain import Domain, as_domain
from library.shared import DEFAULT_TEACHER_MODEL

# Size of the held-out test set: the last 1024 examples' batch files of the directory, the
# same set whatever max_training_examples is (see library/training/batch_handler.py).
DEFAULT_MAX_TEST_EXAMPLES = 1024


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
    model: Optional[type] = None                         # None = the built-in Transformer; a class = a custom student network
    model_settings: Optional[dict] = None                # keyword arguments for the custom model's __init__
    epoch_count: int = 3
    batch_size: int = 32
    learning_rate: float = 0.0015
    lr_warmup_ratio: float = 0.1
    max_training_examples: Optional[int] = 8000          # caps training only; the test set does not move with it
    training_test_ratio: float = 0.8                     # the last (1 - ratio) of the batch files are the test region
    max_test_examples: Optional[int] = DEFAULT_MAX_TEST_EXAMPLES   # test set = the region's last N examples' files; None = the whole region
    shuffle_seed: Optional[int] = None                   # None = batches in file order (every past run); an int shuffles file and example order per epoch
    training_groups: Optional[tuple] = None              # e.g. ("original", "new"): train only on records with these "group" flags; None = all
    auxiliary_token_percentage: float = 1.0
    kl_ratio_start: float = 0.99
    kl_ratio_end: float = 0.5
    distillation_temperature_start: float = 1.0
    distillation_temperature_end: float = 1.0
    teacher_temperature_start: Optional[float] = None   # None = follow distillation_temperature (coupled, legacy)
    teacher_temperature_end: Optional[float] = None
    divergence: str = "forward_kl"                       # forward_kl | reverse_kl | js | tvd
    entropy_kl_weighting: float = 0.0                    # 0 = scheduled ratio only; 1 = per-token ratio = teacher entropy / log V
    kl_ratio_cycle_epochs: float = 0.0                   # 0 = off; 1.0 = one oscillation per epoch
    kl_ratio_cycle_amplitude: float = 0.0                # 0.10 = +/-10% around the scheduled ratio
    attention_noise_start: float = 0.0                   # gaussian sigma on attention logits, training only
    attention_noise_end: float = 0.0                     # start > end anneals noise down, start < end anneals it up
    eval_top_k: int = 20
    eval_cap_multiple: float = 2.0
    description: str = ""

    def __post_init__(self):
        self.domain = as_domain(self.domain)
        if self.model is None and self.model_settings is not None:
            raise ValueError(f"{self.run_name}: model_settings is for a custom model, so pass model= too")
        if self.model is not None:
            check_custom_model(self.model, self.model_settings)

    def as_dict(self) -> dict:
        data = dict(self.__dict__)
        data["domain"] = self.domain.name
        if self.model is None:   # the built-in Transformer: info.json stays as it always was
            del data["model"]
            del data["model_settings"]
        else:
            data["model"] = model_path(self.model)
            data["model_settings"] = dict(self.model_settings or {})
            for key in CUSTOM_MODEL_IGNORES:
                data[key] = None
        return data


# The built-in Transformer's settings, which a custom model replaces with its own model_settings.
CUSTOM_MODEL_IGNORES = ("hidden_dim", "num_layers", "num_heads", "dropout")


def model_path(model_class: type) -> str:
    """Where a model class can be imported from, as info.json records it."""
    return f"{model_class.__module__}.{model_class.__qualname__}"


def load_model_class(path: str) -> type:
    module_name, class_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def check_custom_model(model_class: type, model_settings: Optional[dict]) -> None:
    """Refuse, before any training, a custom model that a finished run could not be rebuilt from."""
    if not isinstance(model_class, type):
        raise ValueError(f"model= takes the model's class (e.g. model=MyStudent), not {model_class!r}")
    if model_class.__module__ == "__main__":
        raise ValueError(f"{model_class.__name__} is defined in the script being run, so a finished run could not "
                         f"be rebuilt from it later: move it to an importable file such as library/model/custom/")
    try:
        found = load_model_class(model_path(model_class))
    except (ImportError, AttributeError):
        found = None
    if found is not model_class:
        raise ValueError(f"{model_path(model_class)} cannot be imported by that name (a class defined inside a "
                         f"function or another class?): define it at the top level of an importable file")
    try:
        json.dumps(model_settings or {})
    except TypeError as error:
        raise ValueError(f"model_settings must be plain values (numbers, strings, lists), as info.json stores "
                         f"them: {error}")


@dataclass
class GenerationJob:
    """One teacher-data-generation job. `teacher_model` maps to the consumer's
    `model_name` key; `domain` serializes to its name string."""
    domain: Union[str, Domain]
    teacher_model: str = DEFAULT_TEACHER_MODEL
    max_examples: int = 8000
    batch_size: int = 32
    load_in_4bit: bool = False       # NF4 via bitsandbytes — how a 12B teacher fits a 12 GB card
    chat_template: bool = False      # wrap the domain prompt in the tokenizer's chat template (Gemma 4 expects it)
    thinking: bool = False           # Gemma 4 thinking mode; the trace is stored as text, only the answer gets logits
    generation_batch_size: int = 1   # inputs generated together (left-padded model.generate); 1 = the one-at-a-time loop
    description: str = ""

    def __post_init__(self):
        self.domain = as_domain(self.domain)
        if self.thinking:
            self.chat_template = True

    def as_dict(self) -> dict:
        return {
            "domain": self.domain.name,
            "model_name": self.teacher_model,
            "max_examples": self.max_examples,
            "batch_size": self.batch_size,
            "load_in_4bit": self.load_in_4bit,
            "chat_template": self.chat_template,
            "thinking": self.thinking,
            "generation_batch_size": self.generation_batch_size,
            "description": self.description,
        }
