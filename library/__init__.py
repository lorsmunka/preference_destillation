"""library — the preference-distillation library.

Subpackages: shared (config, vocabulary, metrics, logging), domain, model, data_gen,
training, tooling. Researcher entry points (train.py / generate_data.py / analyze.py) live
at the repo root and import from here.

    from library import Analysis, RunStore
    Analysis("run-name").summary()

Importing `library` does NOT import torch — that happens only when a model method runs or
`library.model` is imported explicitly.
"""

from library.tooling import Analysis, Comparison, EvalResult, Experiment, Run, RunStore, sort_runs
from library.domain import get_domain, registered_domains

__version__ = "0.1.0"

__all__ = [
    "Analysis",
    "Comparison",
    "EvalResult",
    "RunStore",
    "Run",
    "Experiment",
    "sort_runs",
    "get_domain",
    "registered_domains",
]
