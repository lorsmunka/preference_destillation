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
from library.specs import TrainingRun, GenerationJob

__version__ = "0.1.0"

# Domains are passed as objects (singletons / your own subclass) from `library.domain`;
# the registry (get_domain / register_domain) is an internal detail for resolving built-in
# names off info.json, not part of this front-door facade.
__all__ = [
    "Analysis",
    "Comparison",
    "EvalResult",
    "RunStore",
    "Run",
    "Experiment",
    "sort_runs",
    "TrainingRun",
    "GenerationJob",
]
