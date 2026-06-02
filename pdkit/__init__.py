"""pdkit — unified analytics / evaluation / inference kit for the preference-distillation project.

One importable library behind a single facade (`Analysis`). Post-run analysis only reads logs +
info.json (never reloads the model); the model is reloaded only on the interactive demo/eval
path (`StudentModel`), which is for TDK demos, debugging and testing — not deployment.

    from pdkit import Analysis, RunStore
    Analysis("run-name").summary()
    Analysis.compare(RunStore().completed()).table()

Importing pdkit does NOT import torch — that happens lazily only when a model method runs.
"""

from .analysis import Analysis, Comparison, EvalResult
from .domains import get_domain, registered_domains
from .run import Experiment, Run, RunStore, sort_runs

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
