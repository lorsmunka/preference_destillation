"""tooling — the researcher-facing analytics / evaluation / inference layer.

`Analysis` is the single facade; `Run`/`RunStore`/`Experiment` are the run model + cohort
aggregation; `render` holds plots/tables. Post-run analysis here only reads logs; the model
is loaded lazily (via library.model.StudentModel) only on evaluate/infer.
"""

from library.tooling.analysis import Analysis, Comparison, EvalResult
from library.tooling.run import Experiment, Run, RunStore, sort_runs

__all__ = [
    "Analysis",
    "Comparison",
    "EvalResult",
    "Run",
    "RunStore",
    "Experiment",
    "sort_runs",
]
