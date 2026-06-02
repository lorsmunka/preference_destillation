"""Rendering — thin consumers of computed results (no analysis logic here).

`tables` returns/prints text; `plots` writes PNGs. Keeping these separate from compute is
what makes the numbers reusable and testable without a TTY or matplotlib.
"""

from .tables import comparison_table, experiment_summary, run_summary

__all__ = ["run_summary", "comparison_table", "experiment_summary"]
