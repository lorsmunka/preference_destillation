"""Researcher entry point — edit freely and run `python analyze.py`.

This replaces the input()-menu tools (analysis/, experimental_analysis/, top-k-accruacy-analsy/)
with a single script that imports the library. Everything below is just example usage of pdkit.
"""

from pdkit import Analysis, RunStore
from pdkit.render import comparison_table, experiment_summary

store = RunStore()
print(f"{len(store.all())} runs ({len(store.completed())} completed)\n")

# ── single run: log interpretation (no model) ──────────────────────────
analysis = Analysis("exp-purece-t1-1")
print(analysis.summary())
# analysis.plot_training()                      # -> runs/<name>/logs/training_progress.png

# ── experiments (seed cohorts): mean ± std, reproduces the thesis tables ─
print("\nCohorts:")
print(experiment_summary(list(Analysis.cohorts().values())))

# ── multi-run comparison ───────────────────────────────────────────────
print("\nComparison (first 8 completed, by params):")
print(Analysis.compare([r.name for r in store.completed()[:8]]).sort("params").table())

# ── evaluate a checkpoint with the new eval-time metrics (model path) ───
# Needs batch data + the Gemma tokenizer; uncomment when running where those exist:
# result = Analysis("exp-kl99to50-t1-1").evaluate(split="test", k=20)
# print(result)
