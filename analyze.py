"""Researcher entry point — edit freely and run `python analyze.py`.

This replaces the input()-menu tools (analysis/, experimental_analysis/, top-k-accruacy-analsy/)
with a single script that imports the library. Everything below is just example usage of the library.
With no edits it summarises the first completed run, lists the seed cohorts and compares
a few runs — task accuracy always beside its majority baseline and the excess over it.
"""

import sys

from library import Analysis, RunStore
from library.tooling.render import experiment_summary

sys.stdout.reconfigure(encoding="utf-8")

store = RunStore()
completed = store.completed()
print(f"{len(store.all())} runs ({len(completed)} completed)\n")
if not completed:
    sys.exit("No completed runs yet — train one first (python train.py).")

# ── single run: log interpretation (no model) ──────────────────────────
analysis = Analysis(completed[0])        # or Analysis("<run name>")
print(analysis.summary())
# analysis.plot_training()                      # -> runs/<name>/logs/training_progress.png

# ── seed cohorts: completed runs grouped by experiment tag + configuration, mean ± sample std ─
print("\nCohorts:")
print(experiment_summary(list(Analysis.cohorts().values())))

# ── multi-run comparison ───────────────────────────────────────────────
print("\nComparison (first 8 completed, by params):")
print(Analysis.compare(completed[:8]).sort("params").table())

# ── evaluate a checkpoint with the new eval-time metrics (model path) ───
# Needs batch data + the Gemma tokenizer; uncomment when running where those exist:
# result = Analysis(completed[0]).evaluate(split="test", k=20)
# print(result)
