"""Shared helpers for the experiment scripts in this directory.

Kept out of `library/` on purpose: this is researcher-side scaffolding for running
cohorts, not part of the distillation framework.
"""

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def seeded_run_training(seeded_runs):
    """Train `(seed, TrainingRun)` pairs, seeding torch before each one.

    `TrainingRun` has no seed field, so replicate runs would otherwise differ only by
    whatever RNG state the process happened to be in. Seeding here makes each replicate
    reproducible without touching the framework. Completed runs are skipped, so the
    cohort is resumable.
    """
    import torch
    from library.shared import ExitListener
    from library.training.training_runner import TrainingRunner, is_completed, write_training_queue

    exit_listener = ExitListener()
    write_training_queue([run for _, run in seeded_runs])
    for seed, run in seeded_runs:
        if is_completed(run.run_name):
            print(f"Skipping completed run: {run.run_name}")
            continue
        torch.manual_seed(seed)
        if not TrainingRunner(run, exit_listener).run():
            print("Training interrupted — stopping cohort.")
            break
    exit_listener.stop()


def write_rows(filename: str, columns, rows):
    """Write result rows to experiments/results/<filename>, replacing rows that share a
    key column value rather than discarding the file's other content."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    key = columns[0]

    existing = {}
    if path.exists():
        with open(path, "r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                existing[row[key]] = row

    for row in rows:
        existing[str(row[key])] = row

    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in existing.values():
            writer.writerow(row)
    print(f"-> {path}  ({len(existing)} rows)")
    return path


def best_and_final(run):
    """Best-epoch and final-epoch eval records. These runs overfit at small sample
    sizes, so the last epoch alone understates what the configuration reaches."""
    evals = run.eval_epochs
    if not evals:
        return None, None
    best = max(evals, key=lambda e: e.task_accuracy)
    return best, evals[-1]
