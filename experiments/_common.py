"""Shared helpers for the experiment scripts in this directory.

Kept out of `library/` on purpose: this is researcher-side scaffolding for running
cohorts, not part of the distillation framework.

Spread convention of every report here: the sample std across seeds (n - 1), shown as
mean±std; with a single run there is no std.
"""

import csv
import os
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def seeded_run_training(seeded_runs) -> bool:
    """Train `(seed, TrainingRun)` pairs, seeding torch before each one.

    `TrainingRun` has no seed field, so replicate runs would otherwise differ only by
    whatever RNG state the process happened to be in. Seeding here makes each replicate
    reproducible without touching the framework. Completed runs are skipped, so the
    cohort is resumable. Returns True when every run is completed and False when a stop
    was requested (Stop button / Ctrl+C): the calling script then exits instead of going
    on to its analysis.
    """
    import torch
    from library.shared import ExitListener
    from library.training.training_runner import TrainingRunner, is_completed, write_training_queue

    exit_listener = ExitListener()
    write_training_queue([run for _, run in seeded_runs])
    all_completed = True
    for seed, run in seeded_runs:
        if is_completed(run.run_name):
            print(f"Skipping completed run: {run.run_name}")
            continue
        torch.manual_seed(seed)
        if not TrainingRunner(run, exit_listener).run():
            print("Training interrupted — stopping cohort.")
            all_completed = False
            break
    exit_listener.stop()
    return all_completed


def write_rows(filename: str, columns, rows):
    """Merge result rows into experiments/results/<filename>. The first column is the key:
    a row replaces the stored row with the same key, every other stored row is kept. The
    header is the given columns followed by any stored column they lack, so dropping a
    column from a script never drops data (keys a row has beyond the header are ignored).
    The file is written under a temporary name and then swapped in, so a failure midway
    leaves the previous file whole."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    key = columns[0]

    existing = {}
    stored_columns = []
    if path.exists():
        with open(path, "r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            stored_columns = list(reader.fieldnames or [])
            if stored_columns and key not in stored_columns:
                raise ValueError(f"{path} is keyed on {stored_columns[0]!r}, not {key!r}; nothing written")
            for row in reader:
                existing[row[key]] = row

    for row in rows:
        existing[str(row[key])] = row

    fieldnames = list(columns) + [column for column in stored_columns if column not in columns]
    temporary_path = path.with_name(path.name + ".tmp")
    with open(temporary_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in existing.values():
            writer.writerow(row)
    os.replace(temporary_path, path)
    print(f"-> {path}  ({len(existing)} rows)")
    return path


def mean_std(values):
    """(mean, sample std, n) of the values that are not None; mean None when there are
    none, std None below two."""
    usable = [value for value in values if value is not None]
    if not usable:
        return None, None, 0
    std = statistics.stdev(usable) if len(usable) > 1 else None
    return statistics.mean(usable), std, len(usable)


def best_and_final(run):
    """Best-epoch and final-epoch eval records. These runs overfit at small sample
    sizes, so the last epoch alone understates what the configuration reaches. "Best" is
    picked on the test split itself (there is no validation split), so it is optimistic."""
    evals = run.eval_epochs
    if not evals:
        return None, None
    best = max(evals, key=lambda e: e.task_accuracy)
    return best, evals[-1]
