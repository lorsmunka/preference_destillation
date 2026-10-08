"""Post-generation loss strategy: pure CE vs pure KL vs KL/CE annealing (3.8k, 10 epochs).

One file holds everything this experiment is: the runs it trains, and the analysis that
reads them back.

    python experiments/postgen_kl_anneal.py

Completed runs are skipped, so running it again only trains what is missing and then
re-prints the tables. The top-k pass reloads each checkpoint against the distillation
batches (gitignored, so not on every machine) and is skipped when they are missing —
the last measured numbers stay in results/postgen_kl_anneal_topk.csv either way.

Provenance (code review X-M5): none of these six runs is in runs/ any more, so that CSV
(5 rows, measured 2026-04-22 on 768 test examples under the test-split rule of that time)
is the only record of them; running this script now would retrain all six (~18 h of GPU)
before reporting.
"""

import csv
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `library` imports

from _common import RESULTS_DIR, write_rows

from library import Analysis, RunStore, TrainingRun
from library.domain import POST_GENERATION
from library.shared import DEFAULT_TEACHER_MODEL, batches_dir
from library.shared.batch_files import batch_paths
from library.training.training_runner import run_training

TOP_K_CSV = "postgen_kl_anneal_topk.csv"
RESULTS_PATH = RESULTS_DIR / TOP_K_CSV
TOP_K = 20

# The first three runs recorded the experiment tag "postgen-3.8k-10ep-ce-vs-kl"; the later
# ones "postgen-3.8k-10ep-kl-anneal-comparison". Each TrainingRun keeps the tag its run
# actually wrote to info.json, so cohorts stay consistent with the data on disk.
SHARED_ARCHITECTURE = dict(
    domain=POST_GENERATION,
    hidden_dim=48,
    num_layers=3,
    num_heads=1,
    dropout=0.08,
    epoch_count=10,
    batch_size=32,
    learning_rate=0.0015,
    lr_warmup_ratio=0.1,
    max_training_examples=3840,
    training_test_ratio=0.8,
    auxiliary_token_percentage=1.0,
)

RUNS = [
    TrainingRun(
        **SHARED_ARCHITECTURE,
        run_name="exp-postgen-3.8k-10ep-purece-t1-48h-3L-14M",
        experiment="postgen-3.8k-10ep-ce-vs-kl",
        description="Pure CE, temperature 1",
        kl_ratio_start=0.0,
        kl_ratio_end=0.0,
        distillation_temperature_start=1.0,
        distillation_temperature_end=1.0,
    ),
    TrainingRun(
        **SHARED_ARCHITECTURE,
        run_name="exp-postgen-3.8k-10ep-purekl-t5-48h-3L-14M",
        experiment="postgen-3.8k-10ep-ce-vs-kl",
        description="Pure KL, fixed temperature 5",
        kl_ratio_start=1.0,
        kl_ratio_end=1.0,
        distillation_temperature_start=5.0,
        distillation_temperature_end=5.0,
    ),
    TrainingRun(
        **SHARED_ARCHITECTURE,
        run_name="exp-postgen-3.8k-10ep-kl99to50-t5to3-48h-3L-14M",
        experiment="postgen-3.8k-10ep-ce-vs-kl",
        description="KL 0.99 -> 0.5, temperature 5 -> 3",
        kl_ratio_start=0.99,
        kl_ratio_end=0.5,
        distillation_temperature_start=5.0,
        distillation_temperature_end=3.0,
    ),
    TrainingRun(
        **SHARED_ARCHITECTURE,
        run_name="exp-postgen-3.8k-10ep-purekl-t5to3-48h-3L-14M",
        experiment="postgen-3.8k-10ep-kl-anneal-comparison",
        description="Pure KL, temperature 5 -> 3 — isolates the temperature schedule",
        kl_ratio_start=1.0,
        kl_ratio_end=1.0,
        distillation_temperature_start=5.0,
        distillation_temperature_end=3.0,
    ),
    TrainingRun(
        **SHARED_ARCHITECTURE,
        run_name="exp-postgen-3.8k-10ep-kl99to50-t5to3-48h-3L-14M-2",
        experiment="postgen-3.8k-10ep-kl-anneal-comparison",
        description="KL 0.99 -> 0.5, temperature 5 -> 3, second seed",
        kl_ratio_start=0.99,
        kl_ratio_end=0.5,
        distillation_temperature_start=5.0,
        distillation_temperature_end=3.0,
    ),
    TrainingRun(
        **SHARED_ARCHITECTURE,
        run_name="exp-postgen-3.8k-10ep-kl99to50-t5-48h-3L-14M",
        experiment="postgen-3.8k-10ep-kl-anneal-comparison",
        description="KL 0.99 -> 0.5, fixed temperature 5 — isolates the KL schedule",
        kl_ratio_start=0.99,
        kl_ratio_end=0.5,
        distillation_temperature_start=5.0,
        distillation_temperature_end=5.0,
    ),
]

TOP_K_COLUMNS = ["run", "checkpoint", "split", "top_k", "examples", "steps",
                 "topk_accuracy", "teacher_student_overlap", "mean_target_rank", "measured_at"]


def batches_available() -> bool:
    return bool(batch_paths(batches_dir(POST_GENERATION.name, DEFAULT_TEACHER_MODEL)))


def completed_run_names():
    completed = {run.name for run in RunStore().completed()}
    return [run.run_name for run in RUNS if run.run_name in completed]


def measure_top_k():
    """Re-measure the distribution metrics for every completed run and merge them into the
    results CSV, keeping rows for runs that are not measured this time."""
    measured_at = datetime.now().isoformat(timespec="seconds")
    measured = {}
    for name in completed_run_names():
        print(f"Evaluating {name} ...")
        result = Analysis(name).evaluate(split="test", k=TOP_K)
        measured[name] = {
            "run": result.run,
            "checkpoint": result.checkpoint,
            "split": result.split,
            "top_k": TOP_K,
            "examples": result.examples,
            "steps": result.steps,
            "topk_accuracy": result.topk_accuracy,
            "teacher_student_overlap": result.teacher_student_overlap,
            "mean_target_rank": result.mean_target_rank,
            "measured_at": measured_at,
        }
    write_rows(TOP_K_CSV, TOP_K_COLUMNS, list(measured.values()))


def print_recorded_top_k():
    if not RESULTS_PATH.exists():
        return
    print("\nRecorded top-k (test split):")
    with open(RESULTS_PATH, "r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            print(f"  {row['run']:<52} top{row['top_k']}={float(row['topk_accuracy']) * 100:.2f}%  "
                  f"overlap={float(row['teacher_student_overlap']) * 100:.2f}%  "
                  f"rank={float(row['mean_target_rank']):.1f}")


def main():
    if not run_training(RUNS):
        print("\nStopped — re-run the same command to resume; the analysis runs once every run is done.")
        return

    print("\nTraining logs:")
    print(Analysis.compare(completed_run_names()).table())

    if batches_available():
        measure_top_k()
    else:
        print("\nNo distillation batches locally — skipping the top-k pass.")
    print_recorded_top_k()


if __name__ == "__main__":
    main()
