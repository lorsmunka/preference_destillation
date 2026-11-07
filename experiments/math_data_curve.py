"""The math tutor's data curve on one test set (chapter 9 plan #4, 2026-11-06): the pure-KL student of
experiments/math_ce_vs_kl.py (size m, seed 1, the same settings) trained on the first 6,000, 20,000 and
65,000 training replies. The test set is the same 8,415 test replies for every point (the test files are
chosen by their split labels, whatever max_training_examples is), and math-agent-kl-m-s1 is the point at
all 196,000. So test error and agreement with Gemma can be read against the amount of data on one test,
which figure 2.2's slice curve (different test sets and schedules) cannot.

    python experiments/math_data_curve.py 06k      # one student per process: the three train side by side
    python experiments/math_data_curve.py 20k
    python experiments/math_data_curve.py 65k
    python experiments/math_data_curve.py          # trains what is missing (one after the other), then reports

The rented-card run: experiments/cloud_math_data_curve.sh.
"""

import sys

from _common import RESULTS_DIR, seeded_run_training, size_settings, write_rows

from library import RunStore, TrainingRun
from library.domain.math_tutor.full import FullDomain

TEACHER = "google/gemma-4-12B-it"
DOMAIN = FullDomain("v1")
EXPERIMENT = "math-agent"
SIZE = "m"
SEED = 1
# as STUDENT in experiments/math_ce_vs_kl.py, except the cap on the training replies
STUDENT = dict(dropout=0.1, epoch_count=10, batch_size=32, lr_warmup_ratio=0.1, max_test_examples=None)
POINTS = {"06k": 6000, "20k": 20000, "65k": 65000}
FULL_DATA_RUNS = [f"math-agent-kl-m-s{seed}" for seed in (1, 2, 3)]   # the 196k point (seed 1 = the same start)
RESULT_COLUMNS = ["row", "run", "training_replies", "seed", "pass", "task_accuracy", "majority_baseline",
                  "test_error", "test_token_accuracy"]


def run_name(label):
    return f"{EXPERIMENT}-kl-{SIZE}-s{SEED}-data{label}"


def build_run(label):
    return SEED, TrainingRun(**STUDENT, **size_settings(SIZE), max_training_examples=POINTS[label], domain=DOMAIN,
                             teacher_model=TEACHER, shuffle_seed=SEED, kl_ratio_start=1.0, kl_ratio_end=1.0,
                             run_name=run_name(label), experiment=EXPERIMENT,
                             description=f"pure kl, full math tutor (v1), size {SIZE}, seed {SEED}, the first "
                                         f"{POINTS[label]:,} training replies, every test reply")


def report():
    """Every pass of every point, then the last pass by amount of data."""
    store = RunStore()
    runs = [(POINTS[label], store.get(run_name(label))) for label in POINTS]
    runs += [(None, store.get(name)) for name in FULL_DATA_RUNS]
    rows = []
    for replies, run in runs:
        replies = replies or run.info.get("training_batch_count", 0) * STUDENT["batch_size"]
        seed = run.info.get("shuffle_seed")
        for number, evaluation in enumerate(run.eval_epochs, start=1):
            rows.append({"row": f"{run.name}-pass{number}", "run": run.name, "training_replies": replies, "seed": seed,
                         "pass": number, "task_accuracy": evaluation.task_accuracy,
                         "majority_baseline": run.majority_baseline(evaluation),
                         "test_error": round(evaluation.ce_loss, 4),
                         "test_token_accuracy": evaluation.teacher_forced_accuracy})
    write_rows("math_data_curve.csv", RESULT_COLUMNS, rows)
    print("Last pass by amount of data (the same 8,415 test replies):")
    for row in rows:
        if row["pass"] == STUDENT["epoch_count"]:
            print(f"  {row['training_replies']:>7,} replies (seed {row['seed']}): task accuracy {row['task_accuracy']:.4f}, "
                  f"test error {row['test_error']:.4f}, token accuracy {row['test_token_accuracy']:.4f}")


def main():
    chosen = sys.argv[1:] or list(POINTS)
    if chosen == ["report"]:
        report()
        return
    unknown = [label for label in chosen if label not in POINTS]
    if unknown:
        raise SystemExit(f"unknown points {unknown}: choose from {list(POINTS)}")
    if not seeded_run_training([build_run(label) for label in chosen]):
        return
    from library.training.training_runner import is_completed
    if all(is_completed(run_name(label)) for label in POINTS) and all(is_completed(name) for name in FULL_DATA_RUNS):
        report()
    else:
        print("This point is done; the report follows when every point is (python experiments/math_data_curve.py report).")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
