"""Pure KL vs pure CE on the full math-tutor data (2026-10-27, user: "10-10 passes on pure ce vs pure kl and
graph at round 20m size"): two students, identical but for what they learn from.
  kl  Gemma's whole probability list at every position (KL share 1.0 throughout)
  ce  only the word Gemma wrote (KL share 0)
Size m (16M), 10 passes, the same seed and data order; tested after every pass on every test reply
(max_test_examples=None: 8,415 replies, not the default tail of 1,008), so the two lines of the graph are
far apart only if the difference is real.

    python experiments/math_ce_vs_kl.py        # trains what is missing (one after the other), then reports
    python experiments/math_ce_vs_kl.py kl     # one student per process: the two train side by side on one card
    python experiments/math_ce_vs_kl.py ce
"""

import sys

from _common import seeded_run_training, size_settings, write_rows

from library import RunStore, TrainingRun
from library.domain.math_tutor.full import FullDomain

TEACHER = "google/gemma-4-12B-it"
DOMAIN = FullDomain("v1")
EXPERIMENT = "math-ce-vs-kl"
ARMS = {"kl": (1.0, 1.0), "ce": (0.0, 0.0)}   # arm -> (kl_ratio_start, kl_ratio_end)
SIZE = "m"
SEED = 1
STUDENT = dict(dropout=0.1, epoch_count=10, batch_size=32, lr_warmup_ratio=0.1, max_training_examples=None,
               max_test_examples=None, shuffle_seed=SEED)
CHOSEN = sys.argv[1].split(",") if len(sys.argv) > 1 else list(ARMS)
RESULT_COLUMNS = ["row", "run", "arm", "pass", "task_accuracy", "majority_baseline", "test_error",
                  "test_token_accuracy"]


def run_name(arm):
    return f"{EXPERIMENT}-{arm}-{SIZE}-s{SEED}"


def build_runs(arms):
    return [(SEED, TrainingRun(**STUDENT, **size_settings(SIZE), domain=DOMAIN, teacher_model=TEACHER,
                               kl_ratio_start=ARMS[arm][0], kl_ratio_end=ARMS[arm][1], run_name=run_name(arm),
                               experiment=EXPERIMENT,
                               description=f"pure {arm} (KL share {ARMS[arm][0]}), full math tutor (v1), size {SIZE}, "
                                           f"seed {SEED}, every test reply"))
            for arm in arms]


def report():
    """Every pass of both students: task accuracy next to the most common answer, test error, token accuracy."""
    store = RunStore()
    rows = []
    for arm in ARMS:
        run = store.get(run_name(arm))
        for number, evaluation in enumerate(run.eval_epochs, start=1):
            rows.append({"row": f"{run.name}-pass{number}", "run": run.name, "arm": arm, "pass": number,
                         "task_accuracy": evaluation.task_accuracy,
                         "majority_baseline": run.majority_baseline(evaluation),
                         "test_error": round(evaluation.ce_loss, 4),
                         "test_token_accuracy": evaluation.teacher_forced_accuracy})
    for row in rows:
        print(", ".join(f"{column}={row[column]}" for column in RESULT_COLUMNS[1:]))
    write_rows("math_ce_vs_kl.csv", RESULT_COLUMNS, rows)


def main():
    from library.training.training_runner import is_completed
    unknown = [arm for arm in CHOSEN if arm not in ARMS]
    if unknown:
        raise SystemExit(f"unknown students {unknown}: choose from {list(ARMS)}")
    if not seeded_run_training(build_runs(CHOSEN)):
        return
    if all(is_completed(run_name(arm)) for arm in ARMS):
        report()
    else:
        print("This student is done; the report follows when the other one is (rerun with no arguments).")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
