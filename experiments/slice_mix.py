"""Copies vs new problems (user idea, decided 2026-10-15): at the same GPU cost, does a small student
learn more from renumbered copies of problems it already has (same text, new numbers) or from new
problems? Above all: which one makes it take the numbers from the problem?

Data (library/domain/math_tutor/slice.py, groups): "original" = the slice's 850 train conversations;
"copy" = one renumbered copy of each train original that has one (797); "new" = as many new train
problems from families the slice does not use (797). Gemma plays all of them the same way.
Arms (same student and settings as the learning curve's kd students, seeds 1 and 2):
    original   originals only
    new        originals + new problems
    copy       originals + copies
    both       all three
All are tested on the same 100 test conversations (originals). Read per arm: test error, answers
right next to the most common answer, and digits right with the right conversation vs another one
(input_use_check): the gap between the two is the numbers taken from the problem.

    python -m library.domain.math_tutor.slice conversations   # GPU: the extension (new + copies, mixed)
    python -m library.domain.math_tutor.agentic               # the word list, with the replies
    (then delete the earlier teacher data, batches/math_tutor_slice_v1: this script says when)
    python experiments/slice_mix.py                           # GPU: teacher data, 8 students, report
"""

import json
import os
import statistics
import sys

from _common import seeded_run_training, write_rows
from input_use_check import input_use, record_index, sample_pairs
from slice_curve import BATCHES_DIR, conversations, teacher_data_completed, teacher_data_current
from slice_trial import DOMAIN, STUDENT, TEACHER, ensure_input_vocabulary, score_slice

from library import RunStore, TrainingRun
from library.domain.math_tutor.agentic import OUTPUT_TOKENS
from library.domain.math_tutor.slice import CONVERSATIONS_PATH, extension_items, slice_items
from library.shared.batch_files import batch_paths, read_batch_meta

EXPERIMENT = "slice-mix"
KL_RATIO = (0.9, 0.1)   # the learning curve's kd students
SEEDS = (1, 2)
ARMS = {
    "original": ("original",),
    "new": ("original", "new"),
    "copy": ("original", "copy"),
    "both": ("original", "new", "copy"),
}

RESULT_COLUMNS = [
    "run", "arm", "seed", "train_records", "task_accuracy_best", "task_accuracy_final", "majority_baseline",
    "test_error_best", "test_error_final", "test_token_accuracy_final", "test_digits_own", "test_digits_other",
    "test_digits_gap",
]


def inputs_ready() -> bool:
    """Every original and every extension problem played, and the word list built after them."""
    played = {conversation["item_id"] for conversation in conversations()}
    wanted = [item["id"] for item in slice_items() if item["split"] != "heldout_shape"]
    wanted += [item["id"] for item in extension_items()]
    missing = sum(item_id not in played for item_id in wanted)
    if missing:
        print(f"{missing} of {len(wanted)} conversations not played yet: python -m library.domain.math_tutor.slice conversations")
        return False
    if os.path.getmtime(OUTPUT_TOKENS) < os.path.getmtime(CONVERSATIONS_PATH):
        print("The word list is older than the conversations: python -m library.domain.math_tutor.agentic")
        return False
    return True


def train_records(groups) -> int:
    """The train records of these groups in the saved teacher data."""
    count = 0
    for path in batch_paths(BATCHES_DIR):
        meta = read_batch_meta(path)
        count += sum(split == "train" and group in groups for split, group in zip(meta["split"], meta["group"]))
    return count


def run_name(arm, seed):
    return f"slice-mix-{arm}-kd-s{seed}"


def build_runs():
    start, end = KL_RATIO
    return [(seed, TrainingRun(**{**STUDENT, "shuffle_seed": seed}, domain=DOMAIN, teacher_model=TEACHER,
                               training_groups=groups, kl_ratio_start=start, kl_ratio_end=end,
                               run_name=run_name(arm, seed), experiment=EXPERIMENT,
                               description=f"copies vs new: {arm} ({' + '.join(groups)}), kd ({start}->{end}), seed {seed}"))
            for seed in SEEDS for arm, groups in ARMS.items()]   # every arm's first seed before any second one


def report():
    store = RunStore()
    entries = record_index()
    pairs = sample_pairs(entries, "test", "solve") + sample_pairs(entries, "test", "explain")
    rows = []
    for arm, groups in ARMS.items():
        records = train_records(groups)
        for seed in SEEDS:
            run = store.get(run_name(arm, seed))
            evals = run.eval_epochs
            best, final = max(evals, key=lambda e: e.task_accuracy), evals[-1]
            use = input_use(run.name, pairs)
            rows.append({
                "run": run.name, "arm": arm, "seed": seed, "train_records": records,
                "task_accuracy_best": best.task_accuracy, "task_accuracy_final": final.task_accuracy,
                "majority_baseline": run.majority_baseline(final),
                "test_error_best": round(min(e.ce_loss for e in evals), 4), "test_error_final": round(final.ce_loss, 4),
                "test_token_accuracy_final": final.teacher_forced_accuracy,
                "test_digits_own": use["own"]["digits"], "test_digits_other": use["other"]["digits"],
                "test_digits_gap": round(use["own"]["digits"] - use["other"]["digits"], 3),
            })
    write_rows("slice_mix.csv", RESULT_COLUMNS, rows)

    def mean_and_spread(values):
        return f"{statistics.mean(values):.3f}" + (f"±{statistics.stdev(values):.3f}" if len(values) > 1 else "")

    print(f"\n{'arm':>9} {'records':>7} {'test error':>12} {'answers best':>13} {'digits own':>12} {'other':>12} {'gap':>12}")
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        print(f"{arm:>9} {arm_rows[0]['train_records']:>7} "
              + " ".join(f"{mean_and_spread([row[column] for row in arm_rows]):>12}" for column in
                         ("test_error_best", "task_accuracy_best", "test_digits_own", "test_digits_other", "test_digits_gap")))
    print(f"(mean±std over seeds {SEEDS}; most common answer {rows[0]['majority_baseline']})")


def main():
    if not inputs_ready():
        return
    if not teacher_data_completed():
        DOMAIN.build_corpus()   # derived from the conversations; the same order again on a resume
        if not score_slice():
            print("Stopped while saving the teacher data: rerun to resume.")
            return
    if not teacher_data_current():
        return
    ensure_input_vocabulary()
    if not seeded_run_training(build_runs()):
        return
    report()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
