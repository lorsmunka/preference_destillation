"""Learning curve on the trial slice (decided 2026-10-14): does the student get better as it sees more
of Gemma's conversations? Still improving at the full slice -> playing more conversations (the
renumbered copies) is worth the GPU time; flat -> more data is not the fix.

Students on the first 100, 200, 500 and all ~850 train conversations, in the order Gemma played them
(a seeded random order); all are tested on the same test conversations. Same student and settings as
the trial's kd arm (Gemma's probabilities + its words), 10 passes over each set, so a bigger set also
gets more updates (the trial's test error flattened by pass 6: longer training alone was not missing).

Read per size: answers right next to the most common answer, the error on Gemma's test replies
(cross-entropy), and whether the student takes the numbers from the problem: digit accuracy on
Gemma's test replies with the right conversation vs another one (input_use_check).

    python -m library.domain.math_tutor.slice conversations originals   # GPU: Gemma plays all originals
    python -m library.domain.math_tutor.agentic                         # the word list, with the replies
    (then delete the earlier teacher data, batches/math_tutor_slice_v1: this script says when)
    python experiments/slice_curve.py                                   # GPU: teacher data, students, report
"""

import json
import os
import sys

from _common import seeded_run_training, write_rows
from input_use_check import input_use, record_index, sample_pairs
from slice_trial import DOMAIN, SEED, STUDENT, TEACHER, ensure_input_vocabulary, score_slice

from library import RunStore, TrainingRun
from library.domain.math_tutor.agentic import OUTPUT_TOKENS
from library.domain.math_tutor.slice import CONVERSATIONS_PATH, CURVE_SIZES, corpus_records, slice_items, train_stages
from library.shared import get_output_dir
from library.shared.batch_files import batch_paths, read_batch_meta

EXPERIMENT = "slice-curve"
KL_RATIO = (0.9, 0.1)   # the trial's kd arm
BATCHES_DIR = get_output_dir(DOMAIN.name, TEACHER)

RESULT_COLUMNS = [
    "run", "train_conversations", "train_records", "task_accuracy_best", "task_accuracy_final", "majority_baseline",
    "task_accuracy_excess_best", "validity_rate_final", "test_error_best", "test_error_final",
    "test_token_accuracy_final", "test_digits_own", "test_digits_other",
]


def conversations():
    return [json.loads(line) for line in open(CONVERSATIONS_PATH, encoding="utf-8")]


def inputs_ready() -> bool:
    """Every original played, and the word list built after the last conversation."""
    played = {conversation["item_id"] for conversation in conversations()}
    originals = [item["id"] for item in slice_items() if item["split"] != "heldout_shape"]
    missing = sum(item_id not in played for item_id in originals)
    if missing:
        print(f"{missing} of {len(originals)} originals not played yet: "
              f"python -m library.domain.math_tutor.slice conversations originals")
        return False
    if os.path.getmtime(OUTPUT_TOKENS) < os.path.getmtime(CONVERSATIONS_PATH):
        print("The word list is older than the conversations: python -m library.domain.math_tutor.agentic")
        return False
    return True


def teacher_data_completed() -> bool:
    info_path = os.path.join(BATCHES_DIR, "info.json")
    return os.path.exists(info_path) and json.load(open(info_path, encoding="utf-8")).get("status") == "completed"


def teacher_data_current() -> bool:
    """Whether the saved teacher data covers the conversations of the current corpus."""
    train, test = corpus_records()
    saved = {item_id for path in batch_paths(BATCHES_DIR) for item_id in read_batch_meta(path)["item_id"]}
    if saved == {record["item_id"] for record in train + test}:
        return True
    print(f"The saved teacher data ({len(saved)} conversations) is from an earlier corpus: delete {BATCHES_DIR} "
          f"(ask first) and rerun.")
    return False


def curve_sizes() -> dict:
    """Training size (train conversations) -> (max_training_examples, train records used). A size
    takes the leading training files that hold only its conversations (whole files of 32, so up
    to 31 of its records are left out); None = all training files."""
    stages = train_stages(conversations())
    files = []   # (latest stage, train records) of each training file, in order
    for path in batch_paths(BATCHES_DIR):
        meta = read_batch_meta(path)
        train_ids = [item_id for item_id, split in zip(meta["item_id"], meta["split"]) if split == "train"]
        if train_ids:
            files.append((max(stages[item_id] for item_id in train_ids), len(train_ids)))
    sizes = {}
    for stage, size in enumerate(size for size in CURVE_SIZES if size < len(stages)):
        leading = next((index for index, (file_stage, _) in enumerate(files) if file_stage > stage), len(files))
        sizes[size] = (leading * STUDENT["batch_size"], sum(count for _, count in files[:leading]))
    sizes[len(stages)] = (None, sum(count for _, count in files))
    return sizes


def run_name(size):
    return f"slice-curve-{size}-kd-s{SEED}"


def build_runs(sizes):
    start, end = KL_RATIO
    return [(SEED, TrainingRun(**{**STUDENT, "max_training_examples": examples}, domain=DOMAIN, teacher_model=TEACHER,
                               kl_ratio_start=start, kl_ratio_end=end, run_name=run_name(size), experiment=EXPERIMENT,
                               description=f"learning curve: first {size} train conversations, kd ({start}->{end}), "
                                           f"seed {SEED}"))
            for size, (examples, _) in sizes.items()]


def report(sizes):
    store = RunStore()
    entries = record_index()
    pairs = sample_pairs(entries, "test", "solve") + sample_pairs(entries, "test", "explain")
    rows = []
    for size, (_, records) in sizes.items():
        run = store.get(run_name(size))
        evals = run.eval_epochs
        best, final = max(evals, key=lambda e: e.task_accuracy), evals[-1]
        use = input_use(run.name, pairs)
        rows.append({
            "run": run.name, "train_conversations": size, "train_records": records,
            "task_accuracy_best": best.task_accuracy, "task_accuracy_final": final.task_accuracy,
            "majority_baseline": run.majority_baseline(final), "task_accuracy_excess_best": run.task_accuracy_excess(best),
            "validity_rate_final": final.validity_rate,
            "test_error_best": round(min(e.ce_loss for e in evals), 4), "test_error_final": round(final.ce_loss, 4),
            "test_token_accuracy_final": final.teacher_forced_accuracy,
            "test_digits_own": use["own"]["digits"], "test_digits_other": use["other"]["digits"],
        })
    print(f"\n{'conversations':>13} {'records':>7} {'answers best':>12} {'final':>6} {'baseline':>8} "
          f"{'test error':>10} {'digits own':>10} {'other':>6}")
    for row in rows:
        print(f"{row['train_conversations']:>13} {row['train_records']:>7} {row['task_accuracy_best']:>12.3f} "
              f"{row['task_accuracy_final']:>6.3f} {row['majority_baseline'] or 0:>8.3f} {row['test_error_best']:>10.3f} "
              f"{row['test_digits_own']:>10.3f} {row['test_digits_other']:>6.3f}")
    write_rows("slice_curve.csv", RESULT_COLUMNS, rows)


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
    sizes = curve_sizes()
    if not seeded_run_training(build_runs(sizes)):
        return
    report(sizes)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
