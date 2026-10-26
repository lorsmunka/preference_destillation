"""The full math-tutor run (2026-10-20): teacher data and students on every item's conversation
(library/domain/math_tutor/full.py), after the pass on the rented card.

Steps, each skipped when done (so the script resumes):
  1. the corpus (FullDomain), from conversations/full_v1.jsonl
  2. saving the teacher data: Gemma reads its own replies back (forced), with the setup that wrote them
  3. the input vocabulary
  4. two students, as on the slice: kd (KL share 0.9 -> 0.1) and ce; then the size sweep: kd at
     every other size of _common.STUDENT_SIZES (~3M to ~100M parameters), smallest first
  5. the report: task accuracy next to the most common answer, test error, token accuracy

    python -m library.domain.math_tutor.full conversations   # first: the pass (GPU)
    python -m library.domain.math_tutor.agentic              # the output vocabulary, with the replies
    python experiments/math_full.py
    python experiments/math_full.py v1 kd,kd-l               # only these students (the first loop: kd at s and at l)
    python experiments/math_full.py v1 none                  # only the teacher data (also of records added since: full.py)
    python experiments/math_full.py v1 kd-l                  # one student per process: several train side by side on one card
    python experiments/math_full.py pilot_nf4                # the local trial (its own data, the current vocabulary)
"""

import json
import os
import sys

from _common import BASE_SIZE, STUDENT_SIZES, best_and_final, seeded_run_training, size_settings, write_rows

from library import GenerationJob, RunStore, TrainingRun
from library.domain.math_tutor.full import FullDomain

TEACHER = "google/gemma-4-12B-it"
DOMAIN = FullDomain(sys.argv[1] if len(sys.argv) > 1 else "v1")
EXPERIMENT = "math-full" if DOMAIN.variant == "v1" else f"math-full-{DOMAIN.variant}"
ARMS = {"kd": (0.9, 0.1), "ce": (0.0, 0.0)}
SEED = 1
STUDENT = dict(dropout=0.1, epoch_count=10, batch_size=32, lr_warmup_ratio=0.1, max_training_examples=None,
               shuffle_seed=SEED)
ALL_RUNS = [(arm, BASE_SIZE) for arm in ARMS] + [("kd", size) for size in STUDENT_SIZES if size != BASE_SIZE]
CHOSEN = sys.argv[2].split(",") if len(sys.argv) > 2 else None   # e.g. "kd,kd-l": train (and report) only these
NO_STUDENTS = CHOSEN == ["none"]
RESULT_COLUMNS = ["run", "arm", "size", "parameters", "train_records", "task_accuracy_best", "task_accuracy_final",
                  "majority_baseline", "test_error_best", "test_token_accuracy_final"]


def run_key(arm, size):
    return arm if size == BASE_SIZE else f"{arm}-{size}"


RUNS = [run for run in ALL_RUNS if CHOSEN is None or run_key(*run) in CHOSEN]
if CHOSEN and not NO_STUDENTS and len(RUNS) < len(CHOSEN):
    raise SystemExit(f"unknown students in {CHOSEN}: choose from {[run_key(*run) for run in ALL_RUNS]}")


def run_name(arm, size):
    return f"{EXPERIMENT}-{run_key(arm, size)}-s{SEED}"


def build_runs():
    return [(SEED, TrainingRun(**STUDENT, **size_settings(size), domain=DOMAIN, teacher_model=TEACHER,
                               kl_ratio_start=ARMS[arm][0], kl_ratio_end=ARMS[arm][1], run_name=run_name(arm, size),
                               experiment=EXPERIMENT,
                               description=f"full math tutor ({DOMAIN.variant}), {arm} ({ARMS[arm][0]}->{ARMS[arm][1]}), "
                                           f"size {size}, seed {SEED}"))
            for arm, size in RUNS]


def save_teacher_data(domain) -> bool:
    """The teacher data of every record in the domain's corpus. A corpus whose teacher data exists is
    extended, never rebuilt (a rebuilt one would put the records in another order than the batches);
    the records added since are scored into the batches after the others."""
    from library.agent.runner import recorded_setup
    from library.data_gen.generator import DistillationDataGenerator
    from library.shared import ExitListener, get_output_dir

    info_path = os.path.join(get_output_dir(domain.name, TEACHER), "info.json")
    if os.path.exists(domain.corpus_path):
        domain.extend_corpus()
    elif os.path.exists(info_path):
        raise SystemExit(f"{domain.corpus_path} is missing, but its teacher data exists: copy the corpus file it was "
                         f"made from (a rebuilt one would put the records in another order than the batches)")
    else:
        domain.build_corpus()
    corpus_count = sum(1 for _ in open(domain.corpus_path, encoding="utf-8"))
    if os.path.exists(info_path):
        info = json.load(open(info_path, encoding="utf-8"))
        if info.get("status") == "completed" and info.get("input_count") == corpus_count:
            return True
    load_in_4bit = recorded_setup(domain.conversations_path)
    job = GenerationJob(domain=domain, teacher_model=TEACHER, load_in_4bit=load_in_4bit, chat_template=True,
                        max_examples=10_000_000, batch_size=32, generation_batch_size=4 if load_in_4bit else 16,
                        description=f"full math tutor ({domain.name}): Gemma 4 12B ({'NF4' if load_in_4bit else 'bf16'}) "
                                    f"reads its own replies back")
    exit_listener = ExitListener()
    DistillationDataGenerator(job, exit_listener).run()
    stopped = exit_listener.check_exit()
    exit_listener.stop()
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    return not stopped


def ensure_input_vocabulary() -> None:
    """Built from all the batches, and again when batches were added after it (each run keeps its own copy)."""
    from library.shared.batch_files import batch_paths
    from library.shared.build_input_vocabulary import build_input_vocabulary
    from library.shared.config import get_batches_dir, input_vocabulary_path
    path = input_vocabulary_path(DOMAIN.name, TEACHER)
    newest_batch = max(batch.stat().st_mtime for batch in batch_paths(get_batches_dir(DOMAIN.name, TEACHER)))
    if not path.exists() or path.stat().st_mtime < newest_batch:
        build_input_vocabulary(DOMAIN, TEACHER)


def report():
    store = RunStore()
    train_records = sum(json.loads(line)["split"] == "train" for line in open(DOMAIN.corpus_path, encoding="utf-8"))
    rows = []
    for arm, size in RUNS:
        run = store.get(run_name(arm, size))
        best, final = best_and_final(run)
        rows.append({"run": run.name, "arm": arm, "size": size, "parameters": run.total_parameters,
                     "train_records": train_records,   # in the corpus (a few may be skipped for unknown tokens)
                     "task_accuracy_best": best.task_accuracy, "task_accuracy_final": final.task_accuracy,
                     "majority_baseline": run.majority_baseline(final),
                     "test_error_best": round(min(e.ce_loss for e in run.eval_epochs), 4),
                     "test_token_accuracy_final": final.teacher_forced_accuracy})
    for row in rows:
        print(", ".join(f"{column}={row[column]}" for column in RESULT_COLUMNS))
    write_rows("math_full.csv" if DOMAIN.variant == "v1" else f"math_full_{DOMAIN.variant}.csv", RESULT_COLUMNS, rows)


def main():
    print(f"variant {DOMAIN.variant}: {DOMAIN.conversations_path}")
    if not DOMAIN.conversations_path.exists():
        print("No conversations yet: python -m library.domain.math_tutor.full conversations (GPU)")
        return
    if not save_teacher_data(DOMAIN):
        print("Stopped while saving the teacher data: rerun to resume.")
        return
    if NO_STUDENTS:
        print("The teacher data is ready (no students chosen).")
        return
    ensure_input_vocabulary()
    if not seeded_run_training(build_runs()):
        return
    report()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
