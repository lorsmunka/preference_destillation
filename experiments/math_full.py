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
RUNS = [(arm, BASE_SIZE) for arm in ARMS] + [("kd", size) for size in STUDENT_SIZES if size != BASE_SIZE]
RESULT_COLUMNS = ["run", "arm", "size", "parameters", "train_records", "task_accuracy_best", "task_accuracy_final",
                  "majority_baseline", "test_error_best", "test_token_accuracy_final"]


def run_name(arm, size):
    return f"{EXPERIMENT}-{arm}{'' if size == BASE_SIZE else '-' + size}-s{SEED}"


def build_runs():
    return [(SEED, TrainingRun(**STUDENT, **size_settings(size), domain=DOMAIN, teacher_model=TEACHER,
                               kl_ratio_start=ARMS[arm][0], kl_ratio_end=ARMS[arm][1], run_name=run_name(arm, size),
                               experiment=EXPERIMENT,
                               description=f"full math tutor ({DOMAIN.variant}), {arm} ({ARMS[arm][0]}->{ARMS[arm][1]}), "
                                           f"size {size}, seed {SEED}"))
            for arm, size in RUNS]


def save_teacher_data() -> bool:
    from library.agent.runner import recorded_setup
    from library.data_gen.generator import DistillationDataGenerator
    from library.shared import ExitListener, get_output_dir

    info_path = os.path.join(get_output_dir(DOMAIN.name, TEACHER), "info.json")
    if os.path.exists(info_path) and json.load(open(info_path, encoding="utf-8")).get("status") == "completed":
        return True
    if not os.path.exists(DOMAIN.corpus_path):
        DOMAIN.build_corpus()
    load_in_4bit = recorded_setup(DOMAIN.conversations_path)
    job = GenerationJob(domain=DOMAIN, teacher_model=TEACHER, load_in_4bit=load_in_4bit, chat_template=True,
                        max_examples=10_000_000, batch_size=32, generation_batch_size=4 if load_in_4bit else 16,
                        description=f"full math tutor ({DOMAIN.variant}): Gemma 4 12B ({'NF4' if load_in_4bit else 'bf16'}) "
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
    from library.shared.build_input_vocabulary import build_input_vocabulary
    from library.shared.config import load_input_vocabulary
    if load_input_vocabulary(DOMAIN.name, TEACHER) is None:
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
    if not save_teacher_data():
        print("Stopped while saving the teacher data: rerun to resume.")
        return
    ensure_input_vocabulary()
    if not seeded_run_training(build_runs()):
        return
    report()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
