"""The full math-tutor run (2026-10-20): teacher data and students on every item's conversation
(library/domain/math_tutor/full.py), after the pass on the rented card.

Steps, each skipped when done (so the script resumes):
  1. the corpus (FullDomain), from conversations/full_v1.jsonl
  2. saving the teacher data: Gemma reads its own replies back (forced), with the setup that wrote them
  3. the input vocabulary
  4. two students, as on the slice: kd (KL share 0.9 -> 0.1) and ce
  5. the report: task accuracy next to the most common answer, test error, token accuracy

    python -m library.domain.math_tutor.full conversations   # first: the pass (GPU)
    python -m library.domain.math_tutor.agentic              # the output vocabulary, with the replies
    python experiments/math_full.py
"""

import json
import os
import sys

from _common import best_and_final, seeded_run_training, write_rows

from library import GenerationJob, RunStore, TrainingRun
from library.domain.math_tutor.full import CONVERSATIONS_PATH, FullDomain

EXPERIMENT = "math-full"
TEACHER = "google/gemma-4-12B-it"
DOMAIN = FullDomain()
ARMS = {"kd": (0.9, 0.1), "ce": (0.0, 0.0)}
SEED = 1
STUDENT = dict(hidden_dim=128, num_layers=4, num_heads=4, dropout=0.1, epoch_count=10, batch_size=32,
               learning_rate=0.0015, lr_warmup_ratio=0.1, max_training_examples=None, shuffle_seed=SEED)
RESULT_COLUMNS = ["run", "arm", "train_records", "task_accuracy_best", "task_accuracy_final", "majority_baseline",
                  "test_error_best", "test_token_accuracy_final"]


def run_name(arm):
    return f"math-full-{arm}-s{SEED}"


def build_runs():
    return [(SEED, TrainingRun(**STUDENT, domain=DOMAIN, teacher_model=TEACHER, kl_ratio_start=start, kl_ratio_end=end,
                               run_name=run_name(arm), experiment=EXPERIMENT,
                               description=f"full math tutor, {arm} ({start}->{end}), seed {SEED}"))
            for arm, (start, end) in ARMS.items()]


def save_teacher_data() -> bool:
    from library.agent.runner import recorded_setup
    from library.data_gen.generator import DistillationDataGenerator
    from library.shared import ExitListener, get_output_dir

    info_path = os.path.join(get_output_dir(DOMAIN.name, TEACHER), "info.json")
    if os.path.exists(info_path) and json.load(open(info_path, encoding="utf-8")).get("status") == "completed":
        return True
    if not os.path.exists(DOMAIN.corpus_path):
        DOMAIN.build_corpus()
    load_in_4bit = recorded_setup(CONVERSATIONS_PATH)
    job = GenerationJob(domain=DOMAIN, teacher_model=TEACHER, load_in_4bit=load_in_4bit, chat_template=True,
                        max_examples=10_000_000, batch_size=32, generation_batch_size=4 if load_in_4bit else 16,
                        description=f"full math tutor: Gemma 4 12B ({'NF4' if load_in_4bit else 'bf16'}) reads its own replies back")
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
    rows = []
    for arm in ARMS:
        run = store.get(run_name(arm))
        best, final = best_and_final(run)
        rows.append({"run": run.name, "arm": arm, "train_records": run.info.get("train_examples"),
                     "task_accuracy_best": best.task_accuracy, "task_accuracy_final": final.task_accuracy,
                     "majority_baseline": run.majority_baseline(final),
                     "test_error_best": round(min(e.ce_loss for e in run.eval_epochs), 4),
                     "test_token_accuracy_final": final.teacher_forced_accuracy})
    for row in rows:
        print(", ".join(f"{column}={row[column]}" for column in RESULT_COLUMNS))
    write_rows("math_full.csv", RESULT_COLUMNS, rows)


def main():
    if not CONVERSATIONS_PATH.exists():
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
