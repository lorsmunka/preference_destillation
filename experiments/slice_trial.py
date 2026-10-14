"""Trial slice (2026-10-12): does a small student trained on the 1,000-problem slice of the math
tutor learn more than the most common answer, before the full data run is sized?

Training data: Gemma's own replies in the slice's simulated conversations (slice.py), nothing else.
Steps, each skipped when already done (so the script resumes):
  1. saving the teacher data: Gemma 4 12B (NF4, local GPU) reads its own replies back in one forward
     pass each, and its probabilities at every position are stored (SliceDomain, forced mode)
  2. the input vocabulary of the slice's batches
  3. two students: "kd" (Gemma's probabilities + its words, KL share annealed 0.9 -> 0.1, the E-01
     kl-to-ce arm) and "ce" (its words only)
  4. the report: task accuracy next to the most common answer and Gemma's own accuracy, and on the
     test problems Gemma solved fully, the answers with Gemma's history (earlier turns show its
     solutions) next to the student's own history; the gap is the cost of compounding errors

    python -m library.domain.math_tutor.slice conversations   # first: Gemma plays the conversations (GPU)
    python -m library.domain.math_tutor.agentic           # the output vocabulary, with the replies
    python -m library.domain.math_tutor.slice corpus      # and the slice corpus
    python experiments/slice_trial.py                     # GPU, with the user present
"""

import json
import os
import sys

from _common import best_and_final, seeded_run_training, write_rows

from library import GenerationJob, RunStore, TrainingRun
from library.domain.math_tutor.slice import CONVERSATIONS_PATH, SliceDomain, all_items, copy_items

EXPERIMENT = "slice-trial"
TEACHER = "google/gemma-4-12B-it"
DOMAIN = SliceDomain()

SCORING_JOB = GenerationJob(
    domain=DOMAIN,
    teacher_model=TEACHER,
    load_in_4bit=True,
    chat_template=True,
    max_examples=1_000_000,   # the whole slice corpus
    batch_size=32,
    generation_batch_size=8,
    description="trial slice: Gemma 4 12B (NF4) reads its own conversation replies back (forced)",
)

ARMS = {"kd": (0.9, 0.1), "ce": (0.0, 0.0)}   # arm -> (kl_ratio_start, kl_ratio_end)
SEED = 1
STUDENT = dict(
    hidden_dim=128,   # the part-1 students (48 x 3) wrote a few label tokens; these write whole solutions
    num_layers=4,
    num_heads=4,
    dropout=0.1,
    epoch_count=10,
    batch_size=32,
    learning_rate=0.0015,
    lr_warmup_ratio=0.1,
    max_training_examples=None,   # all of the slice's train records
    shuffle_seed=SEED,
)

RESULT_COLUMNS = [
    "run", "arm", "task_accuracy_best", "task_accuracy_final", "majority_baseline", "task_accuracy_excess_final",
    "validity_rate_final", "teacher_task_accuracy", "teacher_history_accuracy", "own_history_accuracy", "history_gap",
    "follow_up_turns", "first_turn_accuracy", "route_solution_share",
]


def run_name(arm):
    # "-batched": trained with one optimizer step per batch (2026-10-14). The runs without it
    # (slice-kd-s1, slice-ce-s1) were updated after every example and stay as the before.
    return f"slice-{arm}-batched-s{SEED}"


def build_runs():
    return [(SEED, TrainingRun(**STUDENT, domain=DOMAIN, teacher_model=TEACHER, kl_ratio_start=start, kl_ratio_end=end,
                               run_name=run_name(arm), experiment=EXPERIMENT,
                               description=f"trial slice, {arm} ({start}->{end}), one update per batch, seed {SEED}"))
            for arm, (start, end) in ARMS.items()]


def score_slice() -> bool:
    """Step 1. False when stopped before the end."""
    from library.data_gen.generator import DistillationDataGenerator
    from library.shared import ExitListener, get_output_dir

    info_path = os.path.join(get_output_dir(DOMAIN.name, TEACHER), "info.json")
    if os.path.exists(info_path) and json.load(open(info_path, encoding="utf-8")).get("status") == "completed":
        return True
    if not os.path.exists(DOMAIN.corpus_path):
        DOMAIN.build_corpus()
    exit_listener = ExitListener()
    DistillationDataGenerator(SCORING_JOB, exit_listener).run()
    stopped = exit_listener.check_exit()
    exit_listener.stop()
    return not stopped


def ensure_input_vocabulary() -> None:
    """Step 2."""
    from library.shared.build_input_vocabulary import build_input_vocabulary
    from library.shared.config import load_input_vocabulary
    if load_input_vocabulary(DOMAIN.name, TEACHER) is None:
        build_input_vocabulary(DOMAIN, TEACHER)


def solution_body(text: str) -> str:
    return text.replace("<solution>", "").replace("</solution>", "").strip()


def teacher_solutions(conversations, items_by_id):
    """item id -> Gemma's solution of each task (as the harness stored it), for the test
    conversations where Gemma answered every task correctly; and Gemma's own task accuracy on
    the test conversations (the reference the student is distilled from)."""
    solutions, correct, tasks = {}, 0, 0
    for conversation in conversations:
        if conversation["split"] != "test":
            continue
        task_turns = [turn for turn in conversation["turns"] if turn["event"] == "task"]
        tasks += len(task_turns)
        correct += sum(bool(turn.get("answer_correct")) for turn in task_turns)
        item = items_by_id[conversation["item_id"]]
        if len(task_turns) == 1 + len(item["follow_ups"]) and all(turn.get("answer_correct") for turn in task_turns):
            solutions[item["id"]] = [solution_body(turn.get("stored_completed", turn["output"])) for turn in task_turns]
    return solutions, (round(correct / tasks, 4) if tasks else None)


def history_accuracy(student, items_by_id, solutions):
    """Answers on the follow-up tasks of the test problems Gemma solved fully, with the teacher's
    history (earlier turns show Gemma's correct solutions) and with the student's own history.
    The first task has no history: it is reported on its own."""
    from library.domain.math_tutor.checks import final_answer
    from library.domain.math_tutor.conversation import Conversation
    from library.domain.math_tutor.questions import tasks_of

    teacher_correct = own_correct = follow_up_turns = first_correct = first_turns = solution_routes = replies = 0
    for item_id, teacher_blocks in solutions.items():
        item = items_by_id[item_id]
        teacher, own = Conversation(), Conversation()
        for turn, task in enumerate(tasks_of(item)):
            own_reply = student.generate(own.solve_context(task), max_new_tokens=DOMAIN.max_steps).text
            replies += 1
            solution_routes += own_reply.lstrip().startswith("<solution>")
            if turn == 0:
                first_turns += 1
                first_correct += final_answer(own_reply) == item["answers"][0]
            else:
                teacher_reply = student.generate(teacher.solve_context(task), max_new_tokens=DOMAIN.max_steps).text
                follow_up_turns += 1
                teacher_correct += final_answer(teacher_reply) == item["answers"][turn]
                own_correct += final_answer(own_reply) == item["answers"][turn]
            teacher.add_solution(task, teacher_blocks[turn])
            own.add_solution(task, solution_body(own_reply))
    return {
        "teacher_history_accuracy": round(teacher_correct / follow_up_turns, 4) if follow_up_turns else None,
        "own_history_accuracy": round(own_correct / follow_up_turns, 4) if follow_up_turns else None,
        "follow_up_turns": follow_up_turns,
        "first_turn_accuracy": round(first_correct / first_turns, 4) if first_turns else None,
        "route_solution_share": round(solution_routes / replies, 4) if replies else None,
    }


def report():
    from library.model.student import StudentModel

    store = RunStore()
    items_by_id = {item["id"]: item for item in all_items() + copy_items()}
    conversations = [json.loads(line) for line in open(CONVERSATIONS_PATH, encoding="utf-8")]
    solutions, teacher_accuracy = teacher_solutions(conversations, items_by_id)
    rows = []
    for arm in ARMS:
        run = store.get(run_name(arm))
        best, final = best_and_final(run)
        _, checkpoint_path = store.resolve_checkpoint(run.name, "latest")
        student = StudentModel.from_run(run, checkpoint_path, domain=DOMAIN)
        history = history_accuracy(student, items_by_id, solutions)
        gap = None
        if history["teacher_history_accuracy"] is not None:
            gap = round(history["teacher_history_accuracy"] - history["own_history_accuracy"], 4)
        rows.append({
            "run": run.name, "arm": arm,
            "task_accuracy_best": best.task_accuracy, "task_accuracy_final": final.task_accuracy,
            "majority_baseline": run.majority_baseline(final), "task_accuracy_excess_final": run.task_accuracy_excess(final),
            "validity_rate_final": final.validity_rate,
            "teacher_task_accuracy": teacher_accuracy, "history_gap": gap, **history,
        })
    for row in rows:
        print(f"\n{row['arm']}: " + ", ".join(f"{column}={row[column]}" for column in RESULT_COLUMNS[2:]))
    write_rows("slice_trial.csv", RESULT_COLUMNS, rows)


def main():
    if not score_slice():
        print("Stopped during teacher scoring: rerun to resume.")
        return
    ensure_input_vocabulary()
    if not seeded_run_training(build_runs()):
        return
    report()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
