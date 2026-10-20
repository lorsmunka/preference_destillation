"""Coding agent trial (2026-10-20): the whole pipeline, end to end, on one variant of the data.

Training data: Gemma's own replies in the pass (library/domain/code_agent/harness.py), nothing else.
Steps, each skipped when done (so the script resumes):
  1. the corpus: every played item's records (its scope, and its answer in the narrowed context)
  2. saving the teacher data: Gemma reads its own replies back in one forward pass each (forced),
     its probabilities at every position stored
  3. the input vocabulary of the batches
  4. two students, as on the math slice: "kd" (Gemma's probabilities + its words, KL share 0.9 -> 0.1)
     and "ce" (its words only)
  5. the report, on the test items (whole projects the students never saw): each student in the
     harness doing both steps itself (its own scope narrows its own context), against the true
     places: lines covered, lines on target, places hit; the decision (explain or found); the
     explanations that pass the check. Next to Gemma's own numbers on the same items.

    python experiments/code_agent_trial.py              # the real data (v1) when it exists, else the local pilot
    python experiments/code_agent_trial.py pilot_nf4    # the local pilot (Gemma NF4 on the 4070)
"""

import json
import os
import sys

from _common import best_and_final, seeded_run_training, write_rows

from library import GenerationJob, RunStore, TrainingRun
from library.domain.code_agent import CodeAgentDomain

TEACHER = "google/gemma-4-12B-it"
ARMS = {"kd": (0.9, 0.1), "ce": (0.0, 0.0)}   # arm -> (kl_ratio_start, kl_ratio_end)
SEED = 1
STUDENT = dict(
    hidden_dim=128,   # the math slice's student; a 6,500-token input needs ~7.5 GB to train (2026-10-20)
    num_layers=4,
    num_heads=4,
    dropout=0.1,
    epoch_count=10,
    batch_size=32,
    learning_rate=0.0015,
    lr_warmup_ratio=0.1,
    max_training_examples=None,
    shuffle_seed=SEED,
)
RESULT_COLUMNS = ["run", "arm", "items", "scopes", "recall", "precision", "places_hit", "places", "decision_right",
                  "explanations_pass", "explanations", "task_accuracy_best", "test_error_best", "test_token_accuracy_final"]


def chosen_variant() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    return "v1" if CodeAgentDomain("v1").conversations_path.exists() else "pilot_nf4"


DOMAIN = CodeAgentDomain(chosen_variant())


def scoring_job() -> GenerationJob:
    return GenerationJob(domain=DOMAIN, teacher_model=TEACHER, load_in_4bit=True, chat_template=True, max_examples=1_000_000,
                         batch_size=32, generation_batch_size=1,   # ~6,500-token inputs: one at a time on a 12 GB card
                         description=f"coding agent ({DOMAIN.conversations}): Gemma 4 12B (NF4) reads its own replies back (forced)")


def run_name(arm):
    return f"code-agent-{DOMAIN.conversations}-{arm}-s{SEED}"


def build_runs():
    return [(SEED, TrainingRun(**STUDENT, domain=DOMAIN, teacher_model=TEACHER, kl_ratio_start=start, kl_ratio_end=end,
                               run_name=run_name(arm), experiment=f"code-agent-{DOMAIN.conversations}",
                               description=f"coding agent trial ({DOMAIN.conversations}), {arm} ({start}->{end}), seed {SEED}"))
            for arm, (start, end) in ARMS.items()]


def save_teacher_data() -> bool:
    """Steps 1 and 2. False when stopped before the end."""
    from library.data_gen.generator import DistillationDataGenerator
    from library.shared import ExitListener, get_output_dir

    info_path = os.path.join(get_output_dir(DOMAIN.name, TEACHER), "info.json")
    if os.path.exists(info_path) and json.load(open(info_path, encoding="utf-8")).get("status") == "completed":
        return True
    if not os.path.exists(DOMAIN.corpus_path):
        DOMAIN.build_corpus()
    exit_listener = ExitListener()
    DistillationDataGenerator(scoring_job(), exit_listener).run()
    stopped = exit_listener.check_exit()
    exit_listener.stop()
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()   # free Gemma before the input vocabulary's worker processes start
    return not stopped


def ensure_input_vocabulary() -> None:
    """Step 3."""
    from library.shared.build_input_vocabulary import build_input_vocabulary
    from library.shared.config import load_input_vocabulary
    if load_input_vocabulary(DOMAIN.name, TEACHER) is None:
        build_input_vocabulary(DOMAIN, TEACHER)


def played_test_items():
    """The test items Gemma played, with Gemma's own turn."""
    from library.domain.code_agent.requests import read_items
    items = {item["id"]: item for item in read_items()}
    played = [json.loads(line) for line in open(DOMAIN.conversations_path, encoding="utf-8")]
    return [(items[entry["item_id"]], entry["turn"]) for entry in played if entry["split"] == "test"]


def summarise(turns, label):
    scoped = [turn for turn in turns if "recall" in turn]
    explained = [turn for turn in turns if turn.get("route") == "explain"]
    return {"run": label, "items": len(turns), "scopes": len(scoped),
            "recall": round(sum(t["recall"] for t in scoped) / len(scoped), 3) if scoped else None,
            "precision": round(sum(t["precision"] for t in scoped) / len(scoped), 3) if scoped else None,
            "places_hit": sum(t["places_hit"] for t in scoped), "places": sum(t["places"] for t in scoped),
            "decision_right": sum(bool(t.get("route_correct")) for t in turns),
            "explanations_pass": sum(t.get("check") is None for t in explained), "explanations": len(explained)}


def student_turns(student, test_items):
    """Each test item played by the student in the harness: both steps, its own scope."""
    from library.agent.runner import StudentAgentModel, run_sessions
    from library.domain.code_agent.harness import CodeAgentSession, scored_turn, version_cache

    versions = version_cache()
    sessions = [CodeAgentSession(item, versions(item["project"], item["version"])) for item, _ in test_items]
    run_sessions(StudentAgentModel(student, DOMAIN.max_steps), sessions)
    return [scored_turn(session.item, session.version, session.turn) for session in sessions]


def report():
    from library.model.student import StudentModel

    store = RunStore()
    test_items = played_test_items()
    rows = [summarise([turn for _, turn in test_items], "gemma")]
    for arm in ARMS:
        run = store.get(run_name(arm))
        best, final = best_and_final(run)
        _, checkpoint_path = store.resolve_checkpoint(run.name, "latest")
        student = StudentModel.from_run(run, checkpoint_path, domain=DOMAIN)
        row = summarise(student_turns(student, test_items), run.name)
        row.update(arm=arm, task_accuracy_best=best.task_accuracy, test_error_best=round(min(e.ce_loss for e in run.eval_epochs), 4),
                   test_token_accuracy_final=final.teacher_forced_accuracy)
        rows.append(row)
    for row in rows:
        print(f"{row['run']:<28} scopes {row['scopes']}/{row['items']}, lines covered {row['recall']}, on target "
              f"{row['precision']}, places hit {row['places_hit']}/{row['places']}, decision right "
              f"{row['decision_right']}/{row['items']}, explanations pass {row['explanations_pass']}/{row['explanations']}")
    write_rows(f"code_agent_trial_{DOMAIN.conversations}.csv", RESULT_COLUMNS, rows)


def main():
    print(f"variant {DOMAIN.conversations}: {DOMAIN.conversations_path}")
    if not DOMAIN.conversations_path.exists():
        print("No conversations yet: python -m library.domain.code_agent.harness (GPU)")
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
