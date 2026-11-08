"""Experiment 2 of the chapter 9 plan (agents/work/experiment_plan_2026-11-06/plan.md, 2026-11-07): the
math tutor end to end. Each full-run student (size m, final checkpoint; KL, CE and KL->CE) is the tutor
in the pass's own harness on Gemma's held-out test conversations (full.py's pass, split "test"), and
every reply is compared with GEMMA's reply in that turn, never with the gold answer: the student is
distilled from Gemma, so agreeing with Gemma is the goal (a student "beating" Gemma = a wrong measure).

The pupil's messages are Gemma's conversation replayed in order (no new draws from the template
bank, so every turn has a Gemma reply to compare with). Two histories:
  gold   each turn starts from the context Gemma read in that turn (Gemma's earlier replies)
  own    the context holds the student's own earlier replies, as the harness stored them
The harness is harness.TutorSession: the route from the first tag, a solution or an explanation
checked and asked again once (sampled, T=0.7), an off-topic message answered by the harness itself.
Per turn: the route agrees with Gemma's; on Gemma's solutions, the answer agrees with Gemma's
answer; the turn agrees when both do. Per conversation: every turn agrees. The gap gold - own is the
cost of compounding errors (the first turn has no history, so it is the same in both).

CPU by default (CUDA_VISIBLE_DEVICES=-1 unless it is set), THREADS torch threads. Each finished
conversation is appended to runs/<student>/end_to_end.jsonl, so a stopped run (Ctrl+C) resumes where
it stopped and a student with every conversation played is skipped; the report is rebuilt from
those files each time. No STOP-file listener: the repo's STOP file belongs to other paused runs.

    python experiments/math_end_to_end.py                      # every student, then the report
    python experiments/math_end_to_end.py math-agent-kl-m-s1   # only the students named, then the report
                                                               # (two processes side by side: ~2 h, 2026-11-07)
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")   # CPU unless a GPU is asked for (CUDA_VISIBLE_DEVICES=0);
                                                     # "-1", not "": Windows drops an empty variable

import copy
import json
import random
import sys
import time
import zlib

from _common import mean_std, write_rows

from library.agent.runner import Request, run_sessions
from library.domain.math_tutor.conversation import Conversation
from library.domain.math_tutor.full import conversations_path
from library.domain.math_tutor.harness import EXPLANATION_PREFIX, TutorSession
from library.shared.config import run_dir

ARMS = {"kl": (1, 2, 3), "ce": (1, 2, 3), "kl-to-ce": (1, 2)}
STUDENTS = [(arm, seed, f"math-agent-{arm}-m-s{seed}") for arm, seeds in ARMS.items() for seed in seeds]
HISTORIES = ("gold", "own")
CONVERSATION_COUNT = None   # None = every test conversation; else a seeded sample, the same for every student
SAMPLE_SEED = 0
THREADS = 4                 # as fast as 8 for one student on the 13700K (0.13 s per turn, 2026-11-07); two
                            # processes side by side then stay within 8 threads (another CPU job may run)
POSITIONS = (("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4", 4, 4), ("5", 5, 5), ("6-8", 6, 8), ("9+", 9, 99))
EVENTS = ("task", "question", "what_if", "off_topic")
TURN_FIELDS = ("event", "pupil_turn", "message", "kind", "expected_route", "expected_answer")

RESULT_COLUMNS = [
    "key", "model", "arm", "seed", "history", "compared_with", "conversations", "turns",
    "route_agreement", "answer_turns", "answer_agreement", "turn_agreement",
    "conversation_agreement", "conversation_answer_agreement",
    "first_turn_answer_agreement", "follow_up_answer_agreement",
    "solution_check_pass", "explanation_check_pass", "retried_share", "seconds_per_conversation",
]
GROUP_COLUMNS = [
    "key", "model", "arm", "seed", "history", "grouping", "group", "turns",
    "route_agreement", "answer_turns", "answer_agreement", "turn_agreement",
]
SUMMARY_MEASURES = ("route_agreement", "answer_agreement", "follow_up_answer_agreement", "turn_agreement",
                    "conversation_agreement", "conversation_answer_agreement")


# --- playing a conversation ------------------------------------------------------------------

class RememberedReplies:
    """The student's greedy replies, remembered by context for one conversation. The harness asks
    the same context twice whenever the route context is also the reply's context (the route call
    already wrote the whole reply), and the two histories share contexts while the student's
    replies match Gemma's word for word. Greedy decoding gives the same text again, so only the
    time changes. Sampled retries are never remembered."""

    def __init__(self, student):
        self.student, self.replies = student, {}

    def generate(self, text, max_new_tokens, temperature=0.0):
        if temperature:
            return self.student.generate(text, max_new_tokens, temperature)
        key = (text, max_new_tokens)
        if key not in self.replies:
            self.replies[key] = self.student.generate(text, max_new_tokens)
        return self.replies[key]


def store_reply(conversation: Conversation, turn) -> None:
    """A finished turn into the conversation, as TutorSession.receive stores it."""
    if turn["route"] == "out_of_scope":   # the harness answered: the message never enters the context
        return
    if turn["route"] == "explanation":
        conversation.add_explanation(turn["message"], turn["output"][len(EXPLANATION_PREFIX):].replace("</explanation>", "").strip())
        return
    stored = turn.get("stored_completed", turn["output"])
    conversation.add_solution(turn["message"], stored.replace("<solution>", "").replace("</solution>", "").strip())


class ReplaySession(TutorSession):
    """Gemma's test conversation with the student as the tutor: the pupil's messages in Gemma's
    order, through the harness's own turn logic (TutorSession.receive). History "gold": every turn
    starts from the conversation Gemma had at that point; "own": from the student's own replies."""

    def __init__(self, gemma_turns, history: str):
        self.gemma_turns, self.history = gemma_turns, history
        self.gemma_conversation = Conversation()   # what Gemma read before the next turn
        super().__init__(pupil=None)

    def advance(self) -> None:
        index = len(self.transcript)
        if index == len(self.gemma_turns):
            self.pending = None
            return
        if index:
            store_reply(self.gemma_conversation, self.gemma_turns[index - 1])
        if self.history == "gold":
            self.conversation = copy.deepcopy(self.gemma_conversation)
        gemma_turn = self.gemma_turns[index]
        self.turn = {field: gemma_turn[field] for field in TURN_FIELDS if field in gemma_turn}
        self.turn["attempts"] = 0
        self.pending = Request("route", self.conversation.route_context(gemma_turn["message"]), max_tokens=24)


def effective_route(route: str) -> str:
    return "solution" if route == "unclear" else route   # the harness treats a reply without a tag as a task


def scored_turn(student_turn, gemma_turn, position: int):
    gemma_route = effective_route(gemma_turn["route"])
    gemma_answer = gemma_turn.get("answer") if gemma_route == "solution" else None
    route_agrees = effective_route(student_turn["route"]) == gemma_route
    answer_agrees = None if gemma_answer is None else student_turn.get("answer") == gemma_answer
    return {"position": position, "event": gemma_turn["event"], "gemma_route": gemma_route, "route": student_turn["route"],
            "gemma_answer": gemma_answer, "answer": student_turn.get("answer"), "route_agrees": route_agrees,
            "answer_agrees": answer_agrees, "agrees": route_agrees and answer_agrees is not False,
            "check": student_turn.get("check"), "retried": bool(student_turn.get("rejected")), "output": student_turn["output"]}


def play(model, conversation):
    """Both histories of one test conversation: {history: its scored turns}."""
    import torch
    played = {}
    for history in HISTORIES:
        torch.manual_seed(zlib.crc32(conversation["item_id"].encode()))   # the sampled retries: reproducible, the same in both histories at turn 1
        session = ReplaySession(conversation["turns"], history)
        run_sessions(model, [session])
        played[history] = [scored_turn(student_turn, gemma_turn, position) for position, (student_turn, gemma_turn)
                           in enumerate(zip(session.transcript, conversation["turns"]))]
    model.student.replies.clear()
    return played


def test_conversations():
    """Gemma's held-out test conversations in file order, or a seeded sample of CONVERSATION_COUNT
    of them (the same for every student)."""
    conversations = [conversation for line in open(conversations_path("v1"), encoding="utf-8")
                     for conversation in [json.loads(line)] if conversation["split"] == "test"]
    if CONVERSATION_COUNT is None or CONVERSATION_COUNT >= len(conversations):
        return conversations
    chosen = set(random.Random(SAMPLE_SEED).sample(range(len(conversations)), CONVERSATION_COUNT))
    return [conversation for index, conversation in enumerate(conversations) if index in chosen]


def results_path(run_name: str):
    return run_dir(run_name) / "end_to_end.jsonl"


def saved(run_name: str):
    """item id -> the saved record of each conversation the student has played."""
    path, records = results_path(run_name), {}
    if path.exists():
        for line in open(path, encoding="utf-8"):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:   # a line cut off by a stop
                continue
            records[record["item_id"]] = record
    return records


def play_student(run_name: str, conversations) -> None:
    """Plays every conversation the student has not played yet, appending each when it ends."""
    from library.domain.math_tutor.demo import load_student

    done = saved(run_name)
    todo = [conversation for conversation in conversations if conversation["item_id"] not in done]
    if not todo:
        print(f"{run_name}: all {len(conversations)} conversations played, skipped")
        return
    _, model = load_student(run_name)
    model.student = RememberedReplies(model.student)
    print(f"{run_name}: {len(conversations) - len(todo)} conversations played, {len(todo)} to go", flush=True)
    started = time.time()
    with open(results_path(run_name), "a", encoding="utf-8") as file:
        for count, conversation in enumerate(todo, 1):
            conversation_started = time.time()
            played = play(model, conversation)
            file.write(json.dumps({"item_id": conversation["item_id"], "seconds": round(time.time() - conversation_started, 2),
                                   **played}, ensure_ascii=False) + "\n")
            file.flush()
            if count % 25 == 0 or count == len(todo):
                elapsed = time.time() - started
                print(f"  {count}/{len(todo)} in {elapsed / 60:.1f} min, about {elapsed / count * (len(todo) - count) / 60:.0f} min to go",
                      flush=True)


# --- the report ------------------------------------------------------------------------------

def share(values):
    values = list(values)
    return round(sum(values) / len(values), 4) if values else None


def agreement(turn_lists):
    """Agreement with Gemma over a list of conversations (each a list of scored turns)."""
    turns = [turn for turns in turn_lists for turn in turns]
    answered = [turn for turn in turns if turn["answer_agrees"] is not None]
    return {
        "conversations": len(turn_lists), "turns": len(turns),
        "route_agreement": share(turn["route_agrees"] for turn in turns),
        "answer_turns": len(answered),
        "answer_agreement": share(turn["answer_agrees"] for turn in answered),
        "turn_agreement": share(turn["agrees"] for turn in turns),
        "conversation_agreement": share(all(turn["agrees"] for turn in turns) for turns in turn_lists),
        "conversation_answer_agreement": share(all(turn["answer_agrees"] is not False for turn in turns) for turns in turn_lists),
    }


def student_rows(arm, seed, run_name, records, conversations):
    """One row per history (math_end_to_end.csv) and its rows per turn type and position."""
    rows, group_rows = [], []
    for history in HISTORIES:
        turn_lists = [records[conversation["item_id"]][history] for conversation in conversations]
        turns = [turn for turns in turn_lists for turn in turns]
        answered = [turn for turn in turns if turn["answer_agrees"] is not None]
        solutions = [turn for turn in turns if effective_route(turn["route"]) == "solution"]
        explanations = [turn for turn in turns if turn["route"] == "explanation"]
        base = {"model": run_name, "arm": arm, "seed": seed, "history": history}
        rows.append({"key": f"{run_name}/{history}", **base, "compared_with": "Gemma's reply", **agreement(turn_lists),
                     "first_turn_answer_agreement": share(turn["answer_agrees"] for turn in answered if turn["position"] == 0),
                     "follow_up_answer_agreement": share(turn["answer_agrees"] for turn in answered if turn["position"] > 0),
                     "solution_check_pass": share(turn["check"] is None for turn in solutions),
                     "explanation_check_pass": share(turn["check"] is None for turn in explanations),
                     "retried_share": share(turn["retried"] for turn in turns),
                     "seconds_per_conversation": round(sum(records[c["item_id"]]["seconds"] for c in conversations) / len(conversations), 2)})
        groups = [("event", event, [turn for turn in turns if turn["event"] == event]) for event in EVENTS]
        groups += [("position", name, [turn for turn in turns if first <= turn["position"] + 1 <= last]) for name, first, last in POSITIONS]
        for grouping, group, group_turns in groups:
            measures = agreement([group_turns])
            group_rows.append({"key": f"{run_name}/{history}/{grouping}/{group}", **base, "grouping": grouping, "group": group,
                               **{column: measures[column] for column in GROUP_COLUMNS[7:]}})
    return rows, group_rows


def gemma_rows(conversations):
    """Gemma's own reference in its conversations: the route against the pupil script's expected
    route, the answer against the item's code (solve()); a turn is right when both are."""
    turn_lists = []
    for conversation in conversations:
        scored = []
        for position, turn in enumerate(conversation["turns"]):
            answer_right = None if turn.get("expected_answer") is None else bool(turn.get("answer_correct"))
            scored.append({"position": position, "event": turn["event"], "route_agrees": turn["route_correct"],
                           "answer_agrees": answer_right, "agrees": turn["route_correct"] and answer_right is not False})
        turn_lists.append(scored)
    turns = [turn for turns in turn_lists for turn in turns]
    answered = [turn for turn in turns if turn["answer_agrees"] is not None]
    base = {"model": "gemma-4-12B-it", "arm": "teacher", "seed": "", "history": "own"}
    row = {"key": "gemma/own", **base, "compared_with": "pupil script's route, solve()'s answer", **agreement(turn_lists),
           "first_turn_answer_agreement": share(turn["answer_agrees"] for turn in answered if turn["position"] == 0),
           "follow_up_answer_agreement": share(turn["answer_agrees"] for turn in answered if turn["position"] > 0)}
    groups = [("event", event, [turn for turn in turns if turn["event"] == event]) for event in EVENTS]
    groups += [("position", name, [turn for turn in turns if first <= turn["position"] + 1 <= last]) for name, first, last in POSITIONS]
    group_rows = [{"key": f"gemma/own/{grouping}/{group}", **base, "grouping": grouping, "group": group,
                   **{column: measures[column] for column in GROUP_COLUMNS[7:]}}
                  for grouping, group, group_turns in groups for measures in [agreement([group_turns])]]
    return row, group_rows


def formatted(mean, std):
    if mean is None:
        return "-"
    return f"{mean:.3f}" + (f"±{std:.3f}" if std is not None else "")


def print_report(rows, gemma_row):
    by_key = {row["key"]: row for row in rows}
    print(f"\n{rows[0]['conversations']} test conversations, {rows[0]['turns']} turns; agreement with Gemma's reply\n")
    print(f"{'student':26} {'history':7} " + " ".join(f"{measure[:14]:>14}" for measure in SUMMARY_MEASURES))
    for row in rows:
        print(f"{row['model']:26} {row['history']:7} " + " ".join(f"{row[measure]:>14}" for measure in SUMMARY_MEASURES))
    print(f"{'gemma (vs script, solve())':26} {'own':7} " + " ".join(f"{gemma_row[measure]:>14}" for measure in SUMMARY_MEASURES))

    print("\nper arm, mean±sample std (KL n=3, CE n=3, KL->CE n=2); gap = gold - own")
    for arm, seeds in ARMS.items():
        for history in HISTORIES + ("gap",):
            cells = []
            for measure in SUMMARY_MEASURES:
                values = []
                for seed in seeds:
                    name = f"math-agent-{arm}-m-s{seed}"
                    gold, own = by_key[f"{name}/gold"][measure], by_key[f"{name}/own"][measure]
                    values.append(gold - own if history == "gap" else by_key[f"{name}/{history}"][measure])
                mean, std, _ = mean_std(values)
                cells.append(formatted(mean, std))
            print(f"{arm:9} {history:5} " + " ".join(f"{cell:>14}" for cell in cells))

    print("\nKL - CE per seed (s1 / s2 / s3)")
    for history in HISTORIES + ("gap",):
        for measure in SUMMARY_MEASURES:
            differences = []
            for seed in ARMS["ce"]:
                values = {}
                for arm in ("kl", "ce"):
                    name = f"math-agent-{arm}-m-s{seed}"
                    gold, own = by_key[f"{name}/gold"][measure], by_key[f"{name}/own"][measure]
                    values[arm] = gold - own if history == "gap" else by_key[f"{name}/{history}"][measure]
                differences.append(values["kl"] - values["ce"])
            print(f"  {history:5} {measure:30} " + " / ".join(f"{difference:+.4f}" for difference in differences))


def report(conversations) -> None:
    rows, group_rows = [], []
    for arm, seed, run_name in STUDENTS:
        records = saved(run_name)
        if any(conversation["item_id"] not in records for conversation in conversations):
            print(f"{run_name}: not every conversation played yet, report waits")
            return
        student, groups = student_rows(arm, seed, run_name, records, conversations)
        rows += student
        group_rows += groups
    gemma_row, gemma_groups = gemma_rows(conversations)
    print_report(rows, gemma_row)
    write_rows("math_end_to_end.csv", RESULT_COLUMNS, rows + [gemma_row])
    write_rows("math_end_to_end_by_turn.csv", GROUP_COLUMNS, group_rows + gemma_groups)


def main():
    import torch
    torch.set_num_threads(THREADS)
    conversations = test_conversations()
    names = [name for _, _, name in STUDENTS]
    chosen = [name for name in names if name in sys.argv[1:]] or names
    try:
        for name in chosen:
            play_student(name, conversations)
    except KeyboardInterrupt:
        print("\nStopped: every finished conversation is saved; rerun to resume.")
        return
    report(conversations)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
