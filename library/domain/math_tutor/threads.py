"""Explanation threads for the training corpus: the pupil's follow-up messages after a solution,
and the tutor's replies. The messages come from the template bank (questions.py), filled in from
the turn's verified facts-first solution; Gemma writes only the explanations. Follow-up tasks are
the items' own follow-ups.

One thread per item, on one of its turns (seeded), 1-3 messages deep; each message after the
first relates to the one before (the same step, a number it used, the step that used its result)
or is a general follow-up. Per message route:

  explanation   Gemma explains (greedy, the agentic teacher prompt); the reply must route to an
                explanation and pass checks.check_explanation, or the thread ends there
  out_of_scope  off-topic: only the route is learned (target <out_of_scope>); the context is left
                unchanged and the thread goes on
  solution      a what-if (only on the item's last turn): the target is the facts-first label
                carried over to the changed number, verified against the re-run code; it ends
                the thread

Gemma's reply is generated for every message (its route and, for a what-if, its answer are
recorded), so the pass also measures the teacher on each kind of message.

    python -m library.domain.math_tutor.threads     # GPU; resumable (done items skipped)

Writes items/threads/threads.jsonl (appended).
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from library.agent.question_templates import related
from .checks import check_explanation, final_answer
from .conversation import Conversation, parse_route
from .labels import read_labels
from .questions import generator as question_generator, kind_weights, solution_foci, tasks_of, what_if_foci

ITEMS_DIR = Path(__file__).resolve().parent / "items"
THREADS_PATH = ITEMS_DIR / "threads" / "threads.jsonl"
SEED = 0
DEPTHS = [1] * 50 + [2] * 35 + [3] * 15       # share of threads by number of messages
BATCH = 16


def solution_body(block: str) -> str:
    return block.replace("<solution>", "").replace("</solution>", "").strip()


def plan_messages(item: Dict, turn: int, block: str, depth: int, generator, random_generator: random.Random) -> List[Dict]:
    """The thread's messages, drawn before any reply exists (the templates only need the solution).
    Each is drawn at the depth of the explanations before it: an off-topic message adds none to
    the context, so the message after it cannot follow up an explanation that is not there."""
    foci = solution_foci(item, turn, block, random_generator)
    if turn == len(item["follow_ups"]):
        foci += [focus for focus in what_if_foci(item, turn, block, random_generator) if focus.data.get("gold")]
    messages, previous, explanations = [], None, 0
    for _ in range(depth):
        candidates = related(foci, previous) if explanations > 0 else foci
        message = generator.draw(candidates, explanations, random_generator, kind_weights=kind_weights(explanations))
        if message is None:
            break
        entry = {"question": message.text, "kind": message.kind, "route": message.route,
                 "template": message.template.text, "focus": message.focus.type}
        if message.kind == "what_if":
            entry.update(target=message.focus.data["gold"], answer=message.focus.data["answer"])
        messages.append(entry)
        if message.kind == "what_if":
            break   # a new solution closes the thread
        if message.kind != "off_topic":
            previous = message.focus
            explanations += 1
    return messages


def plan_threads(items: List[Dict], labels: Dict[tuple, str], generator) -> List[Dict]:
    """Which turn of which item gets a thread, and its messages (seeded)."""
    random_generator = random.Random(SEED)
    plans = []
    for item in items:
        turns = [turn for turn in range(1 + len(item["follow_ups"]))
                 if all((item["id"], earlier) in labels for earlier in range(turn + 1))]
        if not turns:
            continue
        turn = random_generator.choice(turns)
        depth = random_generator.choice(DEPTHS)
        messages = plan_messages(item, turn, labels[(item["id"], turn)], depth, generator, random_generator)
        if messages:
            plans.append({"item": item, "turn": turn, "messages": messages})
    return plans


def main_line(item: Dict, turn: int, labels: Dict[tuple, str]) -> Conversation:
    """The conversation up to and including this turn's facts-first solution."""
    tasks = tasks_of(item)
    conversation = Conversation()
    for earlier in range(turn + 1):
        conversation.add_solution(tasks[earlier], solution_body(labels[(item["id"], earlier)]))
    return conversation


def reply_entry(message: Dict, reply: str) -> Dict:
    """The message with Gemma's reply, its route and the check (an explanation cut off at the
    token limit, without </explanation>, fails it)."""
    routed = parse_route(reply)
    entry = dict(message, routed=routed, reply=reply)
    if message["route"] == "explanation":
        explanation = reply[reply.find("<explanation>") + len("<explanation>"):] if routed == "explanation" else ""
        entry["explanation"] = explanation.split("</explanation>")[0].strip()
        if routed != "explanation":
            entry["check"] = f"routed to {routed}"
        elif "</explanation>" not in explanation:
            entry["check"] = "cut off: no </explanation>"
        else:
            entry["check"] = check_explanation(entry["explanation"], message["kind"])
    else:
        entry["check"] = None if routed == message["route"] else f"routed to {routed}"
        if message["route"] == "solution":
            entry["teacher_answer"] = final_answer(reply)
    return entry


def run_threads(handler, domain, items: Optional[List[Dict]] = None, generator=None,
                labels: Optional[Dict[tuple, str]] = None, threads_path: Path = THREADS_PATH) -> int:
    """Run every planned thread message by message (all threads' first messages in batches, then
    their second...). A thread is appended as soon as it stops or has no message left, so an
    interrupted pass keeps every finished thread and the next run skips those items."""
    if items is None:
        items = [json.loads(line) for line in open(ITEMS_DIR / "items.jsonl", encoding="utf-8")]
    generator = generator or question_generator()
    labels = read_labels() if labels is None else labels
    done = set()
    if threads_path.exists():
        done = {json.loads(line)["item_id"] for line in open(threads_path, encoding="utf-8")}
    plans = [plan for plan in plan_threads(items, labels, generator) if plan["item"]["id"] not in done]
    for plan in plans:
        plan["conversation"] = main_line(plan["item"], plan["turn"], labels)
        plan["thread"] = []
    threads_path.parent.mkdir(parents=True, exist_ok=True)
    with open(threads_path, "a", encoding="utf-8") as out:
        for position in range(max((len(plan["messages"]) for plan in plans), default=0)):
            active = [plan for plan in plans if len(plan["messages"]) > position and not plan.get("stopped")]
            for start in range(0, len(active), BATCH):
                batch = active[start:start + BATCH]
                messages = [plan["messages"][position] for plan in batch]
                prompts = [domain.teacher_prompt(plan["conversation"].route_context(message["question"]))
                           for plan, message in zip(batch, messages)]
                replies = handler.generate_texts(prompts, max_new_tokens=domain.max_steps, stop_strings=domain.stop_strings)
                for plan, message, reply in zip(batch, messages, replies):
                    entry = reply_entry(message, reply)
                    plan["thread"].append(entry)
                    if message["route"] == "explanation":
                        if entry["check"]:   # kept for analysis, but no turn builds on it
                            plan["stopped"] = True
                        else:
                            plan["conversation"].add_explanation(message["question"], entry["explanation"])
                    elif message["route"] == "solution":
                        plan["stopped"] = True
                    if plan.get("stopped") or len(plan["messages"]) == position + 1:
                        out.write(json.dumps({"item_id": plan["item"]["id"], "turn": plan["turn"], "thread": plan["thread"]},
                                             ensure_ascii=False) + "\n")
                        out.flush()
    return len(plans)


def main() -> None:
    from library.data_gen.model_handler import ModelHandler
    from .agentic import MathTutorAgenticDomain
    domain = MathTutorAgenticDomain("facts")
    handler = ModelHandler("google/gemma-4-12B-it", domain, load_in_4bit=True, chat_template=True)
    print(f"{run_threads(handler, domain)} threads written to {THREADS_PATH}")


if __name__ == "__main__":
    main()
