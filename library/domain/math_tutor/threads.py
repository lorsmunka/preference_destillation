"""Explanation threads: follow-up questions (the pupil asks about a step) and the tutor's
explanations; follow-up tasks are the items' own follow-ups. A helper pass on the
GPU writes them as text; the data generation then turns every explanation into a training
turn with the teacher's logits (greedy, so it reproduces the same text).

One thread per kept item, on one of its turns (seeded), 1-3 questions deep:
  pupil   Gemma playing a curious pupil, sampled, one question of a given kind about the
          tutor's last solution (depth 1) or last explanation (depth 2-3)
  tutor   the agentic domain's own teacher prompt on the explain context, greedy

    python -m library.domain.math_tutor.threads     # GPU; resumable (done items skipped)

Writes items/threads/threads.jsonl (appended).
"""

import json
import random
from pathlib import Path
from typing import Dict, List

from .conversation import Conversation
from .labels import read_labels

ITEMS_DIR = Path(__file__).resolve().parent / "items"
THREADS_PATH = ITEMS_DIR / "threads" / "threads.jsonl"
SEED = 0
DEPTHS = [1] * 50 + [2] * 35 + [3] * 15       # share of threads by number of questions
BATCH = 16

QUESTION_KINDS = {
    "why": "why the tutor did one particular step (for example: why multiply here, why take that away)",
    "meaning": "what one particular number in the working stands for",
    "another way": "whether the step could be done another way",
    "lost": "that they do not understand one particular step, asking for it to be explained more simply",
    "check": "whether one particular line is really right, or how the tutor knows",
    "where from": "where one particular number in the working came from",
    "shortcut": "whether there is a quicker way to get the answer",
    "smaller numbers": "to explain one step again with smaller, easier numbers",
}

# Who is asking: varies the voice of the follow-up questions, not their content.
PERSONAS = {
    "curious": "curious and chatty; you start with what you noticed",
    "shy": "shy; you ask in just a few words",
    "confused": "a bit confused; you say which part lost you",
    "confident": "confident; you say what you think first, then ask",
    "impatient": "impatient; you want a quick, short answer",
    "careful": "careful; you ask about one small detail",
}

# How the question starts: drawn per question, so the same kind is not always asked the same way.
OPENINGS = ["Why", "How come", "Wait,", "I don't get", "So", "Is it", "What", "Hmm,", "Could you",
            "But", "Where", "Does", "I thought", "Hang on,", "How"]

PUPIL_PROMPT = """You are a primary-school pupil working with your maths tutor. You are {persona}. Here is your conversation so far:

{conversation}

Ask your tutor ONE question about {about}: {kind}.
Say which step you mean by what it is about in the story (the buses, the stickers, the money left...), not only by its numbers.
Ask it the way a child like you would, in your own words. Start your question with "{opening}" if that fits, otherwise start your own way.
Write only the question."""


def solution_body(block: str) -> str:
    return block.replace("<solution>", "").replace("</solution>", "").strip()


def plan_threads(items: List[Dict], labels: Dict[tuple, str]) -> List[Dict]:
    """Which turn of which item gets a thread, how deep, and which kinds of question (seeded)."""
    random_generator = random.Random(SEED)
    plans = []
    for item in items:
        turns = [turn for turn in range(1 + len(item["follow_ups"])) if (item["id"], turn) in labels]
        if not turns:
            continue
        turn = random_generator.choice(turns)
        depth = random_generator.choice(DEPTHS)
        kinds = [random_generator.choice(list(QUESTION_KINDS)) for _ in range(depth)]
        plans.append({"item": item, "turn": turn, "kinds": kinds})
    return plans


def main_line(item: Dict, turn: int, labels: Dict[tuple, str]) -> Conversation:
    """The conversation up to and including this turn's facts-first solution."""
    tasks = [item["problem"]] + [follow_up["question"] for follow_up in item["follow_ups"]]
    conversation = Conversation()
    for earlier in range(turn + 1):
        conversation.add_solution(tasks[earlier], solution_body(labels[(item["id"], earlier)]))
    return conversation


def run_threads(handler, domain, items: List[Dict] = None) -> int:
    """Build every planned thread depth by depth, appending finished threads."""
    if items is None:
        items = [json.loads(line) for line in open(ITEMS_DIR / "items.jsonl", encoding="utf-8")]
    labels = read_labels()
    done = set()
    if THREADS_PATH.exists():
        done = {json.loads(line)["item_id"] for line in open(THREADS_PATH, encoding="utf-8")}
    plans = [plan for plan in plan_threads(items, labels) if plan["item"]["id"] not in done]
    for plan in plans:
        plan["conversation"] = main_line(plan["item"], plan["turn"], labels)
        plan["thread"] = []
    THREADS_PATH.parent.mkdir(parents=True, exist_ok=True)
    for depth in range(max(len(plan["kinds"]) for plan in plans) if plans else 0):
        active = [plan for plan in plans if len(plan["kinds"]) > depth]
        for start in range(0, len(active), BATCH):
            batch = active[start:start + BATCH]
            pupil_prompts = [PUPIL_PROMPT.format(
                persona=PERSONAS[random.Random(f"{plan['item']['id']}-{depth}").choice(list(PERSONAS))],
                opening=random.Random(f"{plan['item']['id']}-{depth}-opening").choice(OPENINGS),
                conversation=plan["conversation"].history(with_thread=True),
                about="the tutor's last solution" if depth == 0 else "the tutor's last explanation",
                kind=QUESTION_KINDS[plan["kinds"][depth]]) for plan in batch]
            questions = handler.generate_texts(pupil_prompts, max_new_tokens=60, temperature=0.9, seed=SEED + depth + start)
            questions = [question.strip().strip('"').split("\n")[0] for question in questions]
            tutor_prompts = [domain.teacher_prompt(plan["conversation"].explain_context(question))
                             for plan, question in zip(batch, questions)]
            answers = handler.generate_texts(tutor_prompts, max_new_tokens=domain.max_steps,
                                             stop_strings=["</explanation>", "</solution>"])
            for plan, question, answer in zip(batch, questions, answers):
                explanation = answer.replace("<explanation>", "").split("</explanation>")[0].strip()
                plan["thread"].append({"question": question, "kind": plan["kinds"][depth], "explanation": explanation,
                                       "routed": "explanation" if answer.startswith("<explanation>") else
                                       ("solution" if answer.startswith("<solution>") else "neither")})
                plan["conversation"].add_explanation(question, explanation)
    with open(THREADS_PATH, "a", encoding="utf-8") as out:
        for plan in plans:
            out.write(json.dumps({"item_id": plan["item"]["id"], "turn": plan["turn"], "thread": plan["thread"]},
                                 ensure_ascii=False) + "\n")
    return len(plans)


def main() -> None:
    from library.data_gen.model_handler import ModelHandler
    from .agentic import MathTutorAgenticDomain
    domain = MathTutorAgenticDomain("facts")
    handler = ModelHandler("google/gemma-4-12B-it", domain, load_in_4bit=True, chat_template=True)
    print(f"{run_threads(handler, domain)} threads written to {THREADS_PATH}")


if __name__ == "__main__":
    main()
