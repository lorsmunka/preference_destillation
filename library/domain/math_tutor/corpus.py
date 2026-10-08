"""Turns for data generation. Every kept item, and every renumbered copy, becomes one
record per turn. A follow-up turn carries the earlier turns with their correct answers —
the state the tutor harness would hold:

    Problem: Mia bakes 4 trays of 6 muffins. How many muffins is that?
    Answer: 24
    Follow-up: She sells 9 of them. How many are left?

Train turns come first and test turns last, so the trainer's positional train/test cut
never puts a test turn in training (the cut is set from the batches after generation).
Held-out-shape items are left out: they are evaluated against their executed answers.
"""

import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from .items import answer_only_test

PACKAGE = Path(__file__).resolve().parent
ITEMS_DIR = PACKAGE / "items"


def turn_text(item: Dict, turn: int) -> str:
    lines = [f"Problem: {item['problem']}"]
    for index in range(turn):
        lines.append(f"Answer: {item['answers'][index]}")
        lines.append(f"Follow-up: {item['follow_ups'][index]['question']}")
    return "\n".join(lines)


ANSWER_ONLY_TEST_PATH = PACKAGE / "corpus_answer_only_test.jsonl"


def write_answer_only_test(items: List[Dict]) -> None:
    """The paraphrase test: test rewordings (and their copies) with their tasks and gold
    answers only; the student is scored on its final answers, with no teacher data."""
    with open(ANSWER_ONLY_TEST_PATH, "w", encoding="utf-8") as file:   # derived file, rebuilt each time
        for item in items:
            record = {"item_id": item["id"], "family": item["family"], "split": item["split"], "slice": item["slice"],
                      "tasks": [item["problem"]] + [follow_up["question"] for follow_up in item["follow_ups"]],
                      "answers": item["answers"], "shape": item["shape"]}
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def kept_items() -> List[Dict]:
    items = []
    for name in ("items.jsonl", "renumbered.jsonl"):
        path = ITEMS_DIR / name
        if path.exists():
            items += [json.loads(line) for line in open(path, encoding="utf-8")]
    return items


def turn_records(items: List[Dict]) -> Iterator[Dict]:
    for item in items:
        for turn in range(1 + len(item["follow_ups"])):
            yield {
                "text": turn_text(item, turn),
                "answer": item["answers"][turn],
                "item_id": item["id"],
                "family": item["family"],
                "turn": turn,
                "slice": item["slice"],
                "shape": item["shape"].split("|")[turn],
                "problem_shape": item["problem_shape"],
                "split": item["split"],
            }


def build_corpus(path: str, count: Optional[int] = None, seed: int = 0) -> None:
    """All train turns (shuffled), then all test turns (shuffled). `count` caps the train turns."""
    random_generator = random.Random(seed)
    records = list(turn_records(kept_items()))
    train = [record for record in records if record["split"] == "train"]
    test = [record for record in records if record["split"] == "test"]
    random_generator.shuffle(train)
    random_generator.shuffle(test)
    if count is not None:
        train = train[:count]
    with open(path, "w", encoding="utf-8") as file:
        for record in train + test:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    held_out = sum(1 for record in records if record["split"] == "heldout_shape")
    print(f"math_tutor corpus: {len(train)} train turns, then {len(test)} test turns -> {path}"
          f" ({held_out} held-out-shape turns kept out)")


def build_agentic_corpus(path: str, annotation: str = "named", count: Optional[int] = None, seed: int = 0) -> Dict:
    """The tagged-conversation corpus (conversation.py): one solve record per turn, with the
    gold solution block in `annotation` format as its target. Train turns first (shuffled),
    test turns last; held-out shapes left out. Rewordings are skipped for annotations whose
    labels come from variable names (named, code) until a relabel pass. Explain and route
    records join once the explanation threads exist (plan step 5)."""
    from .conversation import labelled_turns, solve_turns, thread_turns, usable
    from .threads import THREADS_PATH

    random_generator = random.Random(seed)
    all_items = kept_items()
    answer_only = [item for item in all_items if answer_only_test(item)]
    write_answer_only_test(answer_only)
    items = [item for item in all_items if not answer_only_test(item)]
    skipped = sum(1 for item in items if not usable(item, annotation))
    partly_labelled = sum(1 for item in items if 0 < labelled_turns(item, annotation) < 1 + len(item["follow_ups"]))
    records = [record for item in items if usable(item, annotation) for record in solve_turns(item, annotation)]
    if THREADS_PATH.exists():   # explanation threads (threads.py) and the route records after them
        by_id = {item["id"]: item for item in items}
        for line in open(THREADS_PATH, encoding="utf-8"):
            thread_record = json.loads(line)
            item = by_id.get(thread_record["item_id"])
            if item is not None and usable(item, annotation):
                records += list(thread_turns(item, annotation, thread_record))
    train = [record for record in records if record["split"] == "train"]
    test = [record for record in records if record["split"] == "test"]
    random_generator.shuffle(train)
    random_generator.shuffle(test)
    if count is not None:
        train = train[:count]
    with open(path, "w", encoding="utf-8") as file:
        for record in train + test:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {"annotation": annotation, "train_turns": len(train), "test_turns": len(test),
               "items_skipped_until_relabel": skipped,
               "items_cut_at_first_unlabelled_turn": partly_labelled,
               "answer_only_test_items": len(answer_only),
               "by_turn_type": dict(Counter(record.get("turn_type", "solve") for record in train + test))}
    print(f"math_tutor agentic corpus ({annotation}): {len(train)} train turns, then {len(test)} test turns -> {path}"
          f" ({skipped} reworded items skipped until relabelled)")
    return summary
