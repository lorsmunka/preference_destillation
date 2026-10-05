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
from pathlib import Path
from typing import Dict, Iterator, List, Optional

ITEMS_DIR = Path(__file__).resolve().parent / "items"


def turn_text(item: Dict, turn: int) -> str:
    lines = [f"Problem: {item['problem']}"]
    for index in range(turn):
        lines.append(f"Answer: {item['answers'][index]}")
        lines.append(f"Follow-up: {item['follow_ups'][index]['question']}")
    return "\n".join(lines)


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
    from .conversation import solve_turns, usable

    random_generator = random.Random(seed)
    items = kept_items()
    skipped = sum(1 for item in items if not usable(item, annotation))
    records = [record for item in items if usable(item, annotation) for record in solve_turns(item, annotation)]
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
               "items_skipped_until_relabel": skipped}
    print(f"math_tutor agentic corpus ({annotation}): {len(train)} train turns, then {len(test)} test turns -> {path}"
          f" ({skipped} reworded items skipped until relabelled)")
    return summary
