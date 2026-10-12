"""The trial slice (decided 2026-10-12): about 1,000 problems run through the whole math-tutor
pipeline (facts-first labels, explanation threads, teacher scoring, one student) before the full data
run, to check that a small student learns more than the most common answer.

The labels and threads it writes go to the real files (items/labels/facts.jsonl, items/threads/), so
the full run reuses them. Renumbered copies of the slice's items get their labels by transfer.

    python -m library.domain.math_tutor.slice labels    # GPU: facts-first labels for the slice
    python -m library.domain.math_tutor.slice threads   # GPU: explanation threads for the slice
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

from .items import answer_only_test

ITEMS_DIR = Path(__file__).resolve().parent / "items"
SLICE_PATH = ITEMS_DIR / "slice_v1.txt"
COUNTS = {"train": 850, "test": 100, "heldout_shape": 50}   # problems per split
SEED = 0


def all_items() -> List[Dict]:
    return [json.loads(line) for line in open(ITEMS_DIR / "items.jsonl", encoding="utf-8")]


def choose_slice() -> List[str]:
    """A seeded random sample per split, from items that get full teacher data."""
    random_generator = random.Random(SEED)
    chosen = []
    for split, count in COUNTS.items():
        candidates = sorted(item["id"] for item in all_items() if item["split"] == split and not answer_only_test(item))
        chosen += random_generator.sample(candidates, min(count, len(candidates)))
    SLICE_PATH.write_text("\n".join(sorted(chosen)) + "\n", encoding="utf-8")
    return chosen


def slice_items() -> List[Dict]:
    if not SLICE_PATH.exists():
        choose_slice()
    wanted = set(SLICE_PATH.read_text(encoding="utf-8").split())
    return [item for item in all_items() if item["id"] in wanted]


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "labels"
    items = slice_items()
    print(f"slice: {len(items)} problems {dict(Counter(item['split'] for item in items))}, "
          f"{sum(1 + len(item['follow_ups']) for item in items)} turns", flush=True)
    from library.data_gen.model_handler import ModelHandler
    from .agentic import MathTutorAgenticDomain
    domain = MathTutorAgenticDomain("facts")
    handler = ModelHandler("google/gemma-4-12B-it", domain, load_in_4bit=True, chat_template=True)
    if command == "labels":
        from . import labels
        if len(sys.argv) > 2:
            labels.BATCH = int(sys.argv[2])
        verified = labels.read_labels()
        log = {"blocked_by_earlier_turn": 0, "verified_by_attempt": {}, "failed": {}}
        for turn in range(4):
            labels.label_wave(handler, items, turn, verified, log)
            done = sum(1 for item in items for t in range(1 + len(item["follow_ups"])) if (item["id"], t) in verified)
            print(f"wave {turn} done: {done} slice turns verified so far", flush=True)
        log["copies"] = dict(labels.transfer_to_copies(labels.read_labels()))
        print(json.dumps(log, indent=2), flush=True)
    elif command == "threads":
        from .threads import run_threads
        print(f"{run_threads(handler, domain, items=items)} threads written", flush=True)


if __name__ == "__main__":
    main()
