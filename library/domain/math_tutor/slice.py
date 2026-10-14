"""The trial slice (decided 2026-10-12): about 1,000 problems run through the whole math-tutor
pipeline before the full data run, to check that a small student learns more than the most
common answer.

Training data (decided 2026-10-13): Gemma's own replies only. The template pupil plays each
problem (its tasks, follow-up tasks and follow-up questions), the harness builds the context and
checks each reply, and Gemma writes every reply (harness.run_conversation_pass). Each
conversation is used up to its first mistake (harness.transcript_records). The teacher's
probabilities are then saved by reading its own replies back (forced mode: the same numbers it
had while writing them).

    python -m library.domain.math_tutor.slice conversations   # GPU: the slice's originals, then their copies
    python -m library.domain.math_tutor.slice conversations originals   # the originals only
    python -m library.domain.math_tutor.agentic               # the output vocabulary (with the replies)
    python -m library.domain.math_tutor.slice corpus          # the slice corpus (SliceDomain)
    python experiments/slice_trial.py                         # GPU: save the teacher data, train, report

The earlier facts-first labels and explanation threads of the slice (commands labels, threads)
stay on disk; the slice corpus no longer uses them. The teacher data and students of the slice
use SliceDomain: its own name, so its own batches folder, which the full run never mistakes for
its own.
"""

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

from .agentic import MathTutorAgenticDomain
from .items import answer_only_test

PACKAGE = Path(__file__).resolve().parent
ITEMS_DIR = PACKAGE / "items"
SLICE_PATH = ITEMS_DIR / "slice_v1.txt"
CONVERSATIONS_PATH = ITEMS_DIR / "conversations" / "slice_v1.jsonl"   # Gemma as the tutor, appended
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


class SliceDomain(MathTutorAgenticDomain):
    """Gemma's replies in the slice's simulated conversations, read back by Gemma (forced mode)."""

    def __init__(self):
        super().__init__("facts", "forced")
        self.name = "math_tutor_slice_v1"

    @property
    def corpus_path(self) -> str:
        return str(PACKAGE / "corpus_slice_v1.jsonl")

    def build_corpus(self, count=None) -> None:
        """Train records (shuffled), then test records (shuffled); held-out shapes are not played."""
        from .harness import transcript_records
        by_id = {item["id"]: item for item in all_items() + copy_items()}
        conversations = [json.loads(line) for line in open(CONVERSATIONS_PATH, encoding="utf-8")]
        records = [record for conversation in conversations
                   for record in transcript_records(by_id[conversation["item_id"]], conversation["turns"])]
        random_generator = random.Random(SEED)
        train = [record for record in records if record["split"] == "train"]
        test = [record for record in records if record["split"] == "test"]
        random_generator.shuffle(train)
        random_generator.shuffle(test)
        with open(self.corpus_path, "w", encoding="utf-8") as file:   # derived file, rebuilt each time
            for record in train[:count] + test:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{len(conversations)} conversations -> {len(train[:count])} train records, then {len(test)} test records "
              f"-> {self.corpus_path}")
        print(dict(Counter((record["split"], record["turn_type"]) for record in records)))


def copy_items() -> List[Dict]:
    """The renumbered copies ("<id>~n2") of the slice's problems."""
    wanted = set(SLICE_PATH.read_text(encoding="utf-8").split())
    copies = [json.loads(line) for line in open(ITEMS_DIR / "renumbered.jsonl", encoding="utf-8")]
    return [item for item in copies if re.sub(r"~n\d+$", "", item["id"]) in wanted]


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "labels"
    if command == "corpus":
        SliceDomain().build_corpus()
        return
    if command == "conversations":
        from library.data_gen.model_handler import ModelHandler
        from .harness import run_conversation_pass
        domain = MathTutorAgenticDomain("facts")
        handler = ModelHandler("google/gemma-4-12B-it", domain, load_in_4bit=True, chat_template=True)
        played = [item for item in slice_items() if item["split"] != "heldout_shape"]   # held-out shapes: student test only
        originals_only = len(sys.argv) > 2 and sys.argv[2] == "originals"
        if run_conversation_pass(handler, domain, played, CONVERSATIONS_PATH) and not originals_only:   # False = stopped
            run_conversation_pass(handler, domain, [item for item in copy_items() if item["split"] != "heldout_shape"],
                                  CONVERSATIONS_PATH)
        return
    items = slice_items()
    print(f"slice: {len(items)} problems {dict(Counter(item['split'] for item in items))}, "
          f"{sum(1 + len(item['follow_ups']) for item in items)} turns", flush=True)
    from library.data_gen.model_handler import ModelHandler
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
        from . import threads
        if len(sys.argv) > 2:
            threads.BATCH = int(sys.argv[2])
        print(f"{threads.run_threads(handler, domain, items=items)} threads written", flush=True)


if __name__ == "__main__":
    main()
