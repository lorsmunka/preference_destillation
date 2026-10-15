"""The trial slice (decided 2026-10-12): about 1,000 problems run through the whole math-tutor
pipeline before the full data run, to check that a small student learns more than the most
common answer.

Training data (decided 2026-10-13): Gemma's own replies only. The template pupil plays each
problem (its tasks, follow-up tasks and follow-up questions), the harness builds the context and
checks each reply, and Gemma writes every reply (harness.run_conversation_pass). Each
conversation is used up to its first mistake (harness.transcript_records). The teacher's
probabilities are then saved by reading its own replies back (forced mode: the same numbers it
had while writing them).

    python -m library.domain.math_tutor.slice conversations   # GPU: the slice's originals, then the extension
    python -m library.domain.math_tutor.slice conversations originals   # the originals only
    python -m library.domain.math_tutor.agentic               # the output vocabulary (with the replies)
    python -m library.domain.math_tutor.slice corpus          # the slice corpus (SliceDomain)
    python experiments/slice_trial.py                         # GPU: save the teacher data, train, report

Groups (decided 2026-10-15): every record carries a "group" flag. "original" = the slice's first
950 problems as played (rewordings among them included); the extension adds, for the copies-vs-new
test, "copy" = one renumbered copy of each train original (same text, new numbers) and "new" = as
many new train problems from families the slice does not use. Training can select by group
(TrainingRun.training_groups).

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
from typing import Dict, List, Tuple

from .agentic import MathTutorAgenticDomain
from .items import answer_only_test

PACKAGE = Path(__file__).resolve().parent
ITEMS_DIR = PACKAGE / "items"
SLICE_PATH = ITEMS_DIR / "slice_v1.txt"
EXTENSION_PATH = ITEMS_DIR / "slice_v1_extension.txt"   # "<item id> <group>" per line (the copies-vs-new test)
GROUPS = ("original", "new", "copy")   # corpus order of the train records
CONVERSATIONS_PATH = ITEMS_DIR / "conversations" / "slice_v1.jsonl"   # Gemma as the tutor, appended
COUNTS = {"train": 850, "test": 100, "heldout_shape": 50}   # problems per split
SEED = 0
CURVE_SIZES = (100, 200, 500)   # learning curve (2026-10-14): students on the first 100, 200, 500 and all train conversations


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


def train_stages(conversations: List[Dict]) -> Dict[str, int]:
    """item id -> learning-curve stage of each train conversation, in the order Gemma played them
    (the file order): 0 for the first 100, 1 for the next 100, 2 up to 500, 3 for the rest."""
    played = [conversation["item_id"] for conversation in conversations
              if conversation["split"] == "train" and conversation.get("group", "original") == "original"]
    return {item_id: sum(position >= size for size in CURVE_SIZES) for position, item_id in enumerate(played)}


def slice_items() -> List[Dict]:
    if not SLICE_PATH.exists():
        choose_slice()
    wanted = set(SLICE_PATH.read_text(encoding="utf-8").split())
    return [item for item in all_items() if item["id"] in wanted]


def corpus_records() -> Tuple[List[Dict], List[Dict]]:
    """The slice corpus as (train, test) records of the saved conversations, each with its group
    (a conversation saved without one is "original"). Train records are ordered by group
    (GROUPS), the originals by learning-curve stage (train_stages), and shuffled within each, so
    the first 100 train conversations' records come first; test records are shuffled."""
    from .harness import transcript_records
    by_id = {item["id"]: item for item in all_items() + copy_items()}
    conversations = [json.loads(line) for line in open(CONVERSATIONS_PATH, encoding="utf-8")]
    records = [{**record, "group": conversation.get("group", "original")} for conversation in conversations
               for record in transcript_records(by_id[conversation["item_id"]], conversation["turns"])]
    stages = train_stages(conversations)
    random_generator = random.Random(SEED)
    train = [record for record in records if record["split"] == "train"]
    test = [record for record in records if record["split"] == "test"]
    random_generator.shuffle(train)
    train.sort(key=lambda record: (GROUPS.index(record["group"]), stages.get(record["item_id"], 0)))   # stable
    random_generator.shuffle(test)
    return train, test


class SliceDomain(MathTutorAgenticDomain):
    """Gemma's replies in the slice's simulated conversations, read back by Gemma (forced mode)."""

    def __init__(self):
        super().__init__("facts", "forced")
        self.name = "math_tutor_slice_v1"

    @property
    def corpus_path(self) -> str:
        return str(PACKAGE / "corpus_slice_v1.jsonl")

    def build_corpus(self, count=None) -> None:
        """Train records, then test records (corpus_records); held-out shapes are not played."""
        train, test = corpus_records()
        with open(self.corpus_path, "w", encoding="utf-8") as file:   # derived file, rebuilt each time
            for record in train[:count] + test:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{len({record['item_id'] for record in train + test})} conversations -> {len(train[:count])} train records, "
              f"then {len(test)} test records -> {self.corpus_path}")
        print(dict(Counter((record["split"], record["turn_type"]) for record in train + test)))
        print(dict(Counter((record["split"], record["group"]) for record in train + test)))


def copy_items() -> List[Dict]:
    """The renumbered copies ("<id>~n2") of the slice's problems."""
    wanted = set(SLICE_PATH.read_text(encoding="utf-8").split())
    copies = [json.loads(line) for line in open(ITEMS_DIR / "renumbered.jsonl", encoding="utf-8")]
    return [item for item in copies if re.sub(r"~n\d+$", "", item["id"]) in wanted]


def copy_number(item_id: str) -> int:
    return int(re.search(r"~n(\d+)$", item_id).group(1))


def choose_extension() -> List[Tuple[str, str]]:
    """The copies-vs-new test (decided 2026-10-15), chosen once and written to EXTENSION_PATH:
    the first renumbered copy of each train original ("copy"), and as many new train problems
    ("new"), one per family, from families the slice does not use (seeded)."""
    originals = slice_items()
    train_ids = {item["id"] for item in originals if item["split"] == "train"}
    copies_by_original = {}
    for item in copy_items():
        original_id = re.sub(r"~n\d+$", "", item["id"])
        if original_id in train_ids:
            copies_by_original.setdefault(original_id, []).append(item["id"])
    copies = sorted(min(ids, key=copy_number) for ids in copies_by_original.values())
    used_families = {item["family"] for item in originals}
    by_family = {}
    for item in all_items():
        if item["split"] == "train" and not answer_only_test(item) and item["family"] not in used_families:
            by_family.setdefault(item["family"], []).append(item["id"])
    random_generator = random.Random(SEED)
    families = random_generator.sample(sorted(by_family), len(copies))
    new = sorted(random_generator.choice(sorted(by_family[family])) for family in families)
    chosen = [(item_id, "new") for item_id in new] + [(item_id, "copy") for item_id in copies]
    EXTENSION_PATH.write_text("".join(f"{item_id} {group}\n" for item_id, group in chosen), encoding="utf-8")
    return chosen


def extension_items() -> List[Dict]:
    """The extension's problems, each with its "group" ("new" or "copy")."""
    if not EXTENSION_PATH.exists():
        choose_extension()
    group_of = dict(line.split() for line in EXTENSION_PATH.read_text(encoding="utf-8").splitlines() if line)
    by_id = {item["id"]: item for item in all_items() + copy_items()}
    return [{**by_id[item_id], "group": group} for item_id, group in group_of.items()]


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
        if not run_conversation_pass(handler, domain, played, CONVERSATIONS_PATH):   # False = stopped
            return
        if len(sys.argv) > 2 and sys.argv[2] == "originals":
            return
        run_conversation_pass(handler, domain, extension_items(), CONVERSATIONS_PATH)   # new and copies, mixed
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
