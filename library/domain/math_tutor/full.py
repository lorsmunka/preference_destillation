"""The full math-tutor data run (decided 2026-10-20: both agents end to end, then the large runs on
rented GPU time). The same pass as the trial slice (slice.py) on every item: Gemma is the tutor in
each problem's simulated conversation (harness.run_conversation_pass), and each of its replies,
in the context it was written in, is a training record (harness.transcript_records). New problems
only, no renumbered copies (the slice's copies-vs-new test, findings X-C1d). Held-out shapes and
the answer-only paraphrase test are not played (student test only, as in the slice).

One Gemma setup for all of it (the cloud card), so its own conversations file and its own domain
name (so its own batches folder): the slice's 4-bit local replies never mix with these.

    python -m library.domain.math_tutor.full conversations        # GPU: every item not played yet
    python -m library.domain.math_tutor.full conversations 16     # 16 replies per model call (a bigger card)
    python -m library.domain.math_tutor.agentic                   # the output vocabulary, with the replies
    python -m library.domain.math_tutor.full corpus               # the corpus (FullDomain)
"""

import json
import random
import sys
from collections import Counter
from typing import Dict, List, Tuple

from .agentic import MathTutorAgenticDomain
from .items import answer_only_test
from .slice import ITEMS_DIR, PACKAGE, all_items

CONVERSATIONS_PATH = ITEMS_DIR / "conversations" / "full_v1.jsonl"   # Gemma as the tutor, appended
SEED = 0


def full_items() -> List[Dict]:
    """Every train and test item that gets teacher data."""
    return [item for item in all_items() if item["split"] in ("train", "test") and not answer_only_test(item)]


def corpus_records() -> Tuple[List[Dict], List[Dict]]:
    """(train, test) records of the saved conversations, each shuffled (seeded)."""
    from .harness import transcript_records
    by_id = {item["id"]: item for item in all_items()}
    records = [record for line in open(CONVERSATIONS_PATH, encoding="utf-8")
               for conversation in [json.loads(line)]
               for record in transcript_records(by_id[conversation["item_id"]], conversation["turns"])]
    random_generator = random.Random(SEED)
    train = [record for record in records if record["split"] == "train"]
    test = [record for record in records if record["split"] == "test"]
    random_generator.shuffle(train)
    random_generator.shuffle(test)
    return train, test


class FullDomain(MathTutorAgenticDomain):
    """Gemma's replies in every item's simulated conversation, read back by Gemma (forced mode)."""

    def __init__(self):
        super().__init__("facts", "forced")
        self.name = "math_tutor_full_v1"

    @property
    def corpus_path(self) -> str:
        return str(PACKAGE / "corpus_full_v1.jsonl")

    def build_corpus(self, count=None) -> None:
        """Train records (the first `count`), then test records."""
        train, test = corpus_records()
        with open(self.corpus_path, "w", encoding="utf-8") as file:   # derived file, rebuilt each time
            for record in train[:count] + test:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{len({record['item_id'] for record in train + test})} conversations -> {len(train[:count])} train records, "
              f"then {len(test)} test records -> {self.corpus_path}")
        print(dict(Counter((record["split"], record["turn_type"]) for record in train + test)))


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "conversations"
    if command == "corpus":
        FullDomain().build_corpus()
        return
    from library.agent.runner import teacher_setup
    from library.data_gen.model_handler import ModelHandler
    from .harness import run_conversation_pass
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    domain = MathTutorAgenticDomain("facts")
    handler = ModelHandler("google/gemma-4-12B-it", domain, load_in_4bit=teacher_setup(CONVERSATIONS_PATH), chat_template=True)
    items = full_items()
    print(f"{len(items)} items ({dict(Counter(item['split'] for item in items))}), "
          f"{sum(1 + len(item['follow_ups']) for item in items)} tasks", flush=True)
    run_conversation_pass(handler, domain, items, CONVERSATIONS_PATH, batch=batch)


if __name__ == "__main__":
    main()
