"""The coding agent's corpus: the training records of every played item (harness.transcript_records),
train records shuffled first, then test records (the test set is the batch directory's tail)."""

import json
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

from .harness import CONVERSATIONS_PATH, transcript_records, version_cache
from .requests import read_items

SEED = 0


def corpus_records() -> Tuple[List[Dict], List[Dict]]:
    items = {item["id"]: item for item in read_items()}
    versions = version_cache()
    records = []
    for line in open(CONVERSATIONS_PATH, encoding="utf-8"):
        played = json.loads(line)
        item = items[played["item_id"]]
        records += transcript_records(item, versions(item["project"], item["version"]), played["turn"])
    random_generator = random.Random(SEED)
    train = [record for record in records if record["split"] == "train"]
    test = [record for record in records if record["split"] == "test"]
    random_generator.shuffle(train)
    random_generator.shuffle(test)
    return train, test


def build_corpus(path: str, count: Optional[int] = None) -> None:
    train, test = corpus_records()
    with open(path, "w", encoding="utf-8") as file:   # derived file, rebuilt each time
        for record in train[:count] + test:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"{len({record['item_id'] for record in train + test})} items -> {len(train[:count])} train records, "
          f"then {len(test)} test records -> {path}")
    print(dict(Counter((record["split"], record["turn_type"]) for record in train + test)))
