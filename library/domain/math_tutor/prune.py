"""Raw items -> the kept training items, in six stages. Every stage logs how many came
in, how many went out, and why each dropped item was dropped:

  1 execute    run the functions, check every rule (items.validate)
  2 normalise  numbers -> N, names -> NAME, lowercase (no drops; what 3-6 compare)
  3 exact      the same normalised problem as an earlier item
  4 near-dup   word-sequence ratio > 0.8 against an earlier kept item (80% the same
               words in the same order; reordered or reworded problems survive)
  5 test leak  against any of the 1,000 SVAMP problems: word-sequence ratio > 0.8, or
               word-trigram Jaccard > 0.5 (stricter: a shared skeleton is enough)
  6 shape cap  inside a slice, no full shape (follow-ups included) above 25% of the
               slice; the most redundant go first

Then each kept item gets its split: held-out shape, test (10% per slice) or train.

    python -m library.domain.math_tutor.prune

Reads items/raw/*.jsonl and the rewordings in items/reword/*.jsonl (written by the
agents, never modified here) and rebuilds
items/items.jsonl, items/dropped.jsonl and items/build_log.json from them.
"""

import json
import math
import random
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, List

from .items import validate
from .reword import read_rewordings
from .similarity import TrigramIndex, WordSequenceIndex, collect_name_words, normalise, word_trigrams

PACKAGE_DIR = Path(__file__).resolve().parent
ITEMS_DIR = PACKAGE_DIR / "items"
RAW_DIR = ITEMS_DIR / "raw"
BENCHMARKS_DIR = PACKAGE_DIR / "benchmarks"

NEAR_DUPLICATE = 0.8       # word-sequence ratio
TOO_CLOSE_TO_TEST = 0.5    # word-trigram Jaccard, on top of NEAR_DUPLICATE against SVAMP
REDUNDANCY_FLOOR = 0.5     # shape cap: nearest neighbours below this count as 0
SHAPE_CAP_SHARE = 0.25     # of the item's slice, per full shape
SHAPE_CAP_FLOOR = 50       # small slices are not capped below this
TEST_SHARE = 0.10
HELDOUT_SHAPES: List[str] = []  # problem shapes never trained on; chosen after the writing run
SEED = 0


def read_raw() -> List[Dict]:
    items = []
    for path in sorted(RAW_DIR.glob("*.jsonl")):
        for line_number, line in enumerate(open(path, encoding="utf-8"), start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                item = {"id": f"{path.stem}:{line_number}", "_broken_json": True}
            item.setdefault("id", f"{path.stem}:{line_number}")
            items.append(item)
    # Rewordings come last, so an original always wins against its own rewording.
    items += read_rewordings({item["id"]: item for item in items if not item.get("_broken_json")})
    return items


def svamp_problems() -> List[Dict]:
    """All 1,000 SVAMP problems: the 746 in the test plus the ones the range filter dropped."""
    problems = [json.loads(line) for line in open(BENCHMARKS_DIR / "real_test.jsonl", encoding="utf-8")]
    problems += [json.loads(line) for line in open(BENCHMARKS_DIR / "dropped.jsonl", encoding="utf-8")
                 if json.loads(line)["set"] == "svamp"]
    return problems


def source_kind(item: Dict) -> str:
    return str(item.get("source", "unknown")).split(":")[0]


def reason_category(message: str) -> str:
    """'turn 1: 18 is in the code but not in the text' -> 'N is in the code but not in the text'."""
    message = re.sub(r"^turn \d+: ", "", message)
    message = re.sub(r"\[.*?\]", "", message)
    message = re.sub(r"'[^']*'", "'…'", message)
    return re.sub(r"\d+", "N", message).strip()


class Pipeline:
    def __init__(self):
        self.stages: List[Dict] = []
        self.dropped: List[Dict] = []

    def record(self, stage: str, before: List[Dict], after: List[Dict], drops: List[Dict], extra: Dict = None) -> None:
        for drop in drops:
            drop["stage"] = stage
        self.dropped.extend(drops)
        entry = {
            "stage": stage, "in": len(before), "out": len(after),
            "dropped_by_reason": dict(Counter(drop["reason"] for drop in drops).most_common()),
            "out_by_slice": dict(sorted(Counter(item.get("slice", "?") for item in after).items())),
            "out_by_source": dict(Counter(source_kind(item) for item in after).most_common()),
        }
        entry.update(extra or {})
        self.stages.append(entry)


def drop_record(item: Dict, reason: str, detail: str = "") -> Dict:
    return {"id": item.get("id"), "slice": item.get("slice"), "source": item.get("source"),
            "reason": reason, "detail": detail, "problem": item.get("problem")}


def prune() -> None:
    pipeline = Pipeline()
    raw = read_raw()
    random_generator = random.Random(SEED)

    # 1 execute
    executed, drops = [], []
    for item in raw:
        if item.get("_broken_json"):
            drops.append(drop_record(item, "not valid json"))
            continue
        if item.get("_reword_error"):
            drops.append(drop_record(item, "rewording does not fit its original", item["_reword_error"]))
            continue
        validated, message = validate(item)
        if validated:
            executed.append(validated)
        else:
            drops.append(drop_record(item, reason_category(message), message))
    pipeline.record("execute", raw, executed, drops)

    # 2 normalise
    svamp = svamp_problems()
    name_words = collect_name_words([item["problem"] for item in executed] + [problem["problem"] for problem in svamp])
    words_by_id = {item["id"]: normalise(item["problem"], name_words) for item in executed}
    examples = [{"problem": item["problem"], "normalised": " ".join(words_by_id[item["id"]])} for item in executed[:5]]
    pipeline.record("normalise", executed, executed, [], {"examples": examples})

    # 3 exact
    first_by_words, unique, drops = {}, [], []
    for item in executed:
        key = tuple(words_by_id[item["id"]])
        if key in first_by_words:
            drops.append(drop_record(item, "exact duplicate", first_by_words[key]))
        else:
            first_by_words[key] = item["id"]
            unique.append(item)
    pipeline.record("exact", executed, unique, drops)

    # 4 near-duplicate
    sequence_index = WordSequenceIndex(NEAR_DUPLICATE, (words_by_id[item["id"]] for item in unique))
    distinct, drops = [], []
    for item in unique:
        neighbour, score = sequence_index.most_similar(words_by_id[item["id"]])
        if score > NEAR_DUPLICATE:
            drops.append(drop_record(item, "near-duplicate", f"{neighbour} sequence={score:.2f}"))
        else:
            sequence_index.add(item["id"], words_by_id[item["id"]])
            distinct.append(item)
    pipeline.record("near-dup", unique, distinct, drops)

    # 5 test leak
    svamp_words = {problem["id"]: normalise(problem["problem"], name_words) for problem in svamp}
    test_trigrams = TrigramIndex()
    test_sequences = WordSequenceIndex(NEAR_DUPLICATE, list(svamp_words.values()) + list(words_by_id.values()))
    for problem_id, words in svamp_words.items():
        test_trigrams.add(problem_id, word_trigrams(words))
        test_sequences.add(problem_id, words)
    clean, drops = [], []
    for item in distinct:
        words = words_by_id[item["id"]]
        trigram_neighbour, trigram_score = test_trigrams.most_similar(word_trigrams(words))
        sequence_neighbour, sequence_score = test_sequences.most_similar(words)
        if sequence_score > NEAR_DUPLICATE:
            drops.append(drop_record(item, "too close to a SVAMP problem", f"{sequence_neighbour} sequence={sequence_score:.2f}"))
        elif trigram_score > TOO_CLOSE_TO_TEST:
            drops.append(drop_record(item, "too close to a SVAMP problem", f"{trigram_neighbour} trigram J={trigram_score:.2f}"))
        else:
            clean.append(item)
    pipeline.record("test leak", distinct, clean, drops)

    # 6 shape cap, inside each slice on the full shape (follow-ups included): one structure
    # may not crowd out the rest of its slice. Across slices a shared shape is legitimate
    # (S01-S04 are all one-step add/sub), so there is no global cap. Within a crowded
    # cell, the items most like another kept item are dropped first.
    slice_sizes = Counter(item["slice"] for item in clean)
    caps = {name: max(math.ceil(SHAPE_CAP_SHARE * size), SHAPE_CAP_FLOOR) for name, size in slice_sizes.items()}
    redundancy_index = WordSequenceIndex(REDUNDANCY_FLOOR, (words_by_id[item["id"]] for item in clean))
    for item in clean:
        redundancy_index.add(item["id"], words_by_id[item["id"]])
    by_shape = defaultdict(list)
    for item in clean:
        by_shape[(item["slice"], item["shape"])].append(item)
    over_cap_ids, drops = set(), []
    for (slice_name, shape), shape_items in sorted(by_shape.items()):
        cap = caps[slice_name]
        if len(shape_items) <= cap:
            continue
        redundancy = {}
        for item in shape_items:
            neighbour, score = redundancy_index.most_similar(words_by_id[item["id"]], exclude=item["id"])
            redundancy[item["id"]] = (neighbour, score) if score >= REDUNDANCY_FLOOR else (None, 0.0)
        tie_breaks = {item["id"]: random_generator.random() for item in shape_items}
        ranked = sorted(shape_items, key=lambda item: (-redundancy[item["id"]][1], tie_breaks[item["id"]]))
        for item in ranked[:len(shape_items) - cap]:
            over_cap_ids.add(item["id"])
            neighbour, score = redundancy[item["id"]]
            drops.append(drop_record(item, "shape over cap", f"{slice_name} {shape}: nearest {neighbour} sequence={score:.2f}"))
    capped = [item for item in clean if item["id"] not in over_cap_ids]
    pipeline.record("shape cap", clean, capped, drops, {"cap_by_slice": dict(sorted(caps.items()))})

    # splits, by family, so later rewordings and renumberings follow their original
    families_by_slice = defaultdict(list)
    for item in capped:
        if item["problem_shape"] in HELDOUT_SHAPES:
            item["split"] = "heldout_shape"
        else:
            families_by_slice[item["slice"]].append(item["family"])
    test_families = set()
    for slice_name, families in sorted(families_by_slice.items()):
        families = sorted(set(families))
        random_generator.shuffle(families)
        test_families.update(families[:math.ceil(TEST_SHARE * len(families))])
    for item in capped:
        if "split" not in item:
            item["split"] = "test" if item["family"] in test_families else "train"

    ITEMS_DIR.mkdir(exist_ok=True)
    with open(ITEMS_DIR / "items.jsonl", "w", encoding="utf-8") as file:
        for item in capped:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(ITEMS_DIR / "dropped.jsonl", "w", encoding="utf-8") as file:
        for drop in pipeline.dropped:
            file.write(json.dumps(drop, ensure_ascii=False) + "\n")
    log = {
        "built": date.today().isoformat(),
        "rules": {"near_duplicate_sequence_ratio": NEAR_DUPLICATE, "too_close_to_test_trigram_jaccard": TOO_CLOSE_TO_TEST,
                  "redundancy_floor": REDUNDANCY_FLOOR,
                  "shape_cap_share": SHAPE_CAP_SHARE, "shape_cap_floor": SHAPE_CAP_FLOOR,
                  "test_share": TEST_SHARE, "heldout_shapes": HELDOUT_SHAPES, "seed": SEED},
        "raw_files": dict(Counter(f"{path.name}" for path in sorted(RAW_DIR.glob("*.jsonl"))
                                  for line in open(path, encoding="utf-8") if line.strip())),
        "stages": pipeline.stages,
        "kept": len(capped),
        "by_split": dict(Counter(item["split"] for item in capped)),
        "by_slice": dict(sorted(Counter(item["slice"] for item in capped).items())),
        "by_source": dict(Counter(source_kind(item) for item in capped).most_common()),
        "by_problem_shape": dict(Counter(item["problem_shape"] for item in capped).most_common()),
        "by_grade": dict(sorted(Counter(item["grade"] for item in capped).items())),
        "by_distractor": dict(Counter(item["distractor"] for item in capped).most_common()),
        "by_context": dict(Counter(item["context"] for item in capped).most_common()),
        "by_follow_up_count": dict(sorted(Counter(len(item["follow_ups"]) for item in capped).items())),
        "turns": sum(1 + len(item["follow_ups"]) for item in capped),
    }
    with open(ITEMS_DIR / "build_log.json", "w", encoding="utf-8") as file:
        json.dump(log, file, indent=2, ensure_ascii=False)

    for stage in pipeline.stages:
        print(f"{stage['stage']:10} {stage['in']:7} -> {stage['out']:7}   {stage['dropped_by_reason'] or ''}")
    print(f"kept {len(capped)} items, {log['turns']} turns, splits {log['by_split']}")


if __name__ == "__main__":
    prune()
