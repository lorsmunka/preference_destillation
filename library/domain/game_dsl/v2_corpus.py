"""v2's corpus (2026-11-09, PLAN.md "v2's corpus"), after Gemma's writing pass on the rented cards
(runs/game_v2/full/written_*.jsonl):

  check       every reply: the repairs (";", the naming rule), the block into its game where the app would put it,
              the app's compiler and checks (kept = adds no problem), the ask's rule checks (follows)
              -> runs/game_v2/full/checked.jsonl, blocks.jsonl (kept and follows: the targets)
  vocabulary  v1's vocabulary texts + the v2 targets -> output_tokens_v2.json (v1's file stays)
  corpus      one record per target: text = the input the app builds, target = Gemma's block in a fence
              -> corpus_v2.jsonl; then the teacher data (Gemma, teacher-forced, card2) on a rented card

    python -m library.domain.game_dsl.v2_corpus check | vocabulary | corpus
"""

import json
import random
import sys
from collections import Counter
from typing import Dict, List

from . import CORPUS_V2, FENCE, OUTPUT_TOKENS_V2
from .asks import (FULL_FOLDER, HOLLOW, UNITS, append_jsonl, block_of, failed_checks, load_games, read_jsonl,
                   repaired, spliced)

CHECKED = FULL_FOLDER / "checked.jsonl"
BLOCKS = FULL_FOLDER / "blocks.jsonl"
CHUNK = 1000        # spliced games per compiler run
SEED = 0


def replies() -> Dict[str, Dict]:
    found = {}
    for path in sorted(FULL_FOLDER.glob("written_*.jsonl")):
        for row in read_jsonl(path):
            if row.get("reply") and row["id"] not in found:
                found[row["id"]] = row
    return found


def check():
    from .check import check_sources
    asks = read_jsonl(FULL_FOLDER / "asks.jsonl")
    written = replies()
    asks = [ask for ask in asks if ask["id"] in written]
    by_id = {ask["id"]: ask for ask in asks}
    games = {game.version_id: game for game in load_games(ids={ask["version"] for ask in asks})}
    print(f"{len(asks):,} replies to check, {len(games):,} game versions", flush=True)
    baseline = {}
    names = sorted(games)
    for first in range(0, len(names), CHUNK):
        found = check_sources({name: games[name].source for name in names[first:first + CHUNK]})
        baseline.update({name: Counter(items) for name, items in found.items()})
    rows = []
    for first in range(0, len(asks), CHUNK):
        chunk, sources = [], {}
        for ask in asks[first:first + CHUNK]:
            game = games[ask["version"]]
            block = block_of(written[ask["id"]]["reply"])
            repairs = []
            if block is not None:
                block, repairs = repaired(ask, block, game)
                sources[ask["id"]] = spliced(game.source, ask, block)
            chunk.append({"id": ask["id"], "unit": ask["unit"], "task": ask["task"], "split": ask["split"],
                          "block": block, "repairs": repairs, "finish": written[ask["id"]].get("finish")})
        found = check_sources(sources) if sources else {}
        for row in chunk:
            ask = by_id[row["id"]]
            if row["block"] is None:
                row.update({"kept": False, "new_problems": ["no complete block"], "failed": ["no block"],
                            "follows": False})
                continue
            added = Counter(found[row["id"]]) - baseline[ask["version"]]
            row["new_problems"] = sorted(item for item in added.elements() if not item.startswith(HOLLOW))
            row["kept"] = not row["new_problems"]
            row["failed"] = failed_checks(ask, row["block"], ask["before"], games[ask["version"]])
            row["follows"] = not row["failed"]
        rows += chunk
        print(f"checked {len(rows):,}/{len(asks):,}", flush=True)
    for path in (CHECKED, BLOCKS):
        path.unlink(missing_ok=True)        # derived from the replies
    append_jsonl(CHECKED, rows)
    append_jsonl(BLOCKS, [{"id": row["id"], "block": row["block"], "repairs": row["repairs"]}
                          for row in rows if row["kept"] and row["follows"]])
    print(f"{'unit':<14}{'split':>6}{'asks':>7}{'kept':>7}{'follows':>9}{'both':>7}")
    for unit in UNITS:
        for split in ("train", "test"):
            unit_rows = [row for row in rows if row["unit"] == unit and row["split"] == split]
            if unit_rows:
                count = len(unit_rows)
                share = lambda test: f"{sum(map(test, unit_rows)) / count:.0%}"
                print(f"{unit:<14}{split:>6}{count:>7}{share(lambda row: row['kept']):>7}"
                      f"{share(lambda row: row['follows']):>9}{share(lambda row: row['kept'] and row['follows']):>7}")
    print(f"{sum(row['kept'] and row['follows'] for row in rows):,} targets -> {BLOCKS}")


def vocabulary():
    from transformers import AutoTokenizer
    from library.shared.config import teacher_weights_path
    from .vocabulary import vocabulary_texts
    tokenizer = AutoTokenizer.from_pretrained(teacher_weights_path("google/gemma-4-12B-it"))
    counts = Counter()
    for text in vocabulary_texts() + [f"{FENCE}\n{row['block']}\n{FENCE}" for row in read_jsonl(BLOCKS)]:
        counts.update(tokenizer.tokenize(text))
    tokens = [token for token, _ in counts.most_common()]
    OUTPUT_TOKENS_V2.write_text(json.dumps({"tokens": tokens}, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(tokens):,} tokens -> {OUTPUT_TOKENS_V2}")


def records() -> List[Dict]:
    asks = {ask["id"]: ask for ask in read_jsonl(FULL_FOLDER / "asks.jsonl")}
    found = []
    for row in read_jsonl(BLOCKS):
        ask = asks[row["id"]]
        found.append({"text": ask["text"], "target": f"{FENCE}\n{row['block']}\n{FENCE}", "teacher": "forced",
                      "item_id": ask["id"], "split": ask["split"], "game": ask["version"].rsplit("|", 1)[0],
                      "task": ask["task"], "unit": ask["unit"], "spot": ask["spot"], "repairs": row["repairs"],
                      "writer": ask.get("writer", "rule"), "turn_type": f"{ask['task']}_{ask['unit']}"})
    return found


def build_corpus(path=CORPUS_V2, count=None):
    found = records()
    train = [record for record in found if record["split"] == "train"]
    test = [record for record in found if record["split"] == "test"]
    random.Random(SEED).shuffle(train)
    random.Random(SEED).shuffle(test)
    with open(path, "w", encoding="utf-8") as file:   # derived file, rebuilt each time
        for record in train[:count] + test:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"{len(train[:count]):,} train + {len(test):,} test records -> {path}")
    print(dict(Counter((record["split"], record["unit"]) for record in found)))


if __name__ == "__main__":
    {"check": check, "vocabulary": vocabulary, "corpus": build_corpus}[sys.argv[1] if len(sys.argv) > 1 else "check"]()
