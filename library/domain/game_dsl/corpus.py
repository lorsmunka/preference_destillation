"""The corpus: one record per problem that has a usable command. text = the short input with the command (a loose
or empty command has no named blocks: the app only fetches the code of #names); target = the real block in a code
fence; Gemma scores it teacher-forced. Train records shuffled first, then test (the test set is the batch
directory's tail; each record's split label wins anyway).

    python -m library.domain.game_dsl.corpus
"""

import json
import random
from collections import Counter

from . import FENCE, PACKAGE
from .commands import load_done
from .problems import OUTPUT, student_input

CORPUS = PACKAGE / "corpus.jsonl"
SEED = 0


def records():
    commands = load_done()
    found = []
    for line in open(OUTPUT / "problems.jsonl", encoding="utf-8"):
        problem = json.loads(line)
        command = commands.get(problem["id"])
        if command is None or command["command"] is None:
            continue
        shown = problem if command["style"] in ("terse", "detailed") else {**problem, "named_blocks": {}}
        found.append({"text": student_input(shown, command["command"]),
                      "target": f"{FENCE}\n{problem['target']}\n{FENCE}", "teacher": "forced",
                      "item_id": problem["id"], "split": problem["split"], "game": problem["game"],
                      "family": problem["family"], "origin": problem["origin"], "task": problem["task"],
                      "kind": problem["kind"], "style": command["style"],
                      "turn_type": f"{problem['task']}_{problem['kind']}"})
    return found


def build_corpus(path=CORPUS, count=None):
    found = records()
    train = [record for record in found if record["split"] == "train"]
    test = [record for record in found if record["split"] == "test"]
    random.Random(SEED).shuffle(train)
    random.Random(SEED).shuffle(test)
    with open(path, "w", encoding="utf-8") as file:   # derived file, rebuilt each time
        for record in train[:count] + test:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"{len(train[:count]):,} train + {len(test):,} test records -> {path}")
    print(dict(Counter((record["split"], record["task"], record["style"]) for record in found)))


if __name__ == "__main__":
    build_corpus()
