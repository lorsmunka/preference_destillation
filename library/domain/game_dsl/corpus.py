"""The corpus: one record per problem with a usable command and a kept Gemma block for that exact command
(gemma_writes.BLOCKS, from experiments/game_gemma_run.py). text = the short input with the command (the code of the
command's #names only, as the app fetches it; none for a loose or empty command); target = Gemma's own block in a
code fence (the redesign of 2026-11-07: the student learns Gemma's code, not the real block); Gemma scores it
teacher-forced with the prompt it wrote it with. Train records shuffled first, then test (the test set is the
batch directory's tail; each record's split label wins anyway).

    python -m library.domain.game_dsl.corpus
"""

import json
import random
from collections import Counter

from . import FENCE, PACKAGE
from .commands import load_done
from .gemma_writes import BLOCKS
from .problems import OUTPUT, as_typed, student_input
from .sonnet_commands import target_crc

CORPUS = PACKAGE / "corpus.jsonl"
SEED = 0


def records():
    commands = load_done()
    blocks = {}
    for line in open(BLOCKS, encoding="utf-8"):
        row = json.loads(line)
        blocks[row["id"]] = row
    found = []
    for line in open(OUTPUT / "problems.jsonl", encoding="utf-8"):
        problem = json.loads(line)
        command = commands.get(problem["id"])
        block = blocks.get(problem["id"])
        if command is None or command["command"] is None or block is None or block["command"] != command["command"]:
            continue        # no command, or no kept Gemma block for this exact command
        if "target_crc" in command and command["target_crc"] != target_crc(problem):
            continue        # written for an older problems.jsonl
        found.append({"text": student_input(as_typed(problem, command["command"]), command["command"]),
                      "target": f"{FENCE}\n{block['block']}\n{FENCE}", "teacher": "forced",
                      "same_tree": block["same_tree"], "same_shape": block["same_shape"],
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
