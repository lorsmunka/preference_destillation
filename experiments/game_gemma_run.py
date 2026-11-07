"""Gemma's full run for the game-language agent (2026-11-07, user: "lets start the runpod stuff"): every problem with
a usable command (library/domain/game_dsl/problems/commands.jsonl), with the fixed reference card in front
(GameDslDomain("card"), user: "fixed is it"). Price test v2 measured on the same card: 1.44 s per written block
(16 per call), 0.57 s per scored block, 75% of the blocks kept.

  inputs  (home)  runs/game_gemma_run/inputs.jsonl: what Gemma reads for each problem (uploaded to the card)
  write   (card)  pass 1: Gemma writes each block, greedy, WRITE_BATCH per call -> written.jsonl (resumes)
  check   (home)  each block into its game at the cursor, compiled and checked (gemma_writes.check_written)
                  -> checked.jsonl; the kept blocks -> library/domain/game_dsl/problems/gemma_blocks.jsonl
  then at home:   python -m library.domain.game_dsl.vocabulary    (Gemma's kept blocks are part of it)
                  python -m library.domain.game_dsl.corpus        (targets = Gemma's kept blocks)
  score   (card)  pass 2: the framework's teacher data, teacher-forced with the same prompt -> batches/game_dsl/

    python experiments/game_gemma_run.py inputs | write | check | score
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from library.domain.game_dsl import GameDslDomain  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "runs" / "game_gemma_run"
INPUTS = OUTPUT / "inputs.jsonl"
WRITTEN = OUTPUT / "written.jsonl"
CHECKED = OUTPUT / "checked.jsonl"
TEACHER = "google/gemma-4-12B-it"
WRITE_BATCH = 16
CHECK_CHUNK = 1000      # spliced games per compiler run


def read_jsonl(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def append_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def inputs():
    from library.domain.game_dsl.commands import load_done
    from library.domain.game_dsl.problems import OUTPUT as PROBLEMS, as_typed, student_input
    from library.domain.game_dsl.sonnet_commands import target_crc
    commands = {key: value for key, value in load_done().items() if value["command"]}
    done = {row["id"] for row in read_jsonl(INPUTS)}
    rows = []
    for line in open(PROBLEMS / "problems.jsonl", encoding="utf-8"):
        problem = json.loads(line)
        command = commands.get(problem["id"])
        if command is None or problem["id"] in done:
            continue
        if "target_crc" in command and command["target_crc"] != target_crc(problem):
            continue        # written for an older problems.jsonl
        rows.append({"id": problem["id"], "task": problem["task"], "kind": problem["kind"], "split": problem["split"],
                     "style": command["style"], "command": command["command"],
                     "text": student_input(as_typed(problem, command["command"]), command["command"]),
                     "target": problem["target"]})        # the real block: only the token budget on the card
    append_jsonl(INPUTS, rows)
    print(f"{len(rows):,} new inputs ({len(done):,} already there) -> {INPUTS}")
    print(dict(Counter((row["split"], row["task"]) for row in read_jsonl(INPUTS))))


def write():
    """Pass 1 on the card: the rows sorted by the real block's length (a call's rows end at about the same time;
    Gemma never sees the real block), budget 1.5x the longest real block + 64 tokens."""
    from library.data_gen.model_handler import ModelHandler
    domain = GameDslDomain("card")
    rows = read_jsonl(INPUTS)
    done = {row["id"] for row in read_jsonl(WRITTEN)}
    todo = sorted([row for row in rows if row["id"] not in done], key=lambda row: len(row["target"]))
    print(f"{len(todo):,} blocks to write ({len(done):,} written before)", flush=True)
    if not todo:
        return
    handler = ModelHandler(TEACHER, domain, load_in_4bit=False, chat_template=True)
    tokenizer = handler.tokenizer
    start = time.time()
    for first in range(0, len(todo), WRITE_BATCH):
        if (REPO / "STOP").exists():
            print("STOP file: stopping (the pass resumes where it stopped)", flush=True)
            return
        chunk = todo[first:first + WRITE_BATCH]
        longest = max(len(tokenizer(row["target"], add_special_tokens=False)["input_ids"]) for row in chunk)
        budget = min(domain.max_steps, int(longest * 1.5) + 64)
        call_start = time.time()
        replies = handler.generate_texts([domain.teacher_prompt(row["text"]) for row in chunk], max_new_tokens=budget)
        append_jsonl(WRITTEN, [{"id": row["id"], "command": row["command"], "reply": reply, "budget": budget}
                               for row, reply in zip(chunk, replies)])
        written = first + len(chunk)
        hours_left = (time.time() - start) / written * (len(todo) - written) / 3600
        print(f"write {written:,}/{len(todo):,}: {time.time() - call_start:.1f} s for {len(chunk)} "
              f"(budget {budget}), ~{hours_left:.1f} h left", flush=True)


def prompts(skip_first=0):
    """For the second card (experiments/gemma_vllm_run.py): every input not written yet, after the first
    `skip_first` of them in write()'s order (those stay with the first card), as the full teacher prompt with the
    same token budget as write() -> vllm_prompts.jsonl (rebuilt each time)."""
    from transformers import AutoTokenizer
    from library.shared.config import teacher_weights_path
    domain = GameDslDomain("card")
    tokenizer = AutoTokenizer.from_pretrained(teacher_weights_path(TEACHER))
    written = {row["id"] for row in read_jsonl(WRITTEN)} | {row["id"] for row in read_jsonl(OUTPUT / "vllm_written.jsonl")
                                                            if row["reply"]}
    todo = sorted([row for row in read_jsonl(INPUTS) if row["id"] not in written], key=lambda row: len(row["target"]))
    rows = []
    for row in todo[skip_first:]:
        budget = min(domain.max_steps, int(len(tokenizer(row["target"], add_special_tokens=False)["input_ids"]) * 1.5) + 64)
        rows.append({"id": row["id"], "command": row["command"], "prompt": domain.teacher_prompt(row["text"]),
                     "budget": budget})
    path = OUTPUT / "vllm_prompts.jsonl"
    path.unlink(missing_ok=True)
    append_jsonl(path, rows)
    print(f"{len(rows):,} prompts for the second card ({len(todo):,} not written, the first {skip_first:,} left "
          f"to the first card) -> {path}")


def check():
    """At home (the app's compiler is a Windows program): every written block into its game, compiled and checked,
    CHECK_CHUNK games per compiler run; rebuilt from written.jsonl each time."""
    from library.domain.game_dsl.gemma_writes import BLOCKS, check_written, summary
    from library.domain.game_dsl.problems import OUTPUT as PROBLEMS
    rows, seen = [], set()
    for row in read_jsonl(WRITTEN) + [row for row in read_jsonl(OUTPUT / "vllm_written.jsonl") if row["reply"]]:
        if row["id"] not in seen:        # both cards' replies (the first card's wins if both wrote one)
            seen.add(row["id"])
            rows.append(row)
    ids = {row["id"] for row in rows}
    problems = {}
    for line in open(PROBLEMS / "problems.jsonl", encoding="utf-8"):
        problem = json.loads(line)
        if problem["id"] in ids:
            problems[problem["id"]] = problem
    checked = []
    for first in range(0, len(rows), CHECK_CHUNK):
        checked += check_written(rows[first:first + CHECK_CHUNK], problems)
        print(f"checked {len(checked):,}/{len(rows):,}", flush=True)
    CHECKED.unlink(missing_ok=True)             # derived from written.jsonl
    append_jsonl(CHECKED, checked)
    BLOCKS.unlink(missing_ok=True)              # derived from written.jsonl
    append_jsonl(BLOCKS, [{key: row[key] for key in ("id", "command", "block", "hollow", "same_tree", "same_shape")}
                          for row in checked if row["kept"]])
    print("all", json.dumps(summary(checked)))
    for key in sorted({(row["task"], row["kind"]) for row in checked}):
        print(" ", key, json.dumps(summary([row for row in checked if (row["task"], row["kind"]) == key])))
    print(f"{sum(row['kept'] for row in checked):,} kept blocks -> {BLOCKS}")


def score():
    """Pass 2 on the card: the framework's teacher data from corpus.jsonl (built at home after the vocabulary)."""
    from library import GenerationJob
    from library.data_gen.generator import DistillationDataGenerator
    from library.shared import ExitListener
    job = GenerationJob(domain=GameDslDomain("card"), teacher_model=TEACHER, load_in_4bit=False, chat_template=True,
                        max_examples=1_000_000, batch_size=32, generation_batch_size=16,
                        description="game_dsl: Gemma's own kept blocks, teacher-forced, reference card, bf16")
    exit_listener = ExitListener()
    DistillationDataGenerator(job, exit_listener).run()
    exit_listener.stop()


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "check"
    if action == "prompts":
        prompts(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    else:
        {"inputs": inputs, "write": write, "check": check, "score": score}[action]()
