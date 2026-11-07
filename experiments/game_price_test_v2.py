"""Price test v2 of the game-language agent (2026-11-07, the redesign: Gemma writes the blocks), on a rented card in
bf16, with the fixed reference card in front of every input (user: "fixed is it"). The 300 pilot problems (the
ones with Sonnet's commands, library/domain/game_dsl/problems/commands.jsonl):

  write   Gemma writes each block (greedy, text only, 16 per call): seconds per block
  score   every complete block, teacher-forced with the same prompt (the second pass): seconds per block; Gemma's
          top guess = its own token (should be ~1, the same prompt for writing and scoring); the share of blocks
          with a token outside the current output vocabulary (the full run rebuilds it from Gemma's blocks)
  check   at home (the app's compiler is a Windows program): each block into its game at the cursor, compiled and
          checked (gemma_writes.check_written): kept share, the same tree / shape as the real block

    python experiments/game_price_test_v2.py sample   # at home -> runs/game_price_test_v2/sample.jsonl
    python experiments/game_price_test_v2.py run      # on the card (~15 min)
    python experiments/game_price_test_v2.py report   # at home: the checks and the summary
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from library.domain.game_dsl import FENCE, GameDslDomain  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "runs" / "game_price_test_v2"
WRITE_BATCH, SCORE_BATCH = 16, 16
PRICE_PER_HOUR = 2.49          # RTX PRO 6000 Blackwell, secure cloud (the first price test's card)


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


def sample():
    from library.domain.game_dsl.commands import load_done
    from library.domain.game_dsl.problems import OUTPUT as PROBLEMS, as_typed, student_input
    path = OUTPUT / "sample.jsonl"
    if path.exists():
        print(f"{path} exists")
        return
    commands = {key: value for key, value in load_done().items() if value["command"]}
    rows = []
    for line in open(PROBLEMS / "problems.jsonl", encoding="utf-8"):
        problem = json.loads(line)
        command = commands.get(problem["id"])
        if command is None:
            continue
        rows.append({"id": problem["id"], "task": problem["task"], "kind": problem["kind"], "split": problem["split"],
                     "style": command["style"], "command": command["command"],
                     "text": student_input(as_typed(problem, command["command"]), command["command"]),
                     "target": problem["target"]})
    append_jsonl(path, rows)
    print(f"{len(rows)} problems -> {path}")


def run():
    import torch
    from library.data_gen.model_handler import ModelHandler
    from library.domain.game_dsl.gemma_writes import block_of
    rows = read_jsonl(OUTPUT / "sample.jsonl")
    domain = GameDslDomain("card")
    handler = ModelHandler("google/gemma-4-12B-it", domain, load_in_4bit=False, chat_template=True)
    tokenizer = handler.tokenizer
    summary = {"card": torch.cuda.get_device_name(0), "problems": len(rows)}

    # pass 1: Gemma writes (sorted by the real block's length so a call's rows end at about the same time; the real
    # block only sets the token budget, Gemma never sees it)
    written_path = OUTPUT / "written.jsonl"
    done = {row["id"] for row in read_jsonl(written_path)}
    todo = sorted([row for row in rows if row["id"] not in done], key=lambda row: len(row["target"]))
    handler.generate_texts([domain.teacher_prompt(todo[0]["text"])], max_new_tokens=8)        # warm-up, not timed
    write_seconds = 0.0
    for first in range(0, len(todo), WRITE_BATCH):
        chunk = todo[first:first + WRITE_BATCH]
        longest = max(len(tokenizer(row["target"], add_special_tokens=False)["input_ids"]) for row in chunk)
        budget = min(domain.max_steps, int(longest * 1.5) + 64)
        torch.cuda.synchronize()
        start = time.time()
        replies = handler.generate_texts([domain.teacher_prompt(row["text"]) for row in chunk], max_new_tokens=budget)
        torch.cuda.synchronize()
        seconds = time.time() - start
        write_seconds += seconds
        append_jsonl(written_path, [{"id": row["id"], "reply": reply, "call_seconds": round(seconds, 2),
                                     "call_size": len(chunk), "budget": budget} for row, reply in zip(chunk, replies)])
        print(f"write {first + len(chunk)}/{len(todo)}: {seconds:.1f} s for {len(chunk)} (budget {budget})", flush=True)
    written = {row["id"]: row for row in read_jsonl(written_path)}
    summary["write"] = {"seconds_per_block": write_seconds / max(len(todo), 1),
                        "prompt_tokens": sum(len(tokenizer(domain.teacher_prompt(row["text"]))["input_ids"])
                                             for row in rows) / len(rows),
                        "reply_tokens": sum(len(tokenizer(row["reply"])["input_ids"]) for row in written.values())
                                        / len(written)}
    print("write", summary["write"], flush=True)

    # pass 2 (here before the checks, for the timing): every complete block, teacher-forced, the same prompt
    complete = [(row, block_of(written[row["id"]]["reply"])) for row in rows]
    complete = [(row, block) for row, block in complete if block is not None]
    texts = [row["text"] for row, _ in complete]
    targets = [f"{FENCE}\n{block}\n{FENCE}" for _, block in complete]
    handler.score_training_examples(texts[:2], targets[:2])                                 # warm-up, not timed
    torch.cuda.synchronize()
    start = time.time()
    scored = []
    for first in range(0, len(texts), SCORE_BATCH):
        scored += handler.score_training_examples(texts[first:first + SCORE_BATCH], targets[first:first + SCORE_BATCH])
    torch.cuda.synchronize()
    score_seconds = time.time() - start
    rows_out, agreements, reasons, tokens = [], [], Counter(), 0
    for (row, _), (example, reason) in zip(complete, scored):
        reasons[reason or "kept"] += 1
        if example is not None:
            agreements.append(example["teacher_agreement"])
            tokens += len(example["steps"])
        rows_out.append({"id": row["id"], "reason": reason,
                         "agreement": example["teacher_agreement"] if example else None,
                         "steps": len(example["steps"]) if example else None})
    append_jsonl(OUTPUT / "scored.jsonl", rows_out)
    summary["score"] = {"blocks": len(complete), "seconds_per_block": score_seconds / max(len(complete), 1),
                        "outcomes": dict(reasons), "mean_agreement": sum(agreements) / max(len(agreements), 1),
                        "min_agreement": min(agreements) if agreements else None,
                        "target_tokens": tokens / max(len(agreements), 1)}
    print("score", summary["score"], flush=True)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


def report():
    from library.domain.game_dsl.gemma_writes import check_written, summary as check_summary
    from library.domain.game_dsl.problems import OUTPUT as PROBLEMS
    rows = read_jsonl(OUTPUT / "written.jsonl")
    ids = {row["id"] for row in rows}
    problems = {}
    for line in open(PROBLEMS / "problems.jsonl", encoding="utf-8"):
        problem = json.loads(line)
        if problem["id"] in ids:
            problems[problem["id"]] = problem
    checked = check_written(rows, problems)
    path = OUTPUT / "checked.jsonl"
    path.unlink(missing_ok=True)            # derived from written.jsonl, rebuilt each time
    append_jsonl(path, checked)
    print("all", json.dumps(check_summary(checked)))
    for key in sorted({(row["task"], row["kind"]) for row in checked}):
        print(" ", key, json.dumps(check_summary([row for row in checked if (row["task"], row["kind"]) == key])))
    summary_path = OUTPUT / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print(json.dumps(summary, indent=1))
        kept = check_summary(checked)["kept"]
        # every block is written, only the kept ones are scored
        per_kept = summary["write"]["seconds_per_block"] / max(kept, 1e-9) + summary["score"]["seconds_per_block"]
        print(f"card time per kept block (written + scored): {per_kept:.2f} s = "
              f"${per_kept * PRICE_PER_HOUR / 3600 * 10000:.0f} per 10,000 kept blocks at ${PRICE_PER_HOUR}/h")


if __name__ == "__main__":
    {"sample": sample, "run": run, "report": report}[sys.argv[1] if len(sys.argv) > 1 else "report"]()
