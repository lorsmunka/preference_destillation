"""The game-language student in the app's own checks (2026-11-08): it writes a block for every test problem (its
short input + command, greedy), the block goes into its game at the cursor, the app's compiler and checks run
(gemma_writes.check_written, as for Gemma's blocks). Reported next to Gemma: the test problems are the ones where
Gemma's own block passed, so Gemma is at 100% here by construction (75% on every test problem).

    python experiments/game_student_check.py [run name]       (default: game-agent-kd-m-s1)
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from library.domain.game_dsl import FENCE, GameDslDomain  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "runs" / "game_student_check"


def read_jsonl(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def main():
    from library import RunStore
    from library.domain.game_dsl.gemma_writes import block_of, check_written, compare, summary
    from library.domain.game_dsl.problems import OUTPUT as PROBLEMS
    from library.model.student import StudentModel

    run_name = sys.argv[1] if len(sys.argv) > 1 else "game-agent-kd-m-s1"
    domain = GameDslDomain("card")
    store = RunStore()
    run = store.get(run_name)
    _, checkpoint_path = store.resolve_checkpoint(run_name, "latest")
    student = StudentModel.from_run(run, checkpoint_path, domain=domain)

    tests = [row for row in read_jsonl(Path(domain.corpus_path)) if row["split"] == "test"]
    replies_path = OUTPUT / f"{run_name}_replies.jsonl"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    done = {row["id"] for row in read_jsonl(replies_path)}
    commands = {row["id"]: row.get("command", "") for row in read_jsonl(REPO / "runs" / "game_gemma_run" / "inputs.jsonl")}
    with open(replies_path, "a", encoding="utf-8") as file:
        for number, row in enumerate(tests, 1):
            if row["item_id"] in done:
                continue
            result = student.generate(row["text"], max_new_tokens=domain.max_steps)
            file.write(json.dumps({"id": row["item_id"], "command": commands.get(row["item_id"], ""),
                                   "reply": result.text, "terminated": result.terminated}, ensure_ascii=False) + "\n")
            if number % 100 == 0:
                print(f"written {number}/{len(tests)}", flush=True)

    rows = read_jsonl(replies_path)
    ids = {row["id"] for row in rows}
    problems = {}
    for line in open(PROBLEMS / "problems.jsonl", encoding="utf-8"):
        problem = json.loads(line)
        if problem["id"] in ids:
            problems[problem["id"]] = problem
    checked = check_written(rows, problems)
    gemma = {row["item_id"]: row["target"][len(FENCE) + 1:-(len(FENCE) + 1)] for row in tests}
    for row in checked:      # the same tree / shape as GEMMA's block (the teacher), not the real one
        row.update({f"{key}_gemma": value for key, value in
                    (compare(row["block"], gemma[row["id"]]) if row["block"] else
                     {"same_tree": False, "same_shape": False}).items()})
    result = summary(checked)
    result["same_tree_as_gemma"] = round(sum(row["same_tree_gemma"] for row in checked) / len(checked), 3)
    result["same_shape_as_gemma"] = round(sum(row["same_shape_gemma"] for row in checked) / len(checked), 3)
    print(run_name, json.dumps(result))
    for key in sorted({(row["task"], row["kind"]) for row in checked}):
        part = [row for row in checked if (row["task"], row["kind"]) == key]
        print(" ", key, len(part), "kept", round(sum(row["kept"] for row in part) / len(part), 3),
              "same tree as Gemma", round(sum(row["same_tree_gemma"] for row in part) / len(part), 3))
    (OUTPUT / f"{run_name}_summary.json").write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
