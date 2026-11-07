"""Commands written by an open model on a rented card (2026-11-07, user: "get as many examples as we can for $65"):
gpt-oss-120b served by vLLM, the same instructions and tasks as the Sonnet agents (sonnet_commands), one task per
request, the reply = the command alone. Checked at home with commands.check_command, like Sonnet's.

    python -m library.domain.game_dsl.open_commands test           # the 120 tasks Sonnet and Haiku did (tasks_127-128)
    python -m library.domain.game_dsl.open_commands tasks 27000    # problems without a command -> tasks.jsonl
    (on the card, with `vllm serve` running: python experiments/open_commands_run.py <tasks file>)
    python -m library.domain.game_dsl.open_commands compare        # the test answers next to Sonnet's
    python -m library.domain.game_dsl.open_commands collect        # the answers to tasks.jsonl -> commands.jsonl
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

from .commands import COMMANDS, check_command, load_done, style_of
from .problems import OUTPUT as PROBLEMS
from .sonnet_commands import INSTRUCTIONS, TASKS as SONNET_TASKS, parse_answers, shape_key, spread_over_games, \
    target_crc, task_text

REPO = Path(__file__).resolve().parents[3]
OUTPUT = REPO / "runs" / "open_commands"
SYSTEM = OUTPUT / "system.txt"
TEST_TASKS = OUTPUT / "test_tasks.jsonl"
TASKS = OUTPUT / "tasks.jsonl"
WRITER = "gpt-oss-120b"
# the Sonnet instructions up to their "Answers:" paragraph (answer files), then what the first test (120 tasks,
# 2026-11-07) showed gpt-oss doing wrong: reading the code back (keywords, hex colours, coordinate lists)
SYSTEM_PROMPT = INSTRUCTIONS[:INSTRUCTIONS.index("Answers:")] + """
The user never reads the code: never write a keyword of the language (OnCollision, OnElapsedTime, whenVar, \
changeVar, spawnHere, setColor, updateBar, ...), a hex colour, an argument list or a list of coordinates. Say what \
happens in the game, in words a player would use (colour names, "every 2 seconds", "near the top", "in three \
columns"). Examples (the change -> a good command; a bad one and why):
- gameOver("-- SWEPT AWAY --") added to #tulip's collision
  good: /edit end the game with a swept away message when #tulip hits it
  bad:  /edit add gameOver("-- SWEPT AWAY --") when moonHaze is 0   (code read back)
- a burst colour changed from "10b981" to "94a3b8" where the enemy dies
  good: /edit make the burst when the enemy dies grey instead of green
  bad:  /edit replace the burst colour code 10b981 with 94a3b8   (hex codes)
- nine spawns of #grain at (350, 400), (350, 800), ... and six of #moth
  good: /insert on a timer, spawn nine #grain bars in three columns and six #moth bars between them
  bad:  /insert spawn #grain at (350,400), (350,800), (350,1200), ...   (a coordinate list)
- changeVar("drillB", -3) added inside whenVar("drillB", "==", 3) of a tap (style loose)
  good: /edit when the box is tapped in its last stage, also reset its stage back to the beginning
  bad:  /edit add a changeVar("drillB", -3) inside the whenVar drillB == 3 block   (keywords and code)
- spawnHere("candle") inside a fast repeating timer
  good: /insert in the repeating row timer, also drop a #candle at the current spot
  bad:  /insert add an action that spawns a #candle during the OnElapsedTime(0.02) loop   (an event name, code numbers)

Reply with the command only: one line that starts with the task's verb, nothing else (no quotes, no explanation, \
no task number or id).
"""
CODE_WORDS = re.compile(r"(?<![#\w])(On[A-Z]\w+|when(Var|Bar)|(change|set|update)[A-Z]\w*|spawn[A-Z]\w*|"
                        r"repeatCount|randomAction|deleteEntity|bounceBack)\b")
HEX = re.compile(r"(?<![#\w])(?=[0-9a-fA-F]*\d)(?=[0-9a-fA-F]*[a-fA-F])[0-9a-fA-F]{6}\b")
COORDINATES = re.compile(r"\(\s*-?\d+(\.\d+)?\s*,\s*-?\d+")


def reads_code(command: str) -> bool:
    """A command that reads the code back (a keyword of the language, a hex colour, coordinates in brackets):
    dropped, the user never sees the code."""
    return bool(CODE_WORDS.search(command) or HEX.search(command) or COORDINATES.search(command))


def checked(problem, style, command):
    command = check_command(problem, style, command)
    return None if command is None or reads_code(command) else command


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


def load_problems(ids=None):
    problems = {}
    for line in open(PROBLEMS / "problems.jsonl", encoding="utf-8"):
        problem = json.loads(line)
        if ids is None or problem["id"] in ids:
            problems[problem["id"]] = problem
    return problems


def task_row(problem):
    style = style_of(problem)
    return {"id": problem["id"], "style": style, "user": task_text(1, problem, style)}


def write_system():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    SYSTEM.write_text(SYSTEM_PROMPT, encoding="utf-8")


def test(numbers=("127", "128")):
    """The tasks of Sonnet task files `numbers` (Sonnet's answers to compare with); 127-128 = the first test (the
    Haiku test's files too), 129-130 = the second (after the examples were added from what the first showed)."""
    write_system()
    ids = []
    for number in numbers:
        ids += re.findall(r"^===== TASK \d+ \| id=(\S+) \|", (SONNET_TASKS / f"tasks_{number}.txt")
                          .read_text(encoding="utf-8"), re.M)
    problems = load_problems(set(ids))
    path = test_path(numbers)
    path.unlink(missing_ok=True)               # derived from the Sonnet task files
    append_jsonl(path, [task_row(problems[problem_id]) for problem_id in ids])
    print(f"{len(ids)} test tasks -> {path}")


def test_path(numbers):
    return TEST_TASKS if tuple(numbers) == ("127", "128") else OUTPUT / f"test_{'_'.join(numbers)}.jsonl"


def tasks(count):
    """`count` more problems without a command (spread over the games, at most SHAPE_CAP per shape counting the
    ones with commands and the ones already queued); the empty-style ones (/insert alone) go straight to
    commands.jsonl, as in sonnet_commands.write_tasks."""
    write_system()
    done = load_done()
    queued = {row["id"] for row in read_jsonl(TASKS)}
    problems = load_problems()
    todo = [problem for problem in problems.values() if problem["id"] not in done and problem["id"] not in queued]
    taken = Counter(shape_key(problem) for problem in problems.values()
                    if problem["id"] in done or problem["id"] in queued)
    picked = spread_over_games(todo, count, taken)
    with open(COMMANDS, "a", encoding="utf-8") as file:
        for problem in picked:
            if style_of(problem) == "empty":
                file.write(json.dumps({"id": problem["id"], "style": "empty", "command": "/insert", "attempts": 0,
                                       "writer": "none", "target_crc": target_crc(problem)}) + "\n")
    asked = [task_row(problem) for problem in picked if style_of(problem) != "empty"]
    append_jsonl(TASKS, asked)
    print(f"{len(picked) - len(asked)} empty-style commands written; {len(asked):,} tasks -> {TASKS} "
          f"({len(queued) + len(asked):,} queued in all)")


def answers_of(tasks_path):
    """{id: the reply's command} from <tasks>_answers.jsonl (a later line for the same id wins)."""
    found = {}
    for row in read_jsonl(tasks_path.with_name(tasks_path.stem + "_answers.jsonl")):
        if row.get("command"):
            found[row["id"]] = " ".join(row["command"].strip().splitlines()[:1]).strip().strip('"')
    return found


def compare(numbers=("127", "128")):
    answers = answers_of(test_path(numbers))
    rows = read_jsonl(test_path(numbers))
    problems = load_problems({row["id"] for row in rows})
    sonnet = {}
    for number in numbers:
        sonnet.update(parse_answers((SONNET_TASKS / f"tasks_{number}.tsv").read_text(encoding="utf-8")))
    passed = [row["id"] for row in rows if row["id"] in answers
              and checked(problems[row["id"]], row["style"], answers[row["id"]])]
    print(f"{WRITER}: answered {len(answers)}/{len(rows)}, pass the check {len(passed)}, mean words "
          f"{sum(len(command.split()) for command in answers.values()) / max(len(answers), 1):.1f}")
    for row in rows:
        print(f"==== {row['id']} {problems[row['id']]['task']} {row['style']}")
        print(f"  S: {sonnet.get(row['id'])}")
        print(f"  O: {answers.get(row['id'])}")


def collect():
    rows = read_jsonl(TASKS)
    answers = answers_of(TASKS)
    done = load_done()
    problems = load_problems({row["id"] for row in rows})
    kept = failed = waiting = 0
    with open(COMMANDS, "a", encoding="utf-8") as file:
        for row in rows:
            if row["id"] in done:
                continue
            if row["id"] not in answers:
                waiting += 1
                continue
            problem = problems[row["id"]]
            command = checked(problem, row["style"], answers[row["id"]])
            record = {"id": row["id"], "style": row["style"], "command": command, "attempts": 1, "writer": WRITER,
                      "target_crc": target_crc(problem)}
            if command is None:
                record["last_reply"] = answers[row["id"]][:300]
                failed += 1
            else:
                kept += 1
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            done[row["id"]] = record
    print(f"{kept:,} commands kept, {failed:,} failed the check, {waiting:,} not answered yet -> {COMMANDS}")


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else ""
    if action == "test":
        test(tuple(sys.argv[2:]) or ("127", "128"))
    elif action == "tasks":
        tasks(int(sys.argv[2]))
    elif action == "compare":
        compare(tuple(sys.argv[2:]) or ("127", "128"))
    elif action == "collect":
        collect()
    else:
        print(__doc__)
