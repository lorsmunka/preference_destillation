"""The commands (what the user types at the cursor) written by Claude Sonnet agents instead of Gemma. Decided
2026-11-07 after a test on 60 problems Gemma had also worded: Sonnet's commands passed the command check 60/60
(Gemma 53/60, mostly code in the command) and say what the code does rather than reading its syntax back. Its one
repeated slip, reading y as pointing down, is answered by the coordinate paragraph in INSTRUCTIONS.

The instructions are written once at the top of each task file; each task carries what the assistant sees, the
code, the style and the #names to use. One Sonnet agent per task file (AGENT_PROMPT); its answers go to a .tsv
next to the task file, which `collect` checks (commands.check_command) and appends to problems/commands.jsonl.
Problems whose style is "empty" (/insert alone, autocomplete) need no agent and are written straight away.

    python -m library.domain.game_dsl.sonnet_commands tasks 300        # 300 problems without a command, 60 per file
    python -m library.domain.game_dsl.sonnet_commands collect          # every answer file -> commands.jsonl
"""

import json
import re
import sys
import zlib
from pathlib import Path
from typing import Dict, List, Optional

from .commands import COMMANDS, load_done, seen_text, style_of, check_command
from .language import parse_fragment
from .problems import OUTPUT

TASKS = OUTPUT / "sonnet"
PER_FILE = 60

INSTRUCTIONS = """You are writing training data for a code assistant inside the editor of a small game language (the app \
"QR Game Studio"). A user types a one-line command at the cursor: /insert (new code at the cursor) or /edit (change the \
block under the cursor), and the assistant writes the code. For each task below, write the command the user typed to \
get exactly that code.

Each task shows what the assistant sees: the WINDOW line; NAMES, one line per entity / group / bar / signal with its \
kind, shape, movement and events; CURSOR, where the cursor is; then the full code of the names the command mentions \
(terse and detailed only). Then the code: for /insert the code written at the cursor, for /edit what the block \
under the cursor (BLOCK) must become.

Coordinates: x and y are pixels from the BOTTOM-left corner and y grows UPWARD. size(w, h) on the WINDOW line is in \
units of 100 px: size(7, 10) is 700 x 1000 px, so y near 1000 is the top edge and y = 0 the bottom edge. \
startPosition(x, y) is an entity's bottom-left corner. worldSize(w, h), when given, is the whole world the window \
scrolls over, also in 100 px. speed(N) is N x 100 px per second; setVelocity and applyImpulse are px per second; \
times are seconds.

Styles (each task names its own):
- terse: a quick note, under 20 words.
- loose: casual plain English, the way someone new to the language would ask, under 25 words. No # marks at all: \
say things in words ("the falling rocks", "the score bar").
- detailed: one to three precise sentences with the numbers that matter (sizes, speeds, times, positions), said in \
words, never copied as an argument list.

Every command:
- starts with the task's verb (/insert or /edit), is one line, reads like something a person typed, and says what \
the code does (behaviour, not syntax);
- has no code: no braces, no word directly followed by "(", no hex colours, no keyword written as code; colour names \
("dark red") are fine;
- terse and detailed: mention the names that matter, each with # in front: the ones the code relies on (for /edit, only the ones the change involves). Only names on the task's "#names" line may get a #. loose: no # at all;
- for /edit, says only what changes;
- is specific enough that a programmer seeing the same context could write close to that code; terse and loose \
need not give every number.

Answers: append to {answers}, one line per task: <n><TAB><id><TAB><command> (the command on one line, no quotes \
around it). Append every ~10 tasks (Bash: cat >> file <<'EOF' ... EOF); never rewrite the file. All tasks, in order.
"""

AGENT_PROMPT = """Read the instructions at the top of {tasks} and do every task in it. The file is {size} KB: read it \
in parts (Read with offset/limit, ~400 lines at a time). Do not run any code, touch any other file, or use git. When \
done, reply with how many commands you wrote and any task whose instructions were impossible to follow (one line \
each)."""


def target_crc(problem: Dict) -> int:
    """Ties a command to the exact problem it was written for (problem ids are only stable while problems.jsonl is)."""
    return zlib.crc32((problem.get("before", "") + "\n" + problem["target"]).encode())


def names_line(problem: Dict, style: str) -> str:
    if style == "loose":
        return "#names: none (loose: no # marks)"
    new = set(problem["new_names"])
    listed = [f"#{name}" + (" (new: not in NAMES)" if name in new else "")
              for name in dict.fromkeys(problem["new_names"] + problem["mentions"])]
    own = edited_name(problem)
    if own and own not in problem["mentions"]:
        listed.append(f"#{own} (the block under the cursor)")
    return "#names: " + (", ".join(listed) if listed else "none")


def edited_name(problem: Dict) -> Optional[str]:
    """The id of the Entity (or the name of the Bar) an /edit rewrites, None for any other block."""
    if problem["task"] != "edit":
        return None
    node = parse_fragment(problem["before"])[0]
    if node.key == "entity" and node.child("id"):
        return node.child("id").first_string()
    return node.first_string() if node.key == "bar" else None


def task_text(number: int, problem: Dict, style: str) -> str:
    verb = "/" + problem["task"]
    parts = [f"===== TASK {number} | id={problem['id']} | {verb} | style={style}", names_line(problem, style),
             "--- what the assistant sees", seen_text(problem, style)]
    if problem["task"] == "insert":
        parts += ["--- the code written at the cursor", problem["target"]]
    else:
        parts += ["--- what the block (BLOCK above) must become", problem["target"]]
    return "\n".join(parts) + "\n"


def spread_over_games(problems: List[Dict], count: int) -> List[Dict]:
    """`count` problems taken round-robin over the games (each game's own order seeded by the problem ids)."""
    by_game = {}
    for problem in sorted(problems, key=lambda problem: zlib.crc32(("pick" + problem["id"]).encode())):
        by_game.setdefault(problem["game"], []).append(problem)
    picked, round_index = [], 0
    while len(picked) < count and any(round_index < len(group) for group in by_game.values()):
        for game in sorted(by_game):
            if round_index < len(by_game[game]) and len(picked) < count:
                picked.append(by_game[game][round_index])
        round_index += 1
    return picked


def write_tasks(count: int) -> List[Path]:
    done = load_done()
    problems = [json.loads(line) for line in open(OUTPUT / "problems.jsonl", encoding="utf-8")]
    todo = [problem for problem in problems if problem["id"] not in done]
    picked = spread_over_games(todo, count)
    with open(COMMANDS, "a", encoding="utf-8") as file:
        for problem in picked:
            if style_of(problem) == "empty":
                file.write(json.dumps({"id": problem["id"], "style": "empty", "command": "/insert", "attempts": 0,
                                       "writer": "none", "target_crc": target_crc(problem)}) + "\n")
    asked = [problem for problem in picked if style_of(problem) != "empty"]
    TASKS.mkdir(parents=True, exist_ok=True)
    first = 1 + max([int(re.search(r"\d+", path.stem).group()) for path in TASKS.glob("tasks_*.txt")] or [0])
    paths = []
    for offset in range(0, len(asked), PER_FILE):
        path = TASKS / f"tasks_{first + offset // PER_FILE:03d}.txt"
        if path.exists():
            raise FileExistsError(path)
        chunk = asked[offset:offset + PER_FILE]
        body = [INSTRUCTIONS.format(answers=path.with_suffix(".tsv").as_posix())]
        body += [task_text(number, problem, style_of(problem)) for number, problem in enumerate(chunk, 1)]
        path.write_text("\n".join(body), encoding="utf-8")
        paths.append(path)
    print(f"{len(picked) - len(asked)} empty-style commands written; {len(asked)} tasks in {len(paths)} files:")
    for path in paths:
        print(f"  {path}  ({path.stat().st_size // 1024} KB)")
    return paths


def parse_answers(text: str) -> Dict[str, str]:
    answers = {}
    for line in text.splitlines():
        parts = line.rstrip("\r").split("\t")
        if len(parts) >= 3 and parts[1].strip():
            answers[parts[1].strip()] = "\t".join(parts[2:]).strip()
    return answers


def collect() -> None:
    done = load_done()
    problems = {}
    for line in open(OUTPUT / "problems.jsonl", encoding="utf-8"):
        problem = json.loads(line)
        problems[problem["id"]] = problem
    kept = failed = waiting = 0
    with open(COMMANDS, "a", encoding="utf-8") as file:
        for tasks_path in sorted(TASKS.glob("tasks_*.txt")):
            asked = re.findall(r"^===== TASK \d+ \| id=(\S+) \|", tasks_path.read_text(encoding="utf-8"), re.M)
            answers_path = tasks_path.with_suffix(".tsv")
            answers = parse_answers(answers_path.read_text(encoding="utf-8")) if answers_path.exists() else {}
            for problem_id in asked:
                if problem_id in done:
                    continue
                if problem_id not in answers:
                    waiting += 1
                    continue
                problem = problems[problem_id]
                style = style_of(problem)
                command = check_command(problem, style, answers[problem_id])
                record = {"id": problem_id, "style": style, "command": command, "attempts": 1, "writer": "sonnet",
                          "target_crc": target_crc(problem)}
                if command is None:
                    record["last_reply"] = answers[problem_id][:300]
                    failed += 1
                else:
                    kept += 1
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
                done[problem_id] = record
    print(f"{kept} commands kept, {failed} failed the check, {waiting} not answered yet -> {COMMANDS}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "tasks":
        write_tasks(int(sys.argv[2]) if len(sys.argv) > 2 else 300)
    elif len(sys.argv) > 1 and sys.argv[1] == "collect":
        collect()
    else:
        print(__doc__)
