"""Gemma words each problem's command: what a user would type at the cursor to get exactly the target. The style
is drawn per problem (stable): terse with #names, loose plain English (no # marks; then the model's input has no
named blocks either, as in the app), detailed with the numbers that matter, or, for /insert only, empty (the
autocomplete case: "/insert" alone, no Gemma call). A command is checked (the right verb, one line, no code, every
#name exists in the game or is the new block's own); a bad one is asked again once, sampled, then dropped.

    python -m library.domain.game_dsl.commands                  # GPU: every problem without a command
    python -m library.domain.game_dsl.commands 200 32           # a trial: the first 200, 32 per model call
    python -m library.domain.game_dsl.commands 200 32 manual    # the same with the manual in the prompt
"""

import json
import re
import sys
import zlib
from pathlib import Path
from typing import Dict, List, Optional

from . import GameDslDomain
from .problems import OUTPUT

COMMANDS = OUTPUT / "commands.jsonl"
STYLES = {"insert": {"terse": 0.50, "loose": 0.25, "detailed": 0.15, "empty": 0.10},
          "edit": {"terse": 0.55, "loose": 0.30, "detailed": 0.15}}
MAX_COMMAND_CHARS = 400

PROMPT = """You are helping make training data for a code assistant inside the editor of a small game language. A user \
types a command at their cursor, starting with /insert (new code at the cursor) or /edit (change the block under the \
cursor), and the assistant writes the code.

What the assistant sees:
{seen}

{change}

Write the command the user typed to get exactly this. {style}
Reply with the command line only."""

STYLE_TEXTS = {
    "terse": "Short, like a quick note, under 20 words. Start with {verb}. Write # in front of every existing name "
             "the code uses ({mentions}){new}. No code and no hex colours.{edit}",
    "loose": "Casual plain English, the way someone new to the language would ask, under 25 words. Start with {verb}. "
             "No # marks: say things in words. No code.{edit}",
    "detailed": "One to three precise sentences with the numbers that matter (sizes, speeds, times, positions). "
                "Start with {verb}. Write # in front of every existing name the code uses ({mentions}){new}. "
                "No code and no hex colours.{edit}",
}


def style_of(problem: Dict) -> str:
    weights = STYLES[problem["task"]]
    point = zlib.crc32(("style" + problem["id"]).encode()) % 1000 / 1000
    total = 0.0
    for style, weight in weights.items():
        total += weight
        if point < total:
            return style
    return list(weights)[-1]


def seen_text(problem: Dict, style: str) -> str:
    from .problems import student_input
    shown = problem if style in ("terse", "detailed") else {**problem, "named_blocks": {}}
    return student_input(shown, "").rsplit("\nCOMMAND", 1)[0]


def prompt_for(problem: Dict, style: str, manual: Optional[str]) -> str:
    verb = "/" + problem["task"]
    if problem["task"] == "insert":
        change = f"The code the assistant must write at the cursor:\n```\n{problem['target']}\n```"
    else:
        change = (f"The block under the cursor now:\n```\n{problem['before']}\n```\n\n"
                  f"What it must become:\n```\n{problem['target']}\n```")
    mentions = ", ".join("#" + name for name in problem["mentions"]) or "none"
    new = f"; the new thing is called #{problem['new_names'][0]}" if problem["new_names"] else ""
    style_text = STYLE_TEXTS[style].format(verb=verb, mentions=mentions, new=new,
                                           edit=" Say only what changes." if problem["task"] == "edit" else "")
    text = PROMPT.format(seen=seen_text(problem, style), change=change, style=style_text)
    return (manual + "\n---\n\n" + text) if manual else text


def game_name_set(problem: Dict) -> set:
    return {line.split()[0][1:] for line in problem["names"]} | set(problem["new_names"]) | {"score", "lives"}


def check_command(problem: Dict, style: str, text: str) -> Optional[str]:
    """The cleaned command, or None when it is not usable."""
    lines = [line.strip() for line in text.strip().strip("`").strip().split("\n") if line.strip()]
    if not lines:
        return None
    command = lines[0].strip('"')
    verb = "/" + problem["task"]
    if not command.lower().startswith(verb) or len(command) > MAX_COMMAND_CHARS or re.search(r"[{}]|\w\(", command):
        return None
    command = verb + command[len(verb):]
    names = re.findall(r"#([A-Za-z_][A-Za-z0-9_]*)", command)
    if style == "loose" and names:
        return None
    if any(name not in game_name_set(problem) for name in names):
        return None
    return command


def load_done() -> Dict[str, Dict]:
    done = {}
    if COMMANDS.exists():
        for line in open(COMMANDS, encoding="utf-8"):
            record = json.loads(line)
            done[record["id"]] = record
    return done


def main():
    from library.data_gen.model_handler import ModelHandler
    from library.agent.runner import teacher_setup
    limit = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "-" else None
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    domain = GameDslDomain()
    manual = domain.manual if "manual" in sys.argv[3:] else None
    problems = [json.loads(line) for line in open(OUTPUT / "problems.jsonl", encoding="utf-8")][:limit]
    done = load_done()
    todo = [problem for problem in problems if problem["id"] not in done]
    with open(COMMANDS, "a", encoding="utf-8") as file:
        for problem in [problem for problem in todo if style_of(problem) == "empty"]:
            file.write(json.dumps({"id": problem["id"], "style": "empty", "command": "/insert", "attempts": 0}) + "\n")
    todo = [problem for problem in todo if style_of(problem) != "empty"]
    if not todo:
        print("every problem has its command")
        return
    handler = ModelHandler("google/gemma-4-12B-it", domain, load_in_4bit=teacher_setup(COMMANDS), chat_template=True)
    todo.sort(key=lambda problem: len(problem["names"]) + len(problem["target"]) // 40)
    kept = dropped = 0
    for start in range(0, len(todo), batch):
        chunk = todo[start:start + batch]
        styles = [style_of(problem) for problem in chunk]
        texts = handler.generate_texts([prompt_for(problem, style, manual) for problem, style in zip(chunk, styles)],
                                       max_new_tokens=120)
        commands = [check_command(problem, style, text) for problem, style, text in zip(chunk, styles, texts)]
        retry = [index for index, command in enumerate(commands) if command is None]
        if retry:
            again = handler.generate_texts([prompt_for(chunk[index], styles[index], manual) for index in retry],
                                           max_new_tokens=120, temperature=0.7, seed=start)
            for index, text in zip(retry, again):
                commands[index] = check_command(chunk[index], styles[index], text)
        with open(COMMANDS, "a", encoding="utf-8") as file:
            for index, (problem, style, command) in enumerate(zip(chunk, styles, commands)):
                if command is None:
                    dropped += 1
                    file.write(json.dumps({"id": problem["id"], "style": style, "command": None,
                                           "attempts": 2, "last_reply": texts[index][:300]}) + "\n")
                else:
                    kept += 1
                    file.write(json.dumps({"id": problem["id"], "style": style, "command": command,
                                           "attempts": 2 if index in retry else 1}) + "\n")
        print(f"{start + len(chunk)}/{len(todo)}: {kept} commands, {dropped} dropped", flush=True)


if __name__ == "__main__":
    main()
