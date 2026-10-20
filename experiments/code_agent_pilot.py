"""Coding agent pilot (user idea, 2026-10-20): can Gemma do the two steps at all?

The context is a whole small project (library/domain/code_agent/projects/banking: 15 files of
20-50 lines, every line numbered). For each pupil message Gemma first writes a scope, the files
and lines that matter:
    <scope>
    accounts.py 29-42
    </scope>
The harness then narrows the context to those lines, and Gemma decides: <explanation>...
</explanation> when the pupil asked for one, <found/> when they only asked where something is.

Scored here: the scope against the true places (found from the code itself), as the share of
the true lines it covers and the share of its lines that are true ones; the decision; and the
names an explanation uses (any that are not in the project are made up). The explanations
themselves are read by hand.

Run 2026-10-20 14:38 (results/code_agent_pilot.jsonl; its stored recall / precision undercount the
questions with two true places in one file, a scoring bug fixed since: rescore from "scope"):
find 10 questions, true lines covered 92%, scope lines that are true 67%, parts hit 13/14, decision
right 10/10; explain 8, covered 92%, on target 62%, parts hit 14/17, decision right 7/8 (the
statement question's reply did not open with a tag; the raw reply was not kept then). All 7
explanations correct on reading, 0 made-up names. The ~6,200-token context fills the 4070
(peak 10 GB + ~3 GB desktop): from question 13 on, 45-270 s per question instead of 6-18 s.

    python experiments/code_agent_pilot.py    # GPU: Gemma 4 12B (NF4), ~15-30 minutes
"""

import ast
import json
import re
import sys
import time
from pathlib import Path

from _common import RESULTS_DIR

PROJECT_DIR = Path(__file__).resolve().parents[1] / "library" / "domain" / "code_agent" / "projects" / "banking"
TEACHER = "google/gemma-4-12B-it"
RESULTS_PATH = RESULTS_DIR / "code_agent_pilot.jsonl"   # appended, one line per question per run

# (kind, pupil message, the true places as (file, name); "Class.method" for a method)
QUESTIONS = [
    ("find", "Where is the PIN checked when someone logs in?", [("auth.py", "check_login")]),
    ("find", "Find the withdraw method.", [("accounts.py", "Account.withdraw")]),
    ("find", "Which file has the Transaction class, and on which lines?", [("transactions.py", "Transaction")]),
    ("find", "Find the transaction logic.", [("transactions.py", "Transaction"), ("transactions.py", "record_transaction"),
                                            ("transactions.py", "transactions_of")]),
    ("find", "Where is the overdraft fee worked out?", [("fees.py", "overdraft_fee")]),
    ("find", "Find the code that sends money from one account to another.", [("transfers.py", "transfer")]),
    ("find", "Where are the transactions saved to the JSON file?", [("storage.py", "save_transactions"),
                                                                    ("storage.py", "save_json")]),
    ("find", "Find where the menu choices are handled.", [("main.py", "handle_choice")]),
    ("find", "Where is the largest allowed withdrawal set?", [("config.py", "MAX_WITHDRAWAL")]),
    ("find", "Find the interest logic.", [("interest.py", "monthly_interest"), ("interest.py", "add_interest")]),
    ("explain", "Explain how a withdrawal works.", [("accounts.py", "Account.withdraw"), ("validation.py", "check_withdrawal"),
                                                     ("fees.py", "overdraft_fee")]),
    ("explain", "Explain how logging in works.", [("auth.py", "login"), ("auth.py", "check_login")]),
    ("explain", "Explain how a transfer works.", [("transfers.py", "transfer")]),
    ("explain", "Explain how the account statement is printed.", [("history.py", "print_statement")]),
    ("explain", "Explain how transactions are stored.", [("transactions.py", "record_transaction"),
                                                          ("transactions.py", "Transaction.to_dict"),
                                                          ("storage.py", "save_transactions"), ("storage.py", "save_json")]),
    ("explain", "Explain what happens at the end of the month.", [("admin.py", "end_of_month"), ("fees.py", "charge_monthly_fee"),
                                                                   ("interest.py", "add_interest")]),
    ("explain", "Explain how the monthly fee is decided.", [("fees.py", "monthly_fee"), ("fees.py", "charge_monthly_fee")]),
    ("explain", "Explain how a new account is opened.", [("admin.py", "open_account")]),
]

SCOPE_PROMPT = """You are the code assistant inside a tool. Below is every file of a small Python project, each line numbered.
A user asks about the code. First pick the lines that matter: reply with a scope block only, one range per line as "file.py start-end", covering every part needed to answer and as little else as possible. Write nothing else.
<scope>
file.py 12-30
</scope>

<project>
{project}
</project>

<message>{message}</message>"""

ANSWER_PROMPT = """You are the code assistant inside a tool. The tool has narrowed a Python project to the lines that matter for the user's message (the line numbers are the files' own).
If the user asked for an explanation, explain how this code works in a few plain sentences for a beginner, naming the functions and classes involved:
<explanation>
...
</explanation>
If the user only asked where something is, reply with exactly <found/> and nothing else.

<context>
{context}
</context>

<message>{message}</message>"""


def project_files():
    return {path.name: path.read_text(encoding="utf-8").splitlines() for path in sorted(PROJECT_DIR.glob("*.py"))}


def render(files, ranges=None):
    """Files with numbered lines; with `ranges` ({file: [(start, end)]}) only those lines."""
    blocks = []
    for name, lines in files.items():
        if ranges is not None and name not in ranges:
            continue
        shown = [(number, line) for number, line in enumerate(lines, 1)
                 if ranges is None or any(start <= number <= end for start, end in ranges[name])]
        body = "\n".join(f"{number:>2}: {line}" for number, line in shown)
        blocks.append(f'<file name="{name}">\n{body}\n</file>')
    return "\n".join(blocks)


def symbol_lines(files):
    """(file, name) -> (first line, last line) of every class, function, method and top-level constant."""
    places = {}
    for name, lines in files.items():
        for node in ast.parse("\n".join(lines)).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                places[(name, node.name)] = (node.lineno, node.end_lineno)
                if isinstance(node, ast.ClassDef):
                    for member in node.body:
                        if isinstance(member, ast.FunctionDef):
                            places[(name, f"{node.name}.{member.name}")] = (member.lineno, member.end_lineno)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        places[(name, target.id)] = (node.lineno, node.end_lineno)
    return places


def project_names(files):
    """Every identifier the project defines or uses, plus the file names without .py."""
    names = {name[:-3] for name in files}
    for lines in files.values():
        for node in ast.walk(ast.parse("\n".join(lines))):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.arg):
                names.add(node.arg)
    return names


def parse_scope(text, files):
    """{file: [(start, end)]} from the scope lines; ranges are clipped to the file."""
    ranges = {}
    for match in re.finditer(r"([\w./-]+\.py)\s+(\d+)(?:\s*-\s*(\d+))?", text):
        name, start = match.group(1).split("/")[-1], int(match.group(2))
        end = int(match.group(3) or start)
        if name in files:
            ranges.setdefault(name, []).append((max(1, start), min(end, len(files[name]))))
    return ranges


def lines_of(ranges):
    return {(name, number) for name, spans in ranges.items() for start, end in spans for number in range(start, end + 1)}


def score_scope(ranges, truth, places):
    entry_lines = [lines_of({name: [places[(name, symbol)]]}) for name, symbol in truth]
    true_lines = set().union(*entry_lines)
    chosen = lines_of(ranges)
    entries_hit = sum(len(lines & chosen) * 2 >= len(lines) for lines in entry_lines)
    return {"recall": round(len(true_lines & chosen) / len(true_lines), 3),
            "precision": round(len(true_lines & chosen) / len(chosen), 3) if chosen else 0.0,
            "entries_hit": entries_hit, "entries": len(truth)}


def made_up_names(explanation, names):
    """Code-looking words (snake_case, CamelCase or followed by a bracket) the project does not have."""
    candidates = set(re.findall(r"`([A-Za-z_][\w.]*)", explanation))
    candidates |= set(re.findall(r"\b([a-z]+_[a-z_]+|[A-Z][a-z]+[A-Z]\w*)\b", explanation))
    candidates |= set(re.findall(r"\b([A-Za-z_]\w*)\(", explanation))
    parts = {part for candidate in candidates for part in candidate.split(".") if part}
    return sorted(part for part in parts if part not in names and not part.endswith("py") and part not in ("json",))


def main():
    from library.data_gen.model_handler import ModelHandler
    from library.domain import CODE_RECALL
    import torch

    files = project_files()
    places = symbol_lines(files)
    unknown = [place for _, _, truth in QUESTIONS for place in truth if place not in places]
    assert not unknown, f"true places not in the project: {unknown}"
    names = project_names(files)
    project = render(files)
    print(f"{len(files)} files, {sum(len(lines) for lines in files.values())} lines, {len(project)} characters")

    handler = ModelHandler(TEACHER, CODE_RECALL, load_in_4bit=True, chat_template=True)
    print(f"context: {len(handler.tokenizer(SCOPE_PROMPT.format(project=project, message=''))['input_ids'])} tokens; "
          f"GPU memory {torch.cuda.memory_allocated() / 2**30:.1f} GB", flush=True)
    run_id = time.strftime("%Y-%m-%d %H:%M")
    rows = []
    for kind, message, truth in QUESTIONS:
        start = time.time()
        scope_text = handler.generate_texts([SCOPE_PROMPT.format(project=project, message=message)], max_new_tokens=100,
                                            stop_strings=["</scope>"], prefixes=["<scope>\n"])[0]
        ranges = parse_scope(scope_text, files)
        scope_score = score_scope(ranges, truth, places)
        answer = handler.generate_texts([ANSWER_PROMPT.format(context=render(files, ranges), message=message)],
                                        max_new_tokens=300, stop_strings=["</explanation>", "<found/>"], prefixes=["<"])[0]
        route = "explain" if answer.startswith("explanation>") else "find" if answer.startswith("found/>") else "other"
        explanation = answer[len("explanation>"):].replace("</explanation>", "").strip() if route == "explain" else ""
        row = {"run": run_id, "kind": kind, "message": message, "truth": truth, "scope": scope_text.replace("</scope>", "").strip(),
               **scope_score, "route": route, "route_right": route == kind, "reply": answer, "explanation": explanation,
               "made_up_names": made_up_names(explanation, names), "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
               "seconds": round(time.time() - start, 1)}
        rows.append(row)
        print(f"\n[{kind}] {message}\n  scope: {row['scope']!r}\n  true lines covered {row['recall']:.0%}, scope lines that are "
              f"true {row['precision']:.0%}, entries hit {row['entries_hit']}/{row['entries']}; route {route}; "
              f"{row['seconds']} s", flush=True)
        if explanation:
            print(f"  explanation: {explanation}\n  made-up names: {row['made_up_names']}", flush=True)
    with open(RESULTS_PATH, "a", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    for kind in ("find", "explain"):
        group = [row for row in rows if row["kind"] == kind]
        print(f"\n{kind}: {len(group)} questions; true lines covered {sum(r['recall'] for r in group) / len(group):.0%}, "
              f"scope lines that are true {sum(r['precision'] for r in group) / len(group):.0%}, entries hit "
              f"{sum(r['entries_hit'] for r in group)}/{sum(r['entries'] for r in group)}, route right "
              f"{sum(r['route_right'] for r in group)}/{len(group)}")
    print(f"-> {RESULTS_PATH}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
