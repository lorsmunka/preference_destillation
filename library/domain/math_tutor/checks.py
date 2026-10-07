"""Checking a written solution block. One parser, three uses:

  harness      a live pupil's problem has no solve(): every line must be true arithmetic and
               well formed (check_lines)
  label pass   Gemma writes the facts-first wording of a gold solution; it is kept only if it
               says exactly what the solve() trace computed (check_against_trace)
  analysis     how teacher-written solutions relate to the function (same numbers or not)

Explanations get their own check (check_explanation): they must not take a checked solution
back, as a tutor led on by a question about an unused detail does ("I missed the hayrides").

A facts-first solution block:

    <solution>
    tetras bought later = 4                  fact: label = number
    price of a tetra = 2
    cost of 4 more tetras = 4 * 2 = 8        calculation: label = expression = result
    money left = 13 - 8 = 5
    Answer: 5
    </solution>
"""

import ast
import re
from collections import Counter
from typing import Dict, List, NamedTuple, Optional

from .items import constants_allowed, digit_numbers, text_numbers

FACT = re.compile(r"^(?P<label>[^=]+?)\s*=\s*(?P<number>\d+)\s*$")
CALCULATION = re.compile(r"^(?P<label>[^=]+?)\s*=\s*(?P<expression>[^=]+?)\s*=\s*(?P<result>.+?)\s*$")
LIST_MARKER = re.compile(r"^\s*(\d+[.)]|[-*•])\s")   # "1. ", "- ": not part of the format
REMAINDER = re.compile(r"^(?P<a>\d+) / (?P<b>\d+) = (?P<q>\d+) remainder (?P<r>\d+)(, rounded up = (?P<up>\d+))?$")


class Line(NamedTuple):
    kind: str            # "fact" | "calculation" | "answer" | "other"
    label: str
    number: Optional[int]   # a fact's number or a calculation's result
    text: str


def parse(block: str) -> List[Line]:
    lines = []
    for raw in block.strip().split("\n"):
        line = raw.strip()
        if not line or line in ("<solution>", "</solution>"):
            continue
        if line.startswith("Answer:"):
            value = line[len("Answer:"):].strip()
            lines.append(Line("answer", "", int(value) if value.isdigit() else None, line))
            continue
        fact = FACT.match(line)
        if fact:
            lines.append(Line("fact", fact.group("label"), int(fact.group("number")), line))
            continue
        calculation = CALCULATION.match(line)
        if calculation:
            result = calculation.group("result")
            remainder = REMAINDER.match(f"{calculation.group('expression')} = {result}")
            if remainder:   # "16 / 3 = 5 remainder 1, rounded up = 6": the value is the last number
                number = int(remainder.group("up") or remainder.group("q"))
            else:
                number = int(result) if result.isdigit() else None
            lines.append(Line("calculation", calculation.group("label"), number, line))
            continue
        lines.append(Line("other", "", None, line))
    return lines


def arithmetic_true(line: Line) -> bool:
    """Whether a calculation line's own arithmetic holds."""
    expression, result = line.text.split("=", 1)[1].rsplit("=", 1) if line.text.count("=") >= 2 else ("", "")
    remainder = REMAINDER.match(f"{expression.strip()} = {result.strip()}")
    if remainder:
        a, b, q, r = (int(remainder.group(name)) for name in ("a", "b", "q", "r"))
        up = remainder.group("up")
        return b > 0 and a == b * q + r and r < b and (up is None or int(up) == q + (1 if r else 0))
    try:
        tree = ast.parse(expression.strip().replace("/", "//"), mode="eval")
        if not all(isinstance(node, (ast.Expression, ast.BinOp, ast.Constant, ast.operator)) for node in ast.walk(tree)):
            return False
        value = eval(compile(tree, "<line>", "eval"), {"__builtins__": {}})
        divisions_exact = all(eval(compile(ast.Expression(node.left), "<l>", "eval"), {"__builtins__": {}})
                              % eval(compile(ast.Expression(node.right), "<r>", "eval"), {"__builtins__": {}}) == 0
                              for node in ast.walk(tree) if isinstance(node, ast.BinOp) and isinstance(node.op, ast.FloorDiv))
        return divisions_exact and line.number is not None and value == line.number
    except Exception:
        return False


def evaluate_expression(expression: str) -> Optional[int]:
    """Whole-number value of '6 * 9' or '84 / 6' (division must be exact), else None."""
    try:
        tree = ast.parse(expression.strip().replace("//", "/").replace("/", "//"), mode="eval")
        if not all(isinstance(node, (ast.Expression, ast.BinOp, ast.Constant, ast.operator)) for node in ast.walk(tree)):
            return None
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.FloorDiv):
                left = eval(compile(ast.Expression(node.left), "<l>", "eval"), {"__builtins__": {}})
                right = eval(compile(ast.Expression(node.right), "<r>", "eval"), {"__builtins__": {}})
                if right == 0 or left % right:
                    return None
        value = eval(compile(tree, "<e>", "eval"), {"__builtins__": {}})
        return value if isinstance(value, int) else None
    except Exception:
        return None


def complete_results(block: str) -> str:
    """The harness's tidy-up before a solution enters the history: a calculation written
    without its result ('total photos = 54 + 12') gets it ('= 66'), so later turns do not copy
    the sloppy form. Lines it cannot evaluate stay as they are. Never used on training data."""
    lines = []
    for line in block.split("\n"):
        parts = line.split("=")
        if len(parts) == 2 and re.search(r"\d\s*[-+*/]\s*\d", parts[1]) and not line.startswith("Answer"):
            value = evaluate_expression(parts[1])
            if value is not None:
                line = f"{line.rstrip()} = {value}"
        lines.append(line)
    return "\n".join(lines)


def check_lines(block: str) -> Optional[str]:
    """The harness check: None if the block is well formed and every calculation is true."""
    lines = parse(block)
    if not lines or lines[-1].kind != "answer" or lines[-1].number is None:
        return "no final Answer: <number> line"
    for line in lines[:-1]:
        if LIST_MARKER.match(line.text):
            return f"numbered or bulleted line: {line.text!r}"
        if line.kind == "other":
            return f"not a fact or a calculation: {line.text!r}"
        if line.kind == "answer":
            return "Answer line before the end"
        if line.kind == "calculation" and not arithmetic_true(line):
            return f"arithmetic does not hold: {line.text!r}"
    return None


CONCESSION = re.compile(
    r"\b(I|we) (missed|forgot|made a mistake|made an error|got it wrong|was wrong|were wrong|should have)\b"
    r"|\bmy (mistake|error)\b|\bsorry\b|\blet me (fix|correct|redo)\b", re.IGNORECASE)


AGREEMENT = re.compile(r"\byou('re| are) (right|correct)\b|\bgood catch\b|\bwell spotted\b", re.IGNORECASE)


def check_explanation(text: str, kind: Optional[str] = None) -> Optional[str]:
    """None if the explanation is usable: not empty, and it does not say the checked solution
    was wrong (that would teach the student to give in to a leading question). To a pupil's
    pushback (kind "pushback") it must not agree either."""
    if not text.strip():
        return "empty explanation"
    concession = CONCESSION.search(text)
    if concession:
        return f"takes back the checked solution: {concession.group(0)!r}"
    if kind == "pushback":
        agreement = AGREEMENT.search(text)
        if agreement:
            return f"agrees with the pupil's wrong answer: {agreement.group(0)!r}"
    return None


def check_against_trace(block: str, computed: List[int], answer: int, context: str) -> Optional[str]:
    """The label-pass check: well formed and true (check_lines); the answer is the function's;
    every calculation's result is a value some operation of the function produced (so a long
    step may be split into child-sized ones, or two joined); every fact is a number in the
    conversation or one a word implies ("a pair" = 2, "a week" = 7); every number a calculation
    uses is a fact, an earlier result, or in the conversation (nothing invented)."""
    problem = check_lines(block)
    if problem:
        return problem
    lines = parse(block)
    if lines[-1].number != answer:
        return f"answer {lines[-1].number}, the function gives {answer}"
    if not any(line.kind == "fact" for line in lines):
        return "no facts listed"
    available = set(text_numbers(context)) | set(digit_numbers(context)) | constants_allowed(context) | {0, 1}
    function_values = set(computed)
    known = set(available)
    for line in lines:
        if line.kind == "fact":
            if line.number not in available:
                return f"fact {line.number} is not in the conversation"
            known.add(line.number)
        elif line.kind == "calculation":
            expression = line.text.split("=", 1)[1].rsplit("=", 1)[0]
            for number in re.findall(r"\d+", expression.split("remainder")[0]):
                if int(number) not in known:
                    return f"calculation uses {number}, which is not a fact, an earlier result or in the conversation"
            if line.number not in function_values:
                return f"result {line.number} is not a value the function computes {sorted(function_values)}"
            known.add(line.number)
    return None


def final_answer(block: str) -> Optional[int]:
    match = re.search(r"Answer:\s*(\d+)", block)
    return int(match.group(1)) if match else None
