"""One training item: a word problem, its solve() and up to three follow-ups, each with a
solve(state) that continues from the dict the previous turn returned. `validate` runs the
functions and checks every rule in thesis_work/domain_specs.txt (VALIDATION); an item
either comes back with its answers and shape filled in, or with the first rule it broke.
"""

import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

from .functions import NotAllowed, execute

SLICES = {
    "S01": "change, join",
    "S02": "change, separate",
    "S03": "combine",
    "S04": "compare, additive",
    "S05": "equal groups",
    "S06": "partitive division",
    "S07": "quotative division, remainders",
    "S08": "multiplicative compare",
    "S09": "rates and money",
    "S10": "multi-step mixed",
}
REMAINDER_SLICE = "S07"
GRADES = (2, 3, 4)
DISTRACTORS = ("none", "number", "sentence", "keyword")
REQUIRED_FIELDS = ("id", "slice", "unknown", "grade", "distractor", "context", "source", "problem", "solve")
MAX_FOLLOW_UPS = 3
MAX_OPERATIONS_PER_TURN = 3
MAX_TEXT_NUMBER = 99

UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "dozen": 12,
}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}

# Numbers a problem may use without writing them as a number: "twice" means 2, "a week" 7.
CONSTANT_WORDS = {
    "twice": {2}, "double": {2}, "doubles": {2}, "half": {2}, "halves": {2}, "pair": {2}, "pairs": {2},
    "both": {2}, "triple": {3}, "triples": {3}, "third": {3}, "thirds": {3}, "quarter": {4, 25},
    "quarters": {4, 25}, "dozen": {12}, "dozens": {12}, "week": {7}, "weeks": {7}, "hour": {60},
    "hours": {60}, "minute": {60}, "minutes": {60}, "day": {24}, "days": {24}, "year": {12},
    "years": {12}, "dime": {10}, "dimes": {10}, "nickel": {5}, "nickels": {5}, "cent": {100}, "cents": {100},
}


def text_numbers(text: str) -> List[int]:
    """Every quantity written in the text, as digits or words: '28 apples and twenty-one
    pears' -> [28, 21]."""
    numbers = []
    for token in re.findall(r"\d+|[A-Za-z]+(?:-[A-Za-z]+)?", text):
        if token.isdigit():
            numbers.append(int(token))
            continue
        parts = token.lower().split("-")
        if len(parts) == 2 and parts[0] in TENS and parts[1] in UNITS and UNITS[parts[1]] < 10:
            numbers.append(TENS[parts[0]] + UNITS[parts[1]])
        else:
            numbers.extend(UNITS.get(part, TENS.get(part)) for part in parts if part in UNITS or part in TENS)
    return numbers


def constants_allowed(text: str) -> set:
    words = re.findall(r"[a-z]+", text.lower())
    return {value for word in words for value in CONSTANT_WORDS.get(word, ())}


def check_turn(text: str, execution, turn: int, distractor: str) -> Optional[str]:
    """The rules that tie one turn's code to its text."""
    if not execution.operations:
        return f"turn {turn}: no operation"
    if len(execution.operations) > MAX_OPERATIONS_PER_TURN:
        return f"turn {turn}: more than {MAX_OPERATIONS_PER_TURN} operations"
    written = text_numbers(text)
    allowed = set(written) | constants_allowed(text) | {0, 1}
    for literal in execution.literals:
        if literal not in allowed:
            return f"turn {turn}: {literal} is in the code but not in the text"
    unused = Counter(n for n in written if n > 1) - Counter(execution.literals)
    unused_count = sum(unused.values())
    expected = 1 if (turn == 0 and distractor == "number") else 0
    if unused_count != expected:
        label = f"distractor '{distractor}'" if turn == 0 else "a follow-up"
        return f"turn {turn}: {label} needs {expected} unused number(s), text has {unused_count} {sorted(unused.elements())}"
    return None


def validate(item: Dict) -> Tuple[Optional[Dict], Optional[str]]:
    """(item with answers, shape and family filled in, None) or (None, first broken rule)."""
    for field in REQUIRED_FIELDS:
        if field not in item:
            return None, f"missing field '{field}'"
    if item["slice"] not in SLICES:
        return None, f"unknown slice {item['slice']!r}"
    if item["grade"] not in GRADES:
        return None, f"grade must be one of {GRADES}"
    if item["distractor"] not in DISTRACTORS:
        return None, f"distractor must be one of {DISTRACTORS}"
    follow_ups = item.get("follow_ups", [])
    if len(follow_ups) > MAX_FOLLOW_UPS:
        return None, f"more than {MAX_FOLLOW_UPS} follow-ups"

    turns = [(item["problem"], item["solve"])] + [(f.get("question", ""), f.get("solve", "")) for f in follow_ups]
    allow_remainders = item["slice"] == REMAINDER_SLICE
    state, answers, shapes = None, [], []
    for turn, (text, source) in enumerate(turns):
        if not text.strip():
            return None, f"turn {turn}: empty text"
        if any(number > MAX_TEXT_NUMBER for number in text_numbers(text)):
            return None, f"turn {turn}: number above {MAX_TEXT_NUMBER} in the text"
        try:
            execution = execute(source, state, allow_remainders)
        except NotAllowed as error:
            return None, f"turn {turn}: {error}"
        if turn > 0 and not execution.reads_state:
            return None, f"turn {turn}: follow-up does not use the previous turn"
        problem = check_turn(text, execution, turn, item["distractor"])
        if problem:
            return None, problem
        state = execution.result
        answers.append(state["answer"])
        shapes.append(">".join(execution.operations))

    validated = dict(item)
    validated["follow_ups"] = follow_ups
    validated["answers"] = answers
    validated["shape"] = "|".join(shapes)
    validated["problem_shape"] = shapes[0]
    validated["family"] = item.get("family", item["id"])
    return validated, None


def main() -> None:
    """python -m library.domain.math_tutor.items [file.jsonl]
    Checks one file of items (a writer's self-check), or every file in items/raw/."""
    import json
    import sys
    from pathlib import Path

    paths = [Path(sys.argv[1])] if len(sys.argv) > 1 else sorted((Path(__file__).resolve().parent / "items" / "raw").glob("*.jsonl"))
    for path in paths:
        valid, failures = 0, []
        for line_number, line in enumerate(open(path, encoding="utf-8"), start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                failures.append(f"line {line_number}: not valid json ({error.msg})")
                continue
            validated, reason = validate(item)
            if validated:
                valid += 1
            else:
                failures.append(f"line {line_number} {item.get('id', '?')}: {reason}")
        print(f"{path.name}: {valid} valid, {len(failures)} failing")
        for failure in failures:
            print("  " + failure)


if __name__ == "__main__":
    main()
