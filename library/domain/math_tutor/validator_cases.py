"""Broken items, one per rule, to show the validator catches each of them.

    python -m library.domain.math_tutor.validator_cases

Each case starts from the same good item and breaks one thing. Prints the reason the
validator gave next to the one expected, and exits with an error if any case slips through.
"""

import sys

from .items import validate

GOOD = {
    "id": "S02-case-0001", "slice": "S02", "unknown": "result", "grade": 2, "distractor": "none",
    "context": "bakery", "source": "hand",
    "problem": "A baker made 36 muffins and sold 19 of them. How many muffins are left?",
    "solve": "def solve():\n    left = 36 - 19\n    return {'left': left, 'answer': left}",
    "follow_ups": [{"question": "She bakes 8 more. How many muffins does she have now?",
                    "solve": "def solve(state):\n    now = state['left'] + 8\n    return {**state, 'left': now, 'answer': now}"}],
}


def broken(problem=None, solve=None, follow_up_solve=None, follow_up_question=None, **fields):
    item = {**GOOD, **fields}
    if problem is not None:
        item["problem"] = problem
    if solve is not None:
        item["solve"] = solve
    if follow_up_solve is not None or follow_up_question is not None:
        follow_up = dict(GOOD["follow_ups"][0])
        follow_up["solve"] = follow_up_solve or follow_up["solve"]
        follow_up["question"] = follow_up_question or follow_up["question"]
        item["follow_ups"] = [follow_up]
    return item


CASES = [
    ("good item passes", GOOD, None),
    ("unknown slice", broken(slice="S99"), "unknown slice"),
    ("grade out of range", broken(grade=6), "grade must be one of"),
    ("too many follow-ups", {**GOOD, "follow_ups": GOOD["follow_ups"] * 4}, "more than 3 follow-ups"),
    ("negative value", broken(solve="def solve():\n    left = 19 - 36\n    return {'answer': left}"), "negative value"),
    ("division with a remainder outside S07",
     broken(problem="A baker shares 36 muffins among 5 boxes. How many in each box?",
            solve="def solve():\n    each = 36 // 5\n    return {'each': each, 'answer': each}"), "division leaves a remainder"),
    ("% outside S07", broken(solve="def solve():\n    left = 36 % 19\n    return {'left': left, 'answer': left}"), "% only in remainder problems"),
    ("answer above 999", broken(problem="A baker made 36 trays of 40 muffins. How many muffins?",
                                solve="def solve():\n    total = 36 * 40\n    return {'answer': total}"), "value above 999"),
    ("number above 99 in the text", broken(problem="A baker made 136 muffins and sold 19. How many are left?"), "number above 99 in the text"),
    ("number in the code but not in the text",
     broken(solve="def solve():\n    left = 36 - 18\n    return {'left': left, 'answer': left}"), "18 is in the code but not in the text"),
    ("label says no distractor, text has an unused number",
     broken(problem="A baker made 36 muffins and 12 pies and sold 19 muffins. How many muffins are left?"), "needs 0 unused number(s), text has 1"),
    ("label says number distractor, text has none", broken(distractor="number"), "needs 1 unused number(s), text has 0"),
    ("number word in plain prose is fine",
     broken(problem="A baker made 36 muffins in her two ovens and sold 19 of them. How many muffins are left?"), None),
    ("distractor written as a word does not count",
     broken(distractor="number", problem="A baker made 36 muffins and twelve pies and sold 19 muffins. How many muffins are left?"),
     "needs 1 unused number(s), text has 0"),
    ("function call", broken(solve="def solve():\n    left = max(36, 19)\n    return {'answer': left}"), "Call not allowed"),
    ("import", broken(solve="def solve():\n    import os\n    return {'answer': 17}"), "Import statement not allowed"),
    ("loop", broken(solve="def solve():\n    for step in range(3):\n        pass\n    return {'answer': 17}"), "For statement not allowed"),
    ("attribute access", broken(solve="def solve():\n    left = (36).real\n    return {'answer': left}"), "Attribute not allowed"),
    ("float literal", broken(solve="def solve():\n    left = 36.0 - 19\n    return {'answer': left}"), "only whole-number literals"),
    ("no answer key", broken(solve="def solve():\n    left = 36 - 19\n    return {'left': left}"), "no 'answer'"),
    ("first turn takes state", broken(solve="def solve(state):\n    return {'answer': 36 - 19}"), "solve must take ()"),
    ("no operation", broken(solve="def solve():\n    return {'answer': 17}"), "no operation"),
    ("follow-up ignores the previous turn",
     broken(follow_up_solve="def solve(state):\n    now = 17 + 8\n    return {'answer': now}"), "follow-up does not use the previous turn"),
    ("follow-up reads a missing state key",
     broken(follow_up_solve="def solve(state):\n    now = state['muffins'] + 8\n    return {'answer': now}"), "state has no 'muffins'"),
    ("follow-up inherits the previous answer",
     broken(follow_up_solve="def solve(state):\n    now = state['left'] + 8\n    return {**state, 'left': now}"), "no 'answer'"),
    ("follow-up's answer overwritten by **state",
     broken(follow_up_solve="def solve(state):\n    now = state['left'] + 8\n    return {'left': now, 'answer': now, **state}"),
     "**state must come before 'answer'"),
    ("follow-up with an unused number",
     broken(follow_up_question="She bakes 8 more and 3 pies. How many muffins does she have now?"), "a follow-up needs 0 unused"),
    ("syntax error", broken(solve="def solve(:\n    return {'answer': 17}"), "syntax error"),
]


def main() -> None:
    failures = 0
    for name, item, expected in CASES:
        validated, reason = validate(item)
        caught = (validated is not None) if expected is None else (reason is not None and expected in reason)
        failures += not caught
        print(f"{'ok  ' if caught else 'MISS'} {name:52} -> {reason or 'valid, answers ' + str(validated['answers'])}")
    print(f"\n{len(CASES) - failures}/{len(CASES)} cases behave as expected")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
