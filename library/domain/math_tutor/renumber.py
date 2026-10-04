"""New numbers for kept items: the arithmetic layer of the training set. The wording stays,
the quantities change, solve() is run again and the copy is kept only if it still passes
every rule (whole numbers, 0..999, exact division, distractor still unused...).

    python -m library.domain.math_tutor.renumber

A "slot" is a number written in the text (as digits or as a word) that is not 0 or 1.
Each slot gets a new value near the old one (half to double, so "4 friends" does not
become 90 friends), drawn without repeats, and every place that value appears — text and
code, in every turn — changes together. Numbers the text only implies ("twice", "a week")
stay as they are. Copies keep the original's family and split, so a test item's copies
stay in test. Reads items/items.jsonl, rebuilds items/renumbered.jsonl and
items/renumber_log.json.
"""

import ast
import json
import random
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

from .items import TENS, UNITS, constants_allowed, text_numbers, validate

ITEMS_DIR = Path(__file__).resolve().parent / "items"
COPIES_PER_ITEM = 3
ATTEMPTS_PER_COPY = 30
SEED = 0

WORD_BY_VALUE = {value: word for word, value in UNITS.items() if word != "dozen"}
WORD_BY_VALUE.update({value: word for word, value in TENS.items()})


def number_to_words(value: int) -> str:
    if value in WORD_BY_VALUE:
        return WORD_BY_VALUE[value]
    return f"{WORD_BY_VALUE[value - value % 10]}-{WORD_BY_VALUE[value % 10]}"


def replace_in_text(text: str, new_values: Dict[int, int]) -> str:
    """Swap every written number that has a new value, keeping digits as digits and words
    as words (with the capital letter if it had one)."""
    def swap(match):
        token = match.group(0)
        if token.isdigit():
            return str(new_values.get(int(token), int(token)))
        values = text_numbers(token)
        if len(values) != 1 or values[0] not in new_values or token.lower() == "dozen":
            return token
        words = number_to_words(new_values[values[0]])
        return words.capitalize() if token[0].isupper() else words
    return re.sub(r"\d+|[A-Za-z]+(?:-[A-Za-z]+)?", swap, text)


def replace_in_code(source: str, new_values: Dict[int, int]) -> str:
    class Swap(ast.NodeTransformer):
        def visit_Constant(self, node):
            if type(node.value) is int and node.value in new_values:
                return ast.copy_location(ast.Constant(new_values[node.value]), node)
            return node
    return ast.unparse(Swap().visit(ast.parse(source)))


def slots_of(item: Dict) -> Optional[List[int]]:
    """The values that can change, or None when a value is ambiguous: written in the text
    but also implied by a word like "twice" somewhere, so the code's copy of it could be either."""
    texts = [item["problem"]] + [follow_up["question"] for follow_up in item["follow_ups"]]
    written = {value for text in texts for value in text_numbers(text) if value > 1}
    implied = set().union(*(constants_allowed(text) for text in texts))
    if written & implied:
        return None
    return sorted(written)


def draw(old: int, taken: set, random_generator: random.Random) -> Optional[int]:
    low, high = max(2, round(old / 2)), min(99, old * 2 + 3)
    choices = [value for value in range(low, high + 1) if value not in taken]
    return random_generator.choice(choices) if choices else None


def renumber_item(item: Dict, random_generator: random.Random) -> (List[Dict], str):
    slots = slots_of(item)
    if slots is None:
        return [], "a written number is also implied by a word"
    if not slots:
        return [], "no written numbers"
    copies, seen_answers = [], {tuple(item["answers"])}
    for attempt in range(COPIES_PER_ITEM * ATTEMPTS_PER_COPY):
        if len(copies) == COPIES_PER_ITEM:
            break
        new_values, taken = {}, set()
        for old in slots:
            new = draw(old, taken, random_generator)
            if new is None:
                break
            new_values[old] = new
            taken.add(new)
        if len(new_values) != len(slots) or all(old == new for old, new in new_values.items()):
            continue
        candidate = dict(item)
        candidate["id"] = f"{item['id']}~n{len(copies) + 1}"
        candidate["problem"] = replace_in_text(item["problem"], new_values)
        candidate["solve"] = replace_in_code(item["solve"], new_values)
        candidate["follow_ups"] = [{"question": replace_in_text(f["question"], new_values),
                                    "solve": replace_in_code(f["solve"], new_values)} for f in item["follow_ups"]]
        validated, _ = validate(candidate)
        if validated is None or tuple(validated["answers"]) in seen_answers:
            continue
        seen_answers.add(tuple(validated["answers"]))
        validated["family"], validated["split"] = item["family"], item["split"]
        validated["renumbered_from"] = item["id"]
        copies.append(validated)
    return copies, "ok" if copies else "no valid draw"


def renumber() -> None:
    random_generator = random.Random(SEED)
    items = [json.loads(line) for line in open(ITEMS_DIR / "items.jsonl", encoding="utf-8")]
    copies, outcomes, copies_per_item = [], Counter(), Counter()
    for item in items:
        item_copies, outcome = renumber_item(item, random_generator)
        outcomes[outcome] += 1
        copies_per_item[len(item_copies)] += 1
        copies.extend(item_copies)
    with open(ITEMS_DIR / "renumbered.jsonl", "w", encoding="utf-8") as file:
        for copy in copies:
            file.write(json.dumps(copy, ensure_ascii=False) + "\n")
    log = {"built": date.today().isoformat(), "copies_per_item_target": COPIES_PER_ITEM, "seed": SEED,
           "items": len(items), "copies": len(copies), "outcomes": dict(outcomes.most_common()),
           "items_by_copies_made": dict(sorted(copies_per_item.items())),
           "copies_by_split": dict(Counter(copy["split"] for copy in copies))}
    with open(ITEMS_DIR / "renumber_log.json", "w", encoding="utf-8") as file:
        json.dump(log, file, indent=2)
    print(f"{len(items)} items -> {len(copies)} renumbered copies  {dict(outcomes)}  copies made: {dict(sorted(copies_per_item.items()))}")


if __name__ == "__main__":
    renumber()
