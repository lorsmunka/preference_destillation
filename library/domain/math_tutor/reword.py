"""Rewordings: the wording layer of the training set. A rewording agent only writes new
text — the problem and each follow-up question — for an item that already passed; its
solve() functions are copied from the original, so the maths cannot drift. The expanded
item must pass the full validator, and the pruning pipeline then drops any rewording
that keeps 80% of the original's words in order (near-duplicate), so only real rewrites
survive.

A rewording file (items/reword/*.jsonl, written by the agents) holds compact records:
    {"id": "S04-mem-b01-0007~w1", "of": "S04-mem-b01-0007",
     "problem": "<new text>", "follow_up_questions": ["<new text>", ...]}

    python -m library.domain.math_tutor.reword [file.jsonl]

checks one rewording file (an agent's self-check), or every file in items/reword/.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .items import validate

ITEMS_DIR = Path(__file__).resolve().parent / "items"
REWORD_DIR = ITEMS_DIR / "reword"


def raw_items_by_id() -> Dict[str, Dict]:
    items = {}
    for path in sorted((ITEMS_DIR / "raw").glob("*.jsonl")):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                items[item.get("id")] = item
    return items


def expand(record: Dict, originals: Dict[str, Dict]) -> Tuple[Optional[Dict], Optional[str]]:
    """Compact rewording -> a full item (the original's fields and code, the new text)."""
    original = originals.get(record.get("of"))
    if original is None:
        return None, f"unknown original {record.get('of')!r}"
    questions = record.get("follow_up_questions", [])
    follow_ups = original.get("follow_ups", [])
    if len(questions) != len(follow_ups):
        return None, f"{len(questions)} follow-up questions for {len(follow_ups)} follow-ups"
    item = dict(original)
    item["id"] = record.get("id")
    item["problem"] = record.get("problem", "")
    item["follow_ups"] = [{"question": question, "solve": follow_up["solve"]} for question, follow_up in zip(questions, follow_ups)]
    item["source"] = f"reword:{original['id']}"
    item["family"] = original.get("family", original["id"])
    return item, None


def read_rewordings(originals: Optional[Dict[str, Dict]] = None) -> List[Dict]:
    """Every rewording expanded to a full item, ready for prune.py (broken ones carry a
    _reword_error the execute stage reports)."""
    originals = originals if originals is not None else raw_items_by_id()
    items = []
    for path in sorted(REWORD_DIR.glob("*.jsonl")):
        for line_number, line in enumerate(open(path, encoding="utf-8"), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                items.append({"id": f"{path.stem}:{line_number}", "_broken_json": True})
                continue
            item, error = expand(record, originals)
            items.append(item if item else {"id": record.get("id", f"{path.stem}:{line_number}"), "_reword_error": error})
    return items


def main() -> None:
    originals = raw_items_by_id()
    paths = [Path(sys.argv[1])] if len(sys.argv) > 1 else sorted(REWORD_DIR.glob("*.jsonl"))
    for path in paths:
        valid, failures = 0, []
        for line_number, line in enumerate(open(path, encoding="utf-8"), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                failures.append(f"line {line_number}: not valid json ({error.msg})")
                continue
            item, error = expand(record, originals)
            if item is not None:
                validated, error = validate(item)
            if error:
                failures.append(f"line {line_number} {record.get('id', '?')}: {error}")
            else:
                valid += 1
        print(f"{path.name}: {valid} valid, {len(failures)} failing")
        for failure in failures:
            print("  " + failure)


if __name__ == "__main__":
    main()
