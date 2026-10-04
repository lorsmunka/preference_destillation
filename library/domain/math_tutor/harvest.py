"""Harvested problems: real word problems copied word for word from openly licensed or
public-domain sources, each with the solve() the harvester wrote for it. They become
extra human seeds (benchmarks.py merges them), never test items.

    python -m library.domain.math_tutor.harvest [file.jsonl]

checks one harvest file (a harvester's self-check), or every file in benchmarks/harvested/.

One record:
    {"id": "hv-im-0001", "problem": "<the text exactly as on the page>",
     "solve": "def solve():\\n    ...\\n    return {'answer': ...}",
     "stated_answer": 15,                 # the page's own answer, or null
     "url": "https://...", "site": "illustrativemathematics.org",
     "licence": "CC BY 4.0", "grade": 3}  # grade or null
"""

import json
import sys
from pathlib import Path
from typing import Dict, Optional

from .functions import NotAllowed, execute
from .items import constants_allowed, digit_numbers, text_numbers

HARVEST_DIR = Path(__file__).resolve().parent / "benchmarks" / "harvested"
REQUIRED_FIELDS = ("id", "problem", "solve", "url", "site", "licence")
MAX_TEXT_NUMBER = 99
MAX_STEPS = 3


def check_harvested(record: Dict) -> Optional[str]:
    """None if the record can become a seed, else the first rule it breaks."""
    for field in REQUIRED_FIELDS:
        if not record.get(field):
            return f"missing field '{field}'"
    if any(number > MAX_TEXT_NUMBER for number in digit_numbers(record["problem"])):
        return "number above 99 in the text"
    try:
        execution = execute(record["solve"], allow_remainders=True)
    except NotAllowed as error:
        return str(error)
    if not execution.operations:
        return "no operation"
    if len(execution.operations) > MAX_STEPS:
        return "more than 3 operations"
    allowed = set(text_numbers(record["problem"])) | constants_allowed(record["problem"]) | {0, 1}
    for literal in execution.literals:
        if literal not in allowed:
            return f"{literal} is in the code but not in the text"
    stated = record.get("stated_answer")
    if stated is not None and stated != execution.result["answer"]:
        return f"solve() gives {execution.result['answer']}, the page says {stated}"
    return None


def main() -> None:
    paths = [Path(sys.argv[1])] if len(sys.argv) > 1 else sorted(HARVEST_DIR.glob("*.jsonl"))
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
            reason = check_harvested(record)
            if reason:
                failures.append(f"line {line_number} {record.get('id', '?')}: {reason}")
            else:
                valid += 1
        print(f"{path.name}: {valid} valid, {len(failures)} failing")
        for failure in failures:
            print("  " + failure)


if __name__ == "__main__":
    main()
