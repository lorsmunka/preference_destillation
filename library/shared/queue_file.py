"""The planned work of a script, written once at start so the dashboard can show a queue.

    write_queue("e01-anneal-direction", [
        {"name": run.run_name, "description": run.description, "info_path": "runs/<name>/info.json"},
        ...
    ])

Each item names the info.json that will hold its status ("in_progress" / "completed"); an
item whose file does not exist yet is pending. Files live in queue/ (gitignored), one per
script invocation tag, overwritten on every start.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List

QUEUE_DIR = Path("queue")


def write_queue(tag: str, items: List[dict], kind: str = "training") -> Path:
    QUEUE_DIR.mkdir(exist_ok=True)
    path = QUEUE_DIR / f"{tag}.json"
    with open(path, "w", encoding="utf-8") as file:
        json.dump({"tag": tag, "kind": kind, "written_at": datetime.now().isoformat(timespec="seconds"),
                   "items": items}, file, indent=1)
    return path
