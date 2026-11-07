"""Gemma writes the blocks (the redesign of 2026-11-07: the student learns Gemma's own code, not the real block),
in two passes with a check in between:

  1. write (rented card): Gemma reads the domain's prompt (GameDslDomain.prompt: card, manual, ...) + the short
     input + the command and writes its reply (greedy, text only, no logits) -> rows {"id", "prompt", "reply"}
  2. check (at home, the app's compiler is a Windows program): the block from the reply is put into the game at the
     cursor (/insert: where the real block was cut out; /edit: in place of the block) and the whole game goes
     through check.check_sources; a block is kept when it adds no problem the game did not have (hollow-game
     signs are reported but do not drop it). Also compared with the real block: the same tree (keywords,
     arguments, nesting) and the same shape (names, numbers and colours ignored).
  3. score (rented card): the kept blocks, teacher-forced, with the same prompt -> the teacher data (corpus.py).
"""

import json
import re
from collections import Counter
from typing import Dict, List, Optional

from . import FENCE
from .check import check_sources
from .language import Node, parse_fragment
from .problems import OUTPUT

VERSIONS = OUTPUT / "versions.jsonl"
HOLLOW = "hollow: "


def block_of(reply: str) -> Optional[str]:
    """The code inside the reply's first ``` block (a language tag after the opening fence is skipped), or None
    when there is no complete block."""
    start = reply.find(FENCE)
    if start < 0:
        return None
    body_start = reply.find("\n", start)
    end = reply.find(FENCE, body_start + 1) if body_start >= 0 else -1
    if end < 0:
        return None
    block = reply[body_start + 1:end].strip("\n")
    return block if block.strip() else None


def spliced_game(source: str, problem: Dict, block: str) -> str:
    """The version's game with `block` in place of the real block (the problem's "cut")."""
    start, end = problem["cut"]
    return source[:start] + block.rstrip("\n") + "\n" + source[end:].lstrip("\n")


def tree(nodes: List[Node], keep_values: bool):
    return tuple((node.key, tuple(node.args) if keep_values else tuple(type(arg).__name__ for arg in node.args),
                  tree(node.children, keep_values)) for node in nodes)


def compare(block: str, target: str) -> Dict[str, bool]:
    try:
        written, real = parse_fragment(block), parse_fragment(target)
    except ValueError:
        return {"same_tree": False, "same_shape": False}
    return {"same_tree": tree(written, True) == tree(real, True),
            "same_shape": tree(written, False) == tree(real, False)}


def load_versions(version_ids: set) -> Dict[str, str]:
    sources = {}
    with open(VERSIONS, encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            if record["id"] in version_ids:
                sources[record["id"]] = record["source"]
    return sources


def version_of(problem_id: str) -> str:
    return problem_id.rsplit("|", 1)[0]


def check_written(rows: List[Dict], problems: Dict[str, Dict]) -> List[Dict]:
    """rows {"id", "reply", ...} -> the same rows with "block", "kept", "new_problems", "hollow", "same_tree",
    "same_shape". One compiler run for all of them."""
    sources = load_versions({version_of(row["id"]) for row in rows})
    baseline = {name: Counter(found) for name, found in check_sources(sources).items()}
    games, checked = {}, []
    for index, row in enumerate(rows):
        problem = problems[row["id"]]
        block = block_of(row["reply"])
        checked.append({**row, "block": block, "task": problem["task"], "kind": problem["kind"]})
        if block is not None:
            games[str(index)] = spliced_game(sources[version_of(row["id"])], problem, block)
    found = check_sources(games) if games else {}
    for index, row in enumerate(checked):
        if row["block"] is None:
            row.update({"kept": False, "new_problems": ["no complete ``` block in the reply"], "hollow": [],
                        "same_tree": False, "same_shape": False})
            continue
        new = Counter(found[str(index)]) - baseline[version_of(row["id"])]
        problems_added = sorted(new.elements())
        row["hollow"] = [item for item in problems_added if item.startswith(HOLLOW)]
        row["new_problems"] = [item for item in problems_added if not item.startswith(HOLLOW)]
        row["kept"] = not row["new_problems"]
        row.update(compare(row["block"], problems[row["id"]]["target"]))
    return checked


def summary(checked: List[Dict]) -> Dict:
    count = len(checked)
    share = lambda key: round(sum(bool(row[key]) for row in checked) / max(count, 1), 3)
    return {"blocks": count, "complete": round(sum(row["block"] is not None for row in checked) / max(count, 1), 3),
            "kept": share("kept"), "kept_no_hollow": round(sum(row["kept"] and not row["hollow"] for row in checked)
                                                             / max(count, 1), 3),
            "same_tree": share("same_tree"), "same_shape": share("same_shape"),
            "problems": Counter(re.sub(r"[:(].*", "", item) for row in checked for item in row["new_problems"])
            .most_common(6)}
