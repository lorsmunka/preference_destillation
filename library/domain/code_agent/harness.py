"""The coding agent = a model + this harness (decided 2026-10-20). One request, two steps:

  1. scope    the model reads the whole project (every file, numbered lines) and the user's
              message, and writes the lines that matter: <scope> file.py 12-30 ... </scope>
              checked (the files exist, the lines are inside them), asked again once
  2. answer   the harness narrows the context to those lines; the model decides: <explanation>
              ... </explanation> when the user asked for one, <found/> when they only asked where
              something is (the scope is then the answer). An explanation is checked (finished,
              names only things the project has), asked again once.

The model is anything with generate(requests) -> texts (library/agent/runner.py): the teacher
for the training data, the distilled student later.

The pass below is the source of the training data: Gemma does both steps of every item, and each
of its replies, in the context it was written in, becomes a record (mistakes included, flagged).

    python -m library.domain.code_agent.harness                    # GPU: Gemma on every item not played yet -> conversations/v1.jsonl
    python -m library.domain.code_agent.harness v1 - 8             # the same, 8 requests per model call (a bigger card)
    python -m library.domain.code_agent.harness v1 - 16 32         # 16 per call, 32 items at a time (default 16)
    python -m library.domain.code_agent.harness pilot_nf4 16 1     # a local trial: 16 train + 8 test items, one at a time
"""

import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

from library.agent.runner import Request, TeacherModel, run_sessions, teacher_setup, trial_items
from .checks import check_explanation, check_scope, score_scope, usable_ranges
from .project import Version, known_names, load_base, make_version, project_context, scoped_context, symbol_lines

PACKAGE = Path(__file__).resolve().parent
SCOPE_PREFIX = "<scope>\n"
ANSWER_TAGS = ("<explanation>", "<found/>")
JUNK_BEFORE_TAG = 40   # characters a reply may have before its tag (Gemma 4 sometimes leaks "<channel|>")
LEAKED_MARKER = re.compile(r"<\|?channel\|?>")   # Gemma 4's channel markers, also seen inside a scope (2026-10-21)
SCOPE_TOKENS, ANSWER_TOKENS = 120, 400


class CodeAgentSession:
    """One item: the two steps above, with one retry each."""

    def __init__(self, item: Dict, version: Version, retries: int = 1):
        self.item, self.version, self.retries = item, version, retries
        self.names = known_names(version.files)
        self.turn = {"message": item["message"], "kind": item["kind"], "scope_attempts": 0, "answer_attempts": 0}
        self.pending: Optional[Request] = self.scope_request()

    @property
    def done(self) -> bool:
        return self.pending is None

    def scope_request(self, temperature: float = 0.0) -> Request:
        return Request("scope", project_context(self.version, self.item["message"]), prefix=SCOPE_PREFIX,
                       stop=["</scope>"], max_tokens=SCOPE_TOKENS, temperature=temperature)

    def receive(self, text: str) -> None:
        turn = self.turn
        text = LEAKED_MARKER.sub("", text)
        if self.pending.kind == "scope":
            turn["scope_attempts"] += 1
            block = SCOPE_PREFIX + text
            problem = check_scope(block, self.version.files)
            if problem and turn["scope_attempts"] <= self.retries:
                turn.setdefault("rejected", []).append({"step": "scope", "output": block, "problem": problem})
                self.pending = self.scope_request(temperature=0.7)
                return
            ranges = usable_ranges(block, self.version.files)
            turn.update(scope=block, scope_check=problem, ranges={name: spans for name, spans in ranges.items()})
            if not ranges:   # nothing to show: the request ends here
                self.pending = None
                return
            self.pending = Request("answer", scoped_context(self.version, ranges, self.item["message"]),
                                   stop=["</explanation>", "<found/>"], max_tokens=ANSWER_TOKENS)
            return
        turn["answer_attempts"] += 1
        output = from_first_tag(text)
        route = "explain" if output.startswith("<explanation>") else "find" if output.startswith("<found/>") else "unclear"
        problem = None
        if route == "unclear":
            problem = "no <explanation> or <found/>"
        elif route == "explain":
            problem = check_explanation(output, self.names)
        if problem and turn["answer_attempts"] <= self.retries:
            turn.setdefault("rejected", []).append({"step": "answer", "output": output, "problem": problem})
            self.pending = Request("answer", self.pending.context, stop=["</explanation>", "<found/>"],
                                   max_tokens=ANSWER_TOKENS, temperature=0.7)
            return
        turn.update(output=output, route=route, route_correct=route == self.item["kind"], check=problem)
        self.pending = None


def from_first_tag(text: str) -> str:
    """The reply from its first <explanation> or <found/> on, when only a little comes before it
    (the reply is free: a forced "<" made Gemma write "<10.555 is refused..." or "<int>"); the
    reply as it is otherwise."""
    text = text.strip()
    starts = [text.find(tag) for tag in ANSWER_TAGS if 0 <= text.find(tag) <= JUNK_BEFORE_TAG]
    return text[min(starts):] if starts else text


def version_cache():
    """make_version, remembered per (project, number): the items of one version share it."""
    bases, versions = {}, {}

    def get(project: str, number: int) -> Version:
        if (project, number) not in versions:
            if project not in bases:
                bases[project] = load_base(project)
            versions[(project, number)] = make_version(bases[project], number)
        return versions[(project, number)]
    return get


def scored_turn(item: Dict, version: Version, turn: Dict) -> Dict:
    """The turn with its scope scored against the true places."""
    if not turn.get("ranges"):
        return turn
    places = symbol_lines(version.files)
    return {**turn, **score_scope(turn["ranges"], [tuple(place) for place in item["truth"]], places)}


def run_pass(handler, domain, items: List[Dict], path: Path, chunk: int = 16, batch: int = 2, seed: int = 0) -> bool:
    """Gemma as the agent on every item not in `path` yet, in a seeded random order, `chunk` items
    at a time (`batch` requests per model call: the whole project is ~6,000 tokens, so 2 on a
    12 GB card, more on a bigger one); each chunk appended when it ends. False when stopped
    (Stop button / Ctrl+C: the round in progress finishes, the chunk is saved as far as it got)."""
    from library.shared import ExitListener

    done = set()
    if path.exists():
        done = {json.loads(line)["item_id"] for line in open(path, encoding="utf-8")}
    order = list(items)
    random.Random(seed).shuffle(order)
    todo = [item for item in order if item["id"] not in done]
    model, versions, exit_listener = TeacherModel(handler, domain, batch=batch), version_cache(), ExitListener()
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"{len(done)} items played, {len(todo)} to go", flush=True)
    saved = len(done)
    for start in range(0, len(todo), chunk):
        sessions = [CodeAgentSession(item, versions(item["project"], item["version"])) for item in todo[start:start + chunk]]
        finished = run_sessions(model, sessions, exit_listener, progress=True)
        with open(path, "a", encoding="utf-8") as out:
            for session in sessions:
                out.write(json.dumps({"item_id": session.item["id"], "split": session.item["split"], "complete": session.done,
                                      "turn": scored_turn(session.item, session.version, session.turn)}, ensure_ascii=False) + "\n")
        saved += len(sessions)
        turns = [scored_turn(session.item, session.version, session.turn) for session in sessions]
        scoped = [turn for turn in turns if "recall" in turn]
        print(f"  {saved} items saved; scopes {len(scoped)}/{len(turns)}, true lines covered "
              f"{sum(t['recall'] for t in scoped) / max(1, len(scoped)):.0%}, places hit {sum(t['places_hit'] for t in scoped)}/"
              f"{sum(t['places'] for t in scoped)}; decision right {sum(bool(t.get('route_correct')) for t in turns)}/{len(turns)}",
              flush=True)
        if not finished:
            print("Stopped: rerun to continue with the next items.", flush=True)
            return False
    return True


def transcript_records(item: Dict, version: Version, turn: Dict) -> List[Dict]:
    """Training records from one played item: the scope Gemma wrote for the whole project, and its
    answer in the context the harness narrowed to that scope. Mistakes stay in, flagged (scope
    recall / precision against the true places, the decision, the checks); a reply the harness
    could not use ends the item (no answer record without a usable scope)."""
    base = {"item_id": item["id"], "project": item["project"], "split": item["split"], "kind": item["kind"],
            "source": item["source"], "teacher": "forced", "truth": item["truth"]}
    records = []
    if not turn.get("ranges"):
        return records
    scored = scored_turn(item, version, turn)
    records.append({**base, "text": project_context(version, item["message"]), "target": turn["scope"], "turn_type": "scope",
                    "scope_check": turn.get("scope_check"), "recall": scored["recall"], "precision": scored["precision"],
                    "places_hit": scored["places_hit"], "places": scored["places"]})
    output = turn.get("output", "")
    if output == "<found/>" or (output.startswith("<explanation>") and output.endswith("</explanation>")):
        ranges = {name: [tuple(span) for span in spans] for name, spans in turn["ranges"].items()}
        records.append({**base, "text": scoped_context(version, ranges, item["message"]), "target": turn["output"],
                        "turn_type": turn["route"], "route_correct": turn["route_correct"], "check": turn.get("check")})
    return records


def main() -> None:
    from library.data_gen.model_handler import ModelHandler
    from library.domain import CODE_RECALL
    from . import CodeAgentDomain
    from .requests import read_items

    # Plain generation never uses the handler's own reduced vocabulary, and the coding agent's is
    # built from these very replies: the handler gets a domain whose vocabulary always exists.
    variant = sys.argv[1] if len(sys.argv) > 1 else "v1"
    count = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != "-" else None
    batch = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    chunk = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    domain = CodeAgentDomain(variant)
    items = read_items() if count is None else trial_items(read_items(), count)
    load_in_4bit = teacher_setup(domain.conversations_path)   # full precision on a card with room for it
    handler = ModelHandler("google/gemma-4-12B-it", CODE_RECALL, load_in_4bit=load_in_4bit, chat_template=True)
    run_pass(handler, domain, items, domain.conversations_path, chunk=chunk, batch=batch)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
