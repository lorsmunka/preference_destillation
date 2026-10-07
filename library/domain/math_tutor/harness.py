"""The tutor agent = a model + this harness. The harness owns everything the model should not:
the conversation state, the context rules, checking each answer and asking again.

Parts, each replaceable on its own:
  TutorModel     anything with generate(requests) -> texts. GemmaTutor runs the teacher (to try
                 the harness before any student exists); the distilled student plugs in later.
  Conversation   the context rules (conversation.py): typed history, newest input <message>,
                 a follow-up task closes the side thread of follow-up questions.
  checker        a function block -> problem or None (checks.check_lines: well formed, every
                 calculation true). A failed solution is asked for once more.
  pupil          ScriptedPupil plays an item: its problem, then its follow-up tasks, with
                 follow-up questions in between (written by Gemma playing a pupil).

One agent turn, for a new pupil message m:
  1. route     context = history + open thread + <message>m</message>; the model's opening tag
  2a. solution history (thread dropped) + <message>m</message>, the model continues after
               "<solution>"; checked, asked again once if the check fails
  2b. explain  the same context as the route, the model continues after "<explanation>"

Sessions are small state machines with one pending request each, so run_sessions can batch the
requests of many conversations into one model call.

    python -m library.domain.math_tutor.harness      # Gemma (NF4) on 16 test items; GPU
"""

import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .checks import check_lines, complete_results
from .conversation import Conversation
from .threads import OPENINGS, PERSONAS, PUPIL_PROMPT, QUESTION_KINDS

PACKAGE = Path(__file__).resolve().parent
RUNS_DIR = PACKAGE / "harness_runs"
SOLUTION_PREFIX, EXPLANATION_PREFIX = "<solution>\n", "<explanation>"


@dataclass
class Request:
    kind: str                 # "route" | "solution" | "explanation" | "pupil"
    context: str              # the conversation text (pupil: the whole pupil prompt)
    prefix: str = ""          # what the model's turn starts with
    max_tokens: int = 120
    temperature: float = 0.0
    stop: List[str] = field(default_factory=list)


class GemmaTutor:
    """The teacher as the tutor model: the agentic domain's prompt around the conversation.
    At most `batch` requests per model call (8 keeps a 12 GB card off its memory limit)."""

    def __init__(self, handler, domain, batch: int = 8):
        self.handler, self.domain, self.batch = handler, domain, batch

    def generate(self, requests: List[Request]) -> List[str]:
        groups: Dict[tuple, List[int]] = {}
        for index, request in enumerate(requests):
            groups.setdefault((request.max_tokens, request.temperature, tuple(request.stop)), []).append(index)
        texts = [""] * len(requests)
        chunks = [(key, indices[start:start + self.batch]) for key, indices in groups.items()
                  for start in range(0, len(indices), self.batch)]
        for (max_tokens, temperature, stop), indices in chunks:
            prompts = [requests[i].context if requests[i].kind == "pupil" else self.domain.teacher_prompt(requests[i].context)
                       for i in indices]
            outputs = self.handler.generate_texts(prompts, max_new_tokens=max_tokens, temperature=temperature,
                                                  stop_strings=list(stop) or None, prefixes=[requests[i].prefix for i in indices])
            for i, output in zip(indices, outputs):
                texts[i] = output
        return texts


def parse_route(text: str) -> str:
    """The route from the first tag the model wrote (anything before it, such as an empty
    thought preamble, is ignored)."""
    positions = {route: text.find(f"<{route}") for route in ("solution", "explanation")}
    found = {route: position for route, position in positions.items() if position >= 0}
    return min(found, key=found.get) if found else "unclear"


def final_answer(block: str) -> Optional[int]:
    match = re.search(r"Answer:\s*(\d+)", block)
    return int(match.group(1)) if match else None


class ScriptedPupil:
    """Plays one item: the problem, each follow-up task in order, and after some solutions one
    or two follow-up questions (kind and persona drawn per question, seeded per item)."""

    def __init__(self, item: Dict, question_chance: float = 0.6, seed: int = 0):
        self.item = item
        random_generator = random.Random(f"{seed}-{item['id']}")
        tasks = [item["problem"]] + [follow_up["question"] for follow_up in item["follow_ups"]]
        self.events = []
        for turn, task in enumerate(tasks):
            self.events.append({"type": "task", "turn": turn, "text": task, "answer": item["answers"][turn]})
            if random_generator.random() < question_chance:
                for _ in range(random_generator.choice([1, 1, 2])):
                    self.events.append({"type": "question", "turn": turn,
                                        "kind": random_generator.choice(list(QUESTION_KINDS)),
                                        "persona": random_generator.choice(list(PERSONAS)),
                                        "opening": random_generator.choice(OPENINGS)})


class TutorSession:
    """One conversation: the scripted pupil's events, driven through the agent turn above."""

    def __init__(self, pupil: ScriptedPupil, checker: Callable[[str], Optional[str]] = check_lines, retries: int = 1):
        self.pupil, self.checker, self.retries = pupil, checker, retries
        self.conversation = Conversation()
        self.transcript: List[Dict] = []
        self.event_index = 0
        self.turn: Optional[Dict] = None
        self.pending: Optional[Request] = None
        self.advance()

    @property
    def done(self) -> bool:
        return self.pending is None

    def advance(self) -> None:
        """Start the next pupil event (or finish)."""
        if self.event_index >= len(self.pupil.events):
            self.pending = None
            return
        event = self.pupil.events[self.event_index]
        self.event_index += 1
        self.turn = {"event": event["type"], "pupil_turn": event["turn"], "attempts": 0}
        if event["type"] == "task":
            self.turn.update(message=event["text"], expected_route="solution", expected_answer=event["answer"])
            self.pending = Request("route", self.conversation.route_context(event["text"]), max_tokens=24)
        else:
            self.turn.update(expected_route="explanation", kind=event["kind"], persona=event["persona"])
            about = "the tutor's last explanation" if self.conversation.thread else "the tutor's last solution"
            prompt = PUPIL_PROMPT.format(persona=PERSONAS[event["persona"]], about=about, kind=QUESTION_KINDS[event["kind"]],
                                         opening=event["opening"], conversation=self.conversation.history(with_thread=True))
            self.pending = Request("pupil", prompt, max_tokens=60, temperature=0.9)

    def receive(self, text: str) -> None:
        request, turn = self.pending, self.turn
        if request.kind == "pupil":
            turn["message"] = text.strip().strip('"').split("\n")[0]
            self.pending = Request("route", self.conversation.route_context(turn["message"]), max_tokens=24)
        elif request.kind == "route":
            turn["route_text"] = text
            turn["route"] = parse_route(text)
            turn["route_correct"] = turn["route"] == turn["expected_route"]
            if turn["route"] == "explanation":
                self.pending = Request("explanation", self.conversation.explain_context(turn["message"]),
                                       prefix=EXPLANATION_PREFIX, stop=["</explanation>"])
            else:   # a solution, or unclear: the harness treats it as a task (the safe side)
                self.pending = Request("solution", self.conversation.solve_context(turn["message"]),
                                       prefix=SOLUTION_PREFIX, stop=["</solution>"])
        elif request.kind == "solution":
            turn["attempts"] += 1
            block = SOLUTION_PREFIX + text
            problem = self.checker(block)
            if problem and turn["attempts"] <= self.retries:
                turn.setdefault("rejected", []).append({"output": block, "problem": problem})
                self.pending = Request("solution", request.context, prefix=SOLUTION_PREFIX, stop=["</solution>"], temperature=0.7)
                return
            turn.update(output=block, check=problem, answer=final_answer(block))
            if turn.get("expected_answer") is not None:
                turn["answer_correct"] = turn["answer"] == turn["expected_answer"]
            stored = complete_results(block)   # a missing "= result" is filled in before it enters the history
            if stored != block:
                turn["stored_completed"] = stored
            body = stored.replace("<solution>", "").replace("</solution>", "").strip()
            self.conversation.add_solution(turn["message"], body)
            self.finish_turn()
        elif request.kind == "explanation":
            explanation = text.replace("</explanation>", "").strip()
            turn.update(output=EXPLANATION_PREFIX + text)
            self.conversation.add_explanation(turn["message"], explanation)
            self.finish_turn()

    def finish_turn(self) -> None:
        self.transcript.append(self.turn)
        self.advance()


def run_sessions(model, sessions: List[TutorSession]) -> None:
    """Drive every session to the end, batching each round's pending requests."""
    while True:
        active = [session for session in sessions if not session.done]
        if not active:
            return
        outputs = model.generate([session.pending for session in active])
        for session, output in zip(active, outputs):
            session.receive(output)


def summarise(sessions: List[TutorSession], tokenizer=None, allowed_tokens=None) -> Dict:
    turns = [turn for session in sessions for turn in session.transcript]
    tasks = [turn for turn in turns if turn["event"] == "task"]
    questions = [turn for turn in turns if turn["event"] == "question"]
    summary = {
        "conversations": len(sessions), "turns": len(turns), "tasks": len(tasks), "questions": len(questions),
        "task_routed_solution": sum(turn.get("route") == "solution" for turn in tasks),
        "question_routed_explanation": sum(turn.get("route") == "explanation" for turn in questions),
        "solutions_pass_check": sum(turn.get("check") is None for turn in tasks if "check" in turn),
        "solutions_needed_retry": sum(bool(turn.get("rejected")) for turn in tasks),
        "answers_correct": sum(bool(turn.get("answer_correct")) for turn in tasks),
        "conversations_all_correct": sum(all(t.get("answer_correct") for t in s.transcript if t["event"] == "task") for s in sessions),
        "question_kinds": dict(Counter(turn["kind"] for turn in questions)),
    }
    if tokenizer is not None and allowed_tokens is not None:
        outside = Counter()
        for turn in turns:
            outside.update(token for token in tokenizer.tokenize(turn.get("output", "")) if token not in allowed_tokens)
        summary["tokens_outside_vocabulary"] = dict(outside.most_common(40))
    return summary


def main() -> None:
    from transformers import AutoTokenizer
    from library.data_gen.model_handler import ModelHandler
    from .agentic import MathTutorAgenticDomain

    items = [json.loads(line) for line in open(PACKAGE / "items" / "items.jsonl", encoding="utf-8")]
    random_generator = random.Random(0)
    test_items = [item for item in items if item["split"] == "test"]
    chosen = random_generator.sample([i for i in test_items if len(i["follow_ups"]) >= 2], 12) + \
             random_generator.sample([i for i in test_items if len(i["follow_ups"]) == 1], 4)
    domain = MathTutorAgenticDomain("facts")
    handler = ModelHandler("google/gemma-4-12B-it", domain, load_in_4bit=True, chat_template=True)
    sessions = [TutorSession(ScriptedPupil(item)) for item in chosen]
    run_sessions(GemmaTutor(handler, domain), sessions)
    tokenizer = AutoTokenizer.from_pretrained("models/gemma-4-12B-it")
    summary = summarise(sessions, tokenizer, set(domain.example_tokens) | {"\n\n", "▁▁"})
    RUNS_DIR.mkdir(exist_ok=True)
    record = {"model": "google/gemma-4-12B-it NF4 (teacher as tutor)", "annotation": "facts", "summary": summary,
              "conversations": [{"item_id": s.pupil.item["id"], "slice": s.pupil.item["slice"], "turns": s.transcript} for s in sessions]}
    (RUNS_DIR / "gemma_pilot.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    page = PACKAGE.parents[2] / "thesis_work" / "harness_transcripts.html"
    page.write_text(transcript_page(record), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"transcripts: {page}")


# --- a local page to read the conversations ------------------------------------------------

TRANSCRIPT_STYLE = """
:root { --bg:#f7f7f5; --card:#fff; --fg:#1d1d1f; --muted:#6b6b70; --border:#e2e2de; --task:#e2f3e7; --task-fg:#1e6a39;
  --q:#e3ecfb; --q-fg:#2a5bb0; --sol:#efe3fb; --sol-fg:#6527ad; --exp:#fdedcf; --exp-fg:#875800; --ok:#2e8b57; --bad:#c0392b; --warn:#b7791f; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#141416; --card:#1d1d20; --fg:#ececee; --muted:#9a9aa2;
  --border:#2e2e33; --task:#173322; --task-fg:#7fd8a0; --q:#1d2b47; --q-fg:#8db4ff; --sol:#2e1f45; --sol-fg:#c9a3ff; --exp:#3a2c0e;
  --exp-fg:#f0c060; --ok:#5cc28a; --bad:#ff7b6b; --warn:#e3b25a; } }
:root[data-theme="dark"] { --bg:#141416; --card:#1d1d20; --fg:#ececee; --muted:#9a9aa2; --border:#2e2e33; --task:#173322; --task-fg:#7fd8a0;
  --q:#1d2b47; --q-fg:#8db4ff; --sol:#2e1f45; --sol-fg:#c9a3ff; --exp:#3a2c0e; --exp-fg:#f0c060; --ok:#5cc28a; --bad:#ff7b6b; --warn:#e3b25a; }
* { box-sizing:border-box; } body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width:900px; margin:0 auto; padding:24px 16px 60px; } h1 { font-size:24px; margin:0 0 6px; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:8px; margin:14px 0; }
.tile { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:8px 10px; } .tile b { display:block; font-size:20px; }
.tile span, .note { color:var(--muted); font-size:13px; }
.conv { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:10px 14px; margin:12px 0; }
.conv > summary { cursor:pointer; } .turn { margin:12px 0; }
.pupil { border-radius:10px; padding:8px 10px; } .pupil.task { background:var(--task); color:var(--task-fg); }
.pupil.question { background:var(--q); color:var(--q-fg); }
.who { display:block; font-size:11px; text-transform:uppercase; letter-spacing:.04em; opacity:.8; }
.meta { margin:4px 0; display:flex; gap:6px; flex-wrap:wrap; }
.badge, .chip { font-size:12px; padding:1px 7px; border-radius:6px; border:1px solid var(--border); }
.badge.ok { color:var(--ok); } .badge.bad { color:var(--bad); font-weight:600; } .badge.warn { color:var(--warn); } .chip { color:var(--muted); }
pre { white-space:pre-wrap; word-break:break-word; font:12.5px/1.5 ui-monospace,Consolas,monospace; margin:4px 0 0; border-radius:8px; padding:8px 10px; }
pre.solution { background:var(--sol); color:var(--sol-fg); } pre.explanation { background:var(--exp); color:var(--exp-fg); }
.rejected summary { font-size:12px; color:var(--warn); cursor:pointer; } .rejected pre { background:var(--bg); border:1px dashed var(--border); }
"""


def badge(ok, good_text: str, bad_text: str) -> str:
    import html
    return f'<span class="badge {"ok" if ok else "bad"}">{html.escape(good_text if ok else bad_text)}</span>'


def transcript_page(record: Dict) -> str:
    """Every conversation of a harness run: pupil messages (green task, blue question), the
    model's replies (purple solution, orange explanation), route / check / answer badges."""
    import html
    summary = record["summary"]
    tiles = [("conversations", summary["conversations"]),
             ("tasks routed to a solution", f"{summary['task_routed_solution']}/{summary['tasks']}"),
             ("questions routed to an explanation", f"{summary['question_routed_explanation']}/{summary['questions']}"),
             ("solutions pass the check", f"{summary['solutions_pass_check']}/{summary['tasks']}"),
             ("needed a second try", summary["solutions_needed_retry"]),
             ("answers correct", f"{summary['answers_correct']}/{summary['tasks']}"),
             ("conversations fully correct", f"{summary['conversations_all_correct']}/{summary['conversations']}")]
    parts = ['<div class="tiles">' + "".join(f'<div class="tile"><b>{html.escape(str(value))}</b><span>{html.escape(name)}</span></div>'
                                             for name, value in tiles) + "</div>"]
    outside = summary.get("tokens_outside_vocabulary") or {}
    listed = ", ".join(f"{token} x{count}" for token, count in outside.items()) or "none"
    parts.append(f'<p class="note">Tokens the model used that are outside the output vocabulary: {html.escape(listed)}</p>')
    for conversation in record["conversations"]:
        rows = []
        for turn in conversation["turns"]:
            meta = [badge(turn.get("route_correct"), f"routed {turn.get('route')}",
                          f"routed {turn.get('route')} (expected {turn['expected_route']})")]
            if turn["event"] == "task":
                meta.append(badge(turn.get("check") is None, "check passed", f"check: {turn.get('check')}"))
                meta.append(badge(turn.get("answer_correct"), "answer correct",
                                  f"answer {turn.get('answer')} (expected {turn.get('expected_answer')})"))
                if turn.get("rejected"):
                    meta.append('<span class="badge warn">second try</span>')
            else:
                meta.append(f'<span class="chip">{html.escape(turn["kind"])} / {html.escape(turn["persona"])}</span>')
            kind = "task" if turn["event"] == "task" else "question"
            rejected = "".join(f'<details class="rejected"><summary>first try, rejected: {html.escape(r["problem"])}</summary>'
                               f'<pre>{html.escape(r["output"])}</pre></details>' for r in turn.get("rejected", []))
            output = turn.get("output", "")
            reply = "solution" if output.startswith("<solution>") else "explanation"
            rows.append(f'<div class="turn"><div class="pupil {kind}"><span class="who">pupil / follow-up {kind}</span>'
                        f'{html.escape(turn.get("message", ""))}</div><div class="meta">{"".join(meta)}</div>{rejected}'
                        f'<pre class="reply {reply}">{html.escape(output)}</pre></div>')
        parts.append(f'<details class="conv" open><summary><b>{html.escape(conversation["item_id"])}</b> / {conversation["slice"]}</summary>'
                     + "".join(rows) + "</details>")
    intro = ("The harness with the model below as the tutor. The pupil is scripted: the item's problem and follow-up tasks, and "
             "follow-up questions written by Gemma playing a pupil. Earlier turns hold the model's own answers, uncorrected.")
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>Tutor Transcripts</title><style>{TRANSCRIPT_STYLE}</style></head><body><main>'
            f'<h1>Tutor transcripts: {html.escape(record["model"])}</h1><p class="note">{html.escape(intro)}</p>'
            + "".join(parts) + "</main></body></html>")


if __name__ == "__main__":
    main()
