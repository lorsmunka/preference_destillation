"""The tutor agent = a model + this harness. The harness owns everything the model should not:
the conversation state, the context rules, checking each answer and asking again, and replying
to off-topic messages itself.

Parts, each replaceable on its own:
  TutorModel     anything with generate(requests) -> texts. TeacherModel (library/agent/runner.py)
                 runs the teacher (to try the harness before any student exists); the distilled
                 student plugs in later.
  Conversation   the context rules (conversation.py): typed history, newest input <message>,
                 a follow-up task closes the side thread of follow-up questions.
  checker        a function block -> problem or None (checks.check_lines: well formed, every
                 calculation true). A failed solution is asked for once more.
  explanation_checker  the same for an explanation (checks.check_explanation: it must not take
                 a checked solution back, nor agree with pushback). Asked for once more.
  pupil          TemplatePupil plays an item: its problem, then its follow-up tasks, with
                 follow-up messages in between from the template bank (questions.py), filled
                 in from the tutor's own last solution.

One agent turn, for a new pupil message m:
  1. route        context = history + open thread + <message>m</message>; the model's first tag
  2a. solution    history (thread dropped) + <message>m</message>, the model continues after
                  "<solution>"; checked, asked again once if the check fails
  2b. explanation the same context as the route, the model continues after "<explanation>"
  2c. out_of_scope the harness replies with OUT_OF_SCOPE_REPLY; the model writes nothing more
                  and the message never enters the model's context

Sessions are small state machines with one pending request each, so run_sessions can batch the
requests of many conversations into one model call.

    python -m library.domain.math_tutor.harness      # Gemma (NF4) on 16 test items; GPU
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

from library.agent.question_templates import Message, QuestionGenerator, related
from library.agent.runner import Request, TeacherModel, run_sessions
from .checks import check_explanation, check_lines, complete_results, final_answer
from .conversation import Conversation, parse_route
from .questions import kind_weights, solution_foci, tasks_of, what_if_foci

PACKAGE = Path(__file__).resolve().parent
RUNS_DIR = PACKAGE / "harness_runs"
SOLUTION_PREFIX, EXPLANATION_PREFIX = "<solution>\n", "<explanation>"
ROUTE_PREFIX = "<"   # a route asked again starts the tag, so the reply cannot open with prose or bullets
OUT_OF_SCOPE_REPLY = "This question is outside the scope of this task."
# characters a route reply may have before its tag (user, 2026-10-21): Gemma 4 sometimes writes "1. ", "100%"
# or a leaked "<|channel>thought<channel|>" first; dropping those cost 5.1% of the slice's replies
JUNK_BEFORE_TAG = 40


class TemplatePupil:
    """Plays one item: the problem, each follow-up task in order, and after some solutions one to
    three follow-up messages from the template bank about the tutor's own solution (each after
    the first related to the one before). On the last task a message may be a what-if, whose new
    answer the item's code gives."""

    def __init__(self, item: Dict, generator: QuestionGenerator, question_chance: float = 0.7, seed: int = 0):
        self.item, self.generator = item, generator
        self.random_generator = random.Random(f"{seed}-{item['id']}")
        self.events = []
        for turn, task in enumerate(tasks_of(item)):
            self.events.append({"type": "task", "turn": turn, "text": task, "answer": item["answers"][turn]})
            if self.random_generator.random() < question_chance:
                for depth in range(self.random_generator.choice([1, 2, 2, 3])):
                    self.events.append({"type": "message", "turn": turn, "depth": depth})
        self.previous_focus = None

    def message(self, event: Dict, block: str, depth: Optional[int] = None) -> Optional[Message]:
        """The next message, drawn at `depth`: the explanations in the open thread (the session
        passes it, so after an off-topic message or a misrouted question the pupil does not
        follow up an explanation that is not there); the planned position if not given."""
        turn = event["turn"]
        depth = event["depth"] if depth is None else depth
        foci = solution_foci(self.item, turn, block, self.random_generator)
        if turn == len(self.item["follow_ups"]):
            foci += what_if_foci(self.item, turn, block, self.random_generator)
        candidates = related(foci, self.previous_focus) if depth > 0 else foci
        message = self.generator.draw(candidates, depth, self.random_generator, kind_weights=kind_weights(depth))
        if message is not None and message.kind != "off_topic":
            self.previous_focus = message.focus
        return message


class TutorSession:
    """One conversation: the pupil's events, driven through the agent turn above."""

    def __init__(self, pupil: TemplatePupil, checker: Callable[[str], Optional[str]] = check_lines,
                 explanation_checker: Callable[[str, Optional[str]], Optional[str]] = check_explanation, retries: int = 1):
        self.pupil, self.checker, self.explanation_checker, self.retries = pupil, checker, explanation_checker, retries
        self.conversation = Conversation()
        self.transcript: List[Dict] = []
        self.event_index = 0
        self.last_block = ""
        self.closed_turns = set()   # turns after which a what-if was solved: their remaining messages are skipped
        self.turn: Optional[Dict] = None
        self.pending: Optional[Request] = None
        self.advance()

    @property
    def done(self) -> bool:
        return self.pending is None

    def advance(self) -> None:
        """Start the next pupil event (or finish)."""
        while self.event_index < len(self.pupil.events):
            event = self.pupil.events[self.event_index]
            self.event_index += 1
            if event["type"] == "task":
                self.turn = {"event": "task", "pupil_turn": event["turn"], "attempts": 0, "message": event["text"],
                             "expected_route": "solution", "expected_answer": event["answer"]}
                self.pending = Request("route", self.conversation.route_context(event["text"]), max_tokens=24)
                return
            if event["turn"] in self.closed_turns:
                continue
            message = self.pupil.message(event, self.last_block, depth=len(self.conversation.thread))
            if message is None:
                continue
            self.turn = {"event": message.kind if message.kind in ("off_topic", "what_if") else "question",
                         "pupil_turn": event["turn"], "attempts": 0, "message": message.text, "kind": message.kind,
                         "expected_route": message.route, "template": message.template.text, "focus": message.focus.type}
            if message.kind == "what_if":
                self.turn["expected_answer"] = message.focus.data["answer"]
            self.pending = Request("route", self.conversation.route_context(message.text), max_tokens=24)
            return
        self.pending = None

    def receive(self, text: str) -> None:
        request, turn = self.pending, self.turn
        if request.kind == "route":
            route_text = request.prefix + text
            if parse_route(route_text) == "unclear" and not request.prefix:   # no tag (e.g. a bulleted reply): ask once more, starting the tag
                turn["route_retry"] = route_text
                self.pending = Request("route", request.context, prefix=ROUTE_PREFIX, max_tokens=24)
                return
            turn["route_text"] = route_text
            turn["route"] = parse_route(route_text)
            turn["route_correct"] = turn["route"] == turn["expected_route"]
            if turn["route"] == "out_of_scope":   # the harness answers; the model's context stays as it was
                turn.update(output=OUT_OF_SCOPE_REPLY, answered_by="harness")
                self.finish_turn()
            elif turn["route"] == "explanation":
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
            self.conversation.add_solution(turn["message"], stored.replace("<solution>", "").replace("</solution>", "").strip())
            self.last_block = stored
            if turn["event"] == "what_if":
                self.closed_turns.add(turn["pupil_turn"])
            self.finish_turn()
        elif request.kind == "explanation":
            turn["attempts"] += 1
            explanation = text.replace("</explanation>", "").strip()
            if "</explanation>" not in text:   # cut off at the token limit
                problem = "cut off: no </explanation>"
            else:
                problem = self.explanation_checker(explanation, turn.get("kind"))
            if problem and turn["attempts"] <= self.retries:
                turn.setdefault("rejected", []).append({"output": EXPLANATION_PREFIX + text, "problem": problem})
                self.pending = Request("explanation", request.context, prefix=EXPLANATION_PREFIX,
                                       stop=["</explanation>"], temperature=0.7)
                return
            turn.update(output=EXPLANATION_PREFIX + text, check=problem)
            self.conversation.add_explanation(turn["message"], explanation)
            self.finish_turn()

    def finish_turn(self) -> None:
        self.transcript.append(self.turn)
        self.advance()


def run_conversation_pass(handler, domain, items: List[Dict], path: Path, chunk: int = 32, batch: int = 4,
                          seed: int = 0) -> bool:
    """Gemma as the tutor in every item's simulated conversation (the template pupil), the source of
    the training data: every reply in it is Gemma's own. Items are played in a seeded random order,
    `chunk` conversations at a time (`batch` replies per model call), each chunk appended to `path`
    when it ends. Graceful stop (Stop button / Ctrl+C): the round in progress finishes, and every
    conversation of the chunk is saved as far as it got ("complete": false for an unfinished one;
    its replies are still training data). Items already in the file are skipped, so a stopped pass
    resumes with the next ones. False when stopped."""
    from library.shared import ExitListener
    from .questions import generator as question_generator

    done = set()
    if path.exists():
        done = {json.loads(line)["item_id"] for line in open(path, encoding="utf-8")}
    order = list(items)
    random.Random(seed).shuffle(order)
    todo = [item for item in order if item["id"] not in done]
    questions, tutor = question_generator(), TeacherModel(handler, domain, batch=batch)
    exit_listener = ExitListener()
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"{len(done)} conversations saved, {len(todo)} to go", flush=True)
    saved = len(done)
    for start in range(0, len(todo), chunk):
        sessions = [TutorSession(TemplatePupil(item, questions)) for item in todo[start:start + chunk]]
        finished = run_sessions(tutor, sessions, exit_listener, progress=True)
        with open(path, "a", encoding="utf-8") as out:
            for session in sessions:
                group = {"group": session.pupil.item["group"]} if "group" in session.pupil.item else {}
                out.write(json.dumps({"item_id": session.pupil.item["id"], "split": session.pupil.item["split"], **group,
                                      "complete": session.done, "turns": session.transcript}, ensure_ascii=False) + "\n")
        saved += len(sessions)
        summary = summarise(sessions)
        print(f"  {saved} conversations saved ({sum(not session.done for session in sessions)} of this chunk unfinished); "
              f"answers correct {summary['answers_correct']}/{summary['tasks']}, explanations pass "
              f"{summary['explanations_pass_check']}/{summary['question_routed_explanation']}", flush=True)
        if not finished:
            print("Stopped: rerun to continue with the next conversations.", flush=True)
            return False
    return True


def parsable(turn: Dict) -> bool:
    """Whether the harness could read the reply: the route's tag within the first JUNK_BEFORE_TAG
    characters (the record's target is the clean reply either way: a solution or an explanation is
    written after its forced tag), and for a solution or an explanation its closing tag (and a
    solution's Answer line)."""
    route, output = turn.get("route"), turn.get("output", "")
    if not 0 <= turn.get("route_text", "").find(f"<{route}>") <= JUNK_BEFORE_TAG:
        return False
    if route == "solution":
        return output.endswith("</solution>") and final_answer(output) is not None
    return route == "out_of_scope" or output.endswith("</explanation>")


def transcript_records(item: Dict, transcript: List[Dict]) -> List[Dict]:
    """Training records from one conversation of the pass above: each reply Gemma wrote, in the
    context it was written in, mistakes included (it is the teacher's behaviour that is distilled;
    each record carries route_correct / check / answer_correct, so a correct-only subset stays
    possible). A conversation ends at the first reply the harness could not parse (parsable). A
    solution is learned in the clean history (the side thread dropped); when a thread was open,
    the route to it is a record of its own. An off-topic route gives a route record
    (<out_of_scope>) and the message never enters the context, as in the harness. `answer` is the
    code's answer, which task accuracy is measured against."""
    conversation = Conversation()
    base = {"item_id": item["id"], "family": item["family"], "split": item["split"], "slice": item["slice"],
            "problem_shape": item["problem_shape"], "teacher": "forced"}
    records = []
    for turn in transcript:
        if not parsable(turn):
            break
        route, message = turn["route"], turn["message"]
        context = {**base, "turn": turn["pupil_turn"], "event": turn["event"], "expected_route": turn["expected_route"],
                   "route_correct": turn["route_correct"], "check": turn.get("check")}
        if route == "out_of_scope":
            records.append({**context, "text": conversation.route_context(message), "target": f"<{route}>", "turn_type": "route"})
            continue
        if route == "explanation":
            records.append({**context, "text": conversation.explain_context(message), "target": turn["output"],
                            "turn_type": "explain", "depth": len(conversation.thread) + 1})
            conversation.add_explanation(message, turn["output"][len(EXPLANATION_PREFIX):].replace("</explanation>", "").strip())
            continue
        if conversation.thread:
            records.append({**context, "text": conversation.route_context(message), "target": "<solution>", "turn_type": "route"})
        records.append({**context, "text": conversation.solve_context(message), "target": turn["output"], "turn_type": "solve",
                        "answer": turn.get("expected_answer"), "answer_correct": turn.get("answer_correct"),   # none for a misrouted question
                        "shape": item["shape"].split("|")[turn["pupil_turn"]]})
        stored = turn.get("stored_completed", turn["output"])
        conversation.add_solution(message, stored.replace("<solution>", "").replace("</solution>", "").strip())
    return records


def summarise(sessions: List[TutorSession], tokenizer=None, allowed_tokens=None) -> Dict:
    turns = [turn for session in sessions for turn in session.transcript]
    by_event = defaultdict(list)
    for turn in turns:
        by_event[turn["event"]].append(turn)
    tasks, questions = by_event["task"], by_event["question"]
    what_ifs, off_topic = by_event["what_if"], by_event["off_topic"]
    routes_by_kind = defaultdict(Counter)
    for turn in turns:
        routes_by_kind[turn.get("kind", "task")][turn.get("route")] += 1
    summary = {
        "conversations": len(sessions), "turns": len(turns), "tasks": len(tasks), "questions": len(questions),
        "what_ifs": len(what_ifs), "off_topic": len(off_topic),
        "task_routed_solution": sum(turn.get("route") == "solution" for turn in tasks),
        "question_routed_explanation": sum(turn.get("route") == "explanation" for turn in questions),
        "what_if_routed_solution": sum(turn.get("route") == "solution" for turn in what_ifs),
        "off_topic_routed_out_of_scope": sum(turn.get("route") == "out_of_scope" for turn in off_topic),
        "solutions_pass_check": sum(turn.get("check") is None for turn in tasks if "check" in turn),
        "solutions_needed_retry": sum(bool(turn.get("rejected")) for turn in tasks),
        "explanations_pass_check": sum(turn.get("check") is None for turn in questions if turn.get("route") == "explanation"),
        "explanations_needed_retry": sum(bool(turn.get("rejected")) for turn in questions),
        "answers_correct": sum(bool(turn.get("answer_correct")) for turn in tasks),
        "what_if_answers_correct": sum(bool(turn.get("answer_correct")) for turn in what_ifs),
        "conversations_all_correct": sum(all(t.get("answer_correct") for t in s.transcript if t["event"] == "task") for s in sessions),
        "routes_by_kind": {kind: dict(counter) for kind, counter in sorted(routes_by_kind.items())},
    }
    if tokenizer is not None and allowed_tokens is not None:
        outside = Counter()
        for turn in turns:
            if turn.get("answered_by") != "harness":
                outside.update(token for token in tokenizer.tokenize(turn.get("output", "")) if token not in allowed_tokens)
        summary["tokens_outside_vocabulary"] = dict(outside.most_common(40))
    return summary


def next_run_number() -> int:
    numbers = [int(path.stem.rsplit("_", 1)[1]) for path in RUNS_DIR.glob("gemma_pilot_*.json") if path.stem.rsplit("_", 1)[1].isdigit()]
    return max(numbers, default=0) + 1


def main() -> None:
    from transformers import AutoTokenizer
    from library.data_gen.model_handler import ModelHandler
    from .agentic import MathTutorAgenticDomain
    from .questions import generator

    items = [json.loads(line) for line in open(PACKAGE / "items" / "items.jsonl", encoding="utf-8")]
    random_generator = random.Random(0)
    test_items = [item for item in items if item["split"] == "test"]
    chosen = random_generator.sample([i for i in test_items if len(i["follow_ups"]) >= 2], 12) + \
             random_generator.sample([i for i in test_items if len(i["follow_ups"]) == 1], 4)
    domain = MathTutorAgenticDomain("facts")
    handler = ModelHandler("google/gemma-4-12B-it", domain, load_in_4bit=True, chat_template=True)
    questions = generator()
    sessions = [TutorSession(TemplatePupil(item, questions)) for item in chosen]
    run_sessions(TeacherModel(handler, domain), sessions)
    tokenizer = AutoTokenizer.from_pretrained("models/gemma-4-12B-it")
    summary = summarise(sessions, tokenizer, set(domain.example_tokens) | {"\n\n", "▁▁"})
    RUNS_DIR.mkdir(exist_ok=True)
    number = next_run_number()
    record = {"model": "google/gemma-4-12B-it NF4 (teacher as tutor)", "annotation": "facts", "pilot": number,
              "pupil": f"template bank, {len(questions.templates)} templates", "summary": summary,
              "conversations": [{"item_id": s.pupil.item["id"], "slice": s.pupil.item["slice"], "turns": s.transcript} for s in sessions]}
    (RUNS_DIR / f"gemma_pilot_{number}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    page = PACKAGE.parents[2] / "thesis_work" / f"harness_transcripts_{number}.html"
    page.write_text(transcript_page(record), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"transcripts: {page}")


# --- a local page to read the conversations ------------------------------------------------

TRANSCRIPT_STYLE = """
:root { --bg:#f7f7f5; --card:#fff; --fg:#1d1d1f; --muted:#6b6b70; --border:#e2e2de; --task:#e2f3e7; --task-fg:#1e6a39;
  --q:#e3ecfb; --q-fg:#2a5bb0; --sol:#efe3fb; --sol-fg:#6527ad; --exp:#fdedcf; --exp-fg:#875800; --ok:#2e8b57; --bad:#c0392b; --warn:#b7791f;
  --off:#ececea; --off-fg:#55555a; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#141416; --card:#1d1d20; --fg:#ececee; --muted:#9a9aa2;
  --border:#2e2e33; --task:#173322; --task-fg:#7fd8a0; --q:#1d2b47; --q-fg:#8db4ff; --sol:#2e1f45; --sol-fg:#c9a3ff; --exp:#3a2c0e;
  --exp-fg:#f0c060; --ok:#5cc28a; --bad:#ff7b6b; --warn:#e3b25a; --off:#2a2a2e; --off-fg:#c4c4ca; } }
:root[data-theme="dark"] { --bg:#141416; --card:#1d1d20; --fg:#ececee; --muted:#9a9aa2; --border:#2e2e33; --task:#173322; --task-fg:#7fd8a0;
  --q:#1d2b47; --q-fg:#8db4ff; --sol:#2e1f45; --sol-fg:#c9a3ff; --exp:#3a2c0e; --exp-fg:#f0c060; --ok:#5cc28a; --bad:#ff7b6b; --warn:#e3b25a;
  --off:#2a2a2e; --off-fg:#c4c4ca; }
* { box-sizing:border-box; } body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width:900px; margin:0 auto; padding:24px 16px 60px; } h1 { font-size:24px; margin:0 0 6px; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:8px; margin:14px 0; }
.tile { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:8px 10px; } .tile b { display:block; font-size:20px; }
.tile span, .note { color:var(--muted); font-size:13px; }
.conv { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:10px 14px; margin:12px 0; }
.conv > summary { cursor:pointer; } .turn { margin:12px 0; }
.pupil { border-radius:10px; padding:8px 10px; } .pupil.task, .pupil.what_if { background:var(--task); color:var(--task-fg); }
.pupil.question { background:var(--q); color:var(--q-fg); } .pupil.off_topic { background:var(--off); color:var(--off-fg); }
.who { display:block; font-size:11px; text-transform:uppercase; letter-spacing:.04em; opacity:.8; }
.meta { margin:4px 0; display:flex; gap:6px; flex-wrap:wrap; }
.badge, .chip { font-size:12px; padding:1px 7px; border-radius:6px; border:1px solid var(--border); }
.badge.ok { color:var(--ok); } .badge.bad { color:var(--bad); font-weight:600; } .badge.warn { color:var(--warn); } .chip { color:var(--muted); }
pre { white-space:pre-wrap; word-break:break-word; font:12.5px/1.5 ui-monospace,Consolas,monospace; margin:4px 0 0; border-radius:8px; padding:8px 10px; }
pre.solution { background:var(--sol); color:var(--sol-fg); } pre.explanation { background:var(--exp); color:var(--exp-fg); }
pre.harness { background:var(--off); color:var(--off-fg); font-style:italic; }
.rejected summary { font-size:12px; color:var(--warn); cursor:pointer; } .rejected pre { background:var(--bg); border:1px dashed var(--border); }
"""


def badge(ok, good_text: str, bad_text: str) -> str:
    import html
    return f'<span class="badge {"ok" if ok else "bad"}">{html.escape(good_text if ok else bad_text)}</span>'


def transcript_page(record: Dict) -> str:
    """Every conversation of a harness run: pupil messages (green task or what-if, blue question,
    grey off-topic), the replies (purple solution, orange explanation, grey: the harness itself),
    route / check / answer badges."""
    import html
    summary = record["summary"]
    tiles = [("conversations", summary["conversations"]),
             ("tasks routed to a solution", f"{summary['task_routed_solution']}/{summary['tasks']}"),
             ("questions routed to an explanation", f"{summary['question_routed_explanation']}/{summary['questions']}"),
             ("what-ifs routed to a solution", f"{summary['what_if_routed_solution']}/{summary['what_ifs']}"),
             ("off-topic routed out of scope", f"{summary['off_topic_routed_out_of_scope']}/{summary['off_topic']}"),
             ("solutions pass the check", f"{summary['solutions_pass_check']}/{summary['tasks']}"),
             ("explanations pass the check", f"{summary['explanations_pass_check']}/{summary['question_routed_explanation']}"),
             ("answers correct", f"{summary['answers_correct']}/{summary['tasks']}"),
             ("what-if answers correct", f"{summary['what_if_answers_correct']}/{summary['what_ifs']}")]
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
            if turn.get("kind"):
                meta.append(f'<span class="chip">{html.escape(turn["kind"])} / {html.escape(turn.get("focus", ""))}</span>')
            if "answer_correct" in turn:
                meta.append(badge(turn.get("check") is None, "check passed", f"check: {turn.get('check')}"))
                meta.append(badge(turn.get("answer_correct"), "answer correct",
                                  f"answer {turn.get('answer')} (expected {turn.get('expected_answer')})"))
            elif turn.get("check") or turn.get("rejected"):
                meta.append(badge(turn.get("check") is None, "second try passed", f"check: {turn.get('check')}"))
            if turn.get("rejected") and "answer_correct" in turn:
                meta.append('<span class="badge warn">second try</span>')
            rejected = "".join(f'<details class="rejected"><summary>first try, rejected: {html.escape(r["problem"])}</summary>'
                               f'<pre>{html.escape(r["output"])}</pre></details>' for r in turn.get("rejected", []))
            output = turn.get("output", "")
            reply = "harness" if turn.get("answered_by") == "harness" else ("solution" if output.startswith("<solution>") else "explanation")
            label = {"task": "task", "question": "follow-up question", "what_if": "what if (a new task)", "off_topic": "off-topic"}[turn["event"]]
            rows.append(f'<div class="turn"><div class="pupil {turn["event"]}"><span class="who">pupil / {label}</span>'
                        f'{html.escape(turn.get("message", ""))}</div><div class="meta">{"".join(meta)}</div>{rejected}'
                        f'<pre class="reply {reply}">{html.escape(output)}</pre></div>')
        parts.append(f'<details class="conv" open><summary><b>{html.escape(conversation["item_id"])}</b> / {conversation["slice"]}</summary>'
                     + "".join(rows) + "</details>")
    intro = ("The harness with the model below as the tutor. The pupil is scripted: the item's problem and follow-up tasks, and "
             "follow-up messages from the template bank filled in from the tutor's own solutions. Grey replies are written by "
             "the harness itself (off-topic). Earlier turns hold the model's own answers, uncorrected.")
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>Tutor Transcripts</title><style>{TRANSCRIPT_STYLE}</style></head><body><main>'
            f'<h1>Tutor transcripts: {html.escape(record["model"])}</h1><p class="note">{html.escape(intro)}</p>'
            + "".join(parts) + "</main></body></html>")


if __name__ == "__main__":
    main()
