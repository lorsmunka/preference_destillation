"""The two agents' harnesses, step by step, as one local HTML page that looks like a chat app: click
the message box to write the next (pre-written) message, send it, and watch the harness build every
context the model reads, the model's reply, the checks and the next step.

    python harness_chat.py      # writes thesis_work/harness_chat.html

Local file only, nothing is uploaded. The conversations are real: Gemma 4 12B's saved replies from
the passes (the math slice; the coding agent's pilot2_nf4 trial), replayed through the real sessions
(TutorSession, CodeAgentSession) with a model that gives back those replies, so every context on the
page is the one the harness built.
"""

import json
import re
import sys
from pathlib import Path

from library.agent.question_templates import Focus, Message, Template
from library.agent.runner import run_sessions

ROOT = Path(__file__).resolve().parent
PAGE = ROOT / "thesis_work" / "harness_chat.html"
MATH_CONVERSATIONS = ROOT / "library" / "domain" / "math_tutor" / "items" / "conversations" / "slice_v1.jsonl"
CODE_CONVERSATIONS = ROOT / "library" / "domain" / "code_agent" / "conversations" / "pilot2_nf4.jsonl"
MATH_CHATS = [("S05-mem-b05-0084", "Hot chocolate at camp"), ("sd-p2-h04-0005~w1", "Tents and lanterns")]
CODE_CHATS = [("Shop, a Python project", ["shop~v35#2", "shop~v35#3"]), ("Travel site, HTML pages", ["travel_site~v43#1"])]
MATH_BLOCK = re.compile(r"<(task|solution|question|explanation|message)>(.*?)</\1>", re.DOTALL)
FILE_BLOCK = re.compile(r'<file name="([^"]+)">\n(.*?)\n</file>', re.DOTALL)
NUMBERED_LINE = re.compile(r"^\s*(\d+):(?: (.*))?$")


class ReplayModel:
    """Gives back saved replies in order and keeps every request it was asked."""

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def generate(self, requests):
        texts = [self.replies.pop(0) for _ in requests]
        self.calls += zip(requests, texts)
        return texts


class RecordedPupil:
    """The pupil of a saved conversation: its messages in their order (TutorSession reads only
    `events` and `message()`)."""

    def __init__(self, transcript):
        self.events = []
        for turn in transcript:
            if turn["event"] == "task":
                self.events.append({"type": "task", "turn": turn["pupil_turn"], "text": turn["message"],
                                    "answer": turn["expected_answer"]})
            else:
                self.events.append({"type": "message", "turn": turn["pupil_turn"], "depth": 0, "saved": turn})

    def message(self, event, block, depth=None):
        turn = event["saved"]
        focus = Focus(turn.get("focus", "none"), data={"answer": turn.get("expected_answer")})
        template = Template(turn["kind"], turn["expected_route"], turn.get("focus", "none"), turn.get("template", turn["message"]))
        return Message(turn["message"], turn["kind"], turn["expected_route"], focus, template)


def read_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


# --- the math tutor -------------------------------------------------------------------------------

def math_replies(transcript):
    """Gemma's replies in the order the session asks for them."""
    from library.domain.math_tutor.harness import EXPLANATION_PREFIX, ROUTE_PREFIX, SOLUTION_PREFIX
    replies = []
    for turn in transcript:
        if "route_retry" in turn:
            replies += [turn["route_retry"], turn["route_text"][len(ROUTE_PREFIX):]]
        else:
            replies.append(turn["route_text"])
        if turn["route"] == "out_of_scope":
            continue
        prefix = EXPLANATION_PREFIX if turn["route"] == "explanation" else SOLUTION_PREFIX
        replies += [attempt["output"][len(prefix):] for attempt in turn.get("rejected", []) + [turn]]
    return replies


def math_context(context):
    blocks = [{"tag": match.group(1), "text": match.group(0)} for match in MATH_BLOCK.finditer(context)]
    assert "\n".join(block["text"] for block in blocks) == context, "a math context did not split into blocks"
    return {"blocks": blocks}


def request_event(request, context, note, stage):
    return {"type": "request", "kind": request.kind, "context": context, "prefix": request.prefix, "stop": request.stop,
            "maxTokens": request.max_tokens, "temperature": request.temperature, "chars": len(request.context),
            "note": note, "stage": stage}


def harness(status, title, text, stage):
    """A step the harness takes (status: route, pass, fail, info); `stage` lights up the page's pipeline."""
    return {"type": "harness", "status": status, "title": title, "text": text, "stage": stage}


def route_reply(prefix, reply):
    """A route reply with the length of its part the harness uses: up to the end of its first tag."""
    from library.domain.math_tutor.conversation import ROUTES
    text = prefix + reply
    starts = [text.find(f"<{route}") for route in ROUTES if text.find(f"<{route}") >= 0]
    end = text.find(">", min(starts)) + 1 if starts else len(text)
    return {"type": "reply", "text": reply, "keep": max(0, (end or len(text)) - len(prefix))}


def math_turn(turn, calls):
    """The steps of one agent turn, from the session's own requests and the saved replies."""
    route_note = ("Route call: the history, the open side thread and the new message. "
                  "Gemma's first tag decides what happens next (at most 24 tokens).")
    events = []
    if "route_retry" in turn:
        request, reply = next(calls)
        events += [request_event(request, math_context(request.context), route_note, "route"), route_reply("", reply),
                   harness("fail", "No tag in the reply", "Asked once more, with the reply started as \"<\".", "route")]
    request, reply = next(calls)
    route_blocks = math_context(request.context)
    events += [request_event(request, route_blocks, route_note if "route_retry" not in turn else "The same context, asked again.",
                             "route"), route_reply(request.prefix, reply)]
    route = turn["route"]
    wrong = "" if turn["route_correct"] else f" (the pupil's message called for {turn['expected_route']}: a wrong route, kept as data)"
    if route == "out_of_scope":
        events += [harness("route", "Route: out of scope" + wrong,
                           "Off topic. The harness replies with a fixed sentence, and the message never enters the model's context.",
                           "oos"),
                   {"type": "drop_message"},
                   {"type": "assistant", "format": "harness", "text": turn["output"]}]
        return events
    attempts = turn.get("rejected", []) + [turn]
    if route == "explanation":
        events.append(harness("route", "Route: explanation" + wrong,
                              "A question about the working. The explanation is asked for in the same context, "
                              "with Gemma's reply started as <explanation>.", "explanation"))
    else:
        events.append(harness("route", "Route: solution" + wrong,
                              "A task. Only the tag is used; the rest of the route reply is thrown away. "
                              "The solution is asked for in its own context.", "solution"))
    for index, attempt in enumerate(attempts):
        request, reply = next(calls)
        context = math_context(request.context)
        dropped = len([block for block in route_blocks["blocks"] if block not in context["blocks"]])
        note = {"solution": "Solution call: the history without the side thread, and the task.",
                "explanation": "Explanation call: the same context as the route call."}[request.kind]
        if index:
            note = "The same context, asked again: this time sampled (temperature 0.7)."
        elif dropped:
            note += f" The side thread ({dropped} blocks) is dropped: a new task closes it."
        events += [request_event(request, context, note, request.kind), {"type": "reply", "text": reply}]
        problem = attempt["problem"] if attempt is not turn else turn.get("check")
        if problem and attempt is not turn:
            events.append(harness("fail", "Check failed", f"{problem}. Asked again once.", "check"))
        elif problem:
            events.append(harness("fail", "Check failed again", f"{problem}. The harness keeps the reply, flagged.", "check"))
        elif route == "explanation":
            events.append(harness("pass", "Check passed", "Finished, and it does not take the checked solution back"
                                  + (" or agree with the pupil's pushback." if turn.get("kind") == "pushback" else "."), "check"))
        else:
            events.append(harness("pass", "Check passed", "Every line is a fact or a true calculation, "
                                  "and it ends with the Answer line.", "check"))
    if route == "explanation":
        events += [harness("info", "Added to the side thread", "Questions after it see it; the next task drops it.", "store"),
                   {"type": "assistant", "format": "explanation",
                    "text": turn["output"].replace("<explanation>", "").replace("</explanation>", "").strip()}]
        return events
    filled = " A calculation without its result was completed first." if "stored_completed" in turn else ""
    events += [harness("info", "Stored in the history", "The task and its solution: every later call sees them." + filled, "store"),
               {"type": "assistant", "format": "solution", "text": turn["output"]}]
    if turn.get("expected_answer") is not None:
        right = turn.get("answer_correct")
        events.append({"type": "grade", "ok": bool(right),
                       "text": f"Graded against the item's code (not part of the harness): {turn['expected_answer']}, "
                               f"{'right' if right else 'wrong'}."})
    return events


def math_chat(item_id, title):
    from library.domain.math_tutor.agentic import MathTutorAgenticDomain
    from library.domain.math_tutor.harness import TutorSession
    saved = next(conversation for conversation in read_jsonl(MATH_CONVERSATIONS) if conversation["item_id"] == item_id)
    model = ReplayModel(math_replies(saved["turns"]))
    session = TutorSession(RecordedPupil(saved["turns"]))
    run_sessions(model, [session])
    keys = ("message", "route", "route_text", "output", "check", "answer_correct")
    assert len(session.transcript) == len(saved["turns"]) and not model.replies, f"{item_id}: the replay went another way"
    for replayed, original in zip(session.transcript, saved["turns"]):
        assert all(replayed.get(key) == original.get(key) for key in keys), f"{item_id}: turn differs: {replayed['message']}"
    calls = iter(model.calls)
    turns = [{"user": turn["message"], "events": math_turn(turn, calls)} for turn in session.transcript]
    return {"id": item_id, "agent": "math", "group": "Math tutor", "title": title,
            "subtitle": "Gemma 4 12B, 4-bit · slice pass",
            "detail": f"Gemma 4 12B (NF4) as the tutor in the slice pass: {saved['split']} item {item_id}",
            "instructions": MathTutorAgenticDomain("facts").teacher_prompt("").strip(), "turns": turns}


# --- the coding agent -----------------------------------------------------------------------------

def code_context(context):
    """{wrapper, files: [{name, lines: [[number, text]]}], message}, checked by rendering it back."""
    from library.domain.code_agent.project import render
    wrapper = "project" if context.startswith("<project>") else "context"
    files = [{"name": name, "lines": [[int(match.group(1)), match.group(2) or ""]
                                       for match in map(NUMBERED_LINE.match, body.split("\n"))]}
             for name, body in FILE_BLOCK.findall(context)]
    message = context.split("<message>")[-1].split("</message>")[0]
    shown = {file["name"]: [line[1] for line in file["lines"]] for file in files}
    if wrapper == "project":
        rebuilt = f"<project>\n{render(shown)}\n</project>\n<message>{message}</message>"
    else:   # render() numbers from 1: rebuild the narrowed lines by hand
        rebuilt = "<context>\n" + "\n".join(
            f'<file name="{file["name"]}">\n' + "\n".join(f"{number:>2}: {text}".rstrip() for number, text in file["lines"])
            + "\n</file>" for file in files) + f"\n</context>\n<message>{message}</message>"
    assert rebuilt == context, "a coding-agent context did not parse back"
    return {"wrapper": wrapper, "files": files, "message": message}


def code_turn(item, saved_turn, version, first):
    from library.domain.code_agent.harness import CodeAgentSession, SCOPE_PREFIX, scored_turn
    rejected = saved_turn.get("rejected", [])
    replies = ([attempt["output"][len(SCOPE_PREFIX):] for attempt in rejected if attempt["step"] == "scope"]
               + [saved_turn["scope"][len(SCOPE_PREFIX):]]
               + [attempt["output"] for attempt in rejected if attempt["step"] == "answer"] + [saved_turn["output"]])
    model = ReplayModel(replies)
    session = CodeAgentSession(item, version)
    run_sessions(model, [session])
    turn = session.turn
    for key in ("scope", "output", "route", "check", "scope_check"):
        assert turn.get(key) == saved_turn.get(key), f"{item['id']}: the replay differs at {key}"
    events = [] if first else [harness("info", "A new request", "The harness starts again from the whole project: "
                                       "nothing is kept between requests.", "scope")]
    scope_attempts = [attempt for attempt in rejected if attempt["step"] == "scope"]
    calls = iter(model.calls)
    for index in range(len(scope_attempts) + 1):
        request, reply = next(calls)
        note = ("Scope call: the whole project, every file with its lines numbered, and the user's message. "
                "Gemma's reply is started as <scope>.") if not index else "The same context, asked again: sampled (temperature 0.7)."
        events += [request_event(request, code_context(request.context), note, "scope"), {"type": "reply", "text": reply}]
        if index < len(scope_attempts):
            events.append(harness("fail", "Scope check failed", f"{scope_attempts[index]['problem']}. Asked again once.", "check1"))
    if turn.get("scope_check"):
        events.append(harness("fail", "Scope check failed again", f"{turn['scope_check']}. The harness uses the ranges that "
                                                                    "point into the project.", "check1"))
    else:
        events.append(harness("pass", "Scope check passed", "The files exist and the lines are inside them.", "check1"))
    events += [{"type": "narrow", "ranges": turn["ranges"]},
               harness("info", "Context narrowed", "Only the scoped lines are kept, with the files' own line numbers.", "narrow")]
    answer_attempts = [attempt for attempt in rejected if attempt["step"] == "answer"]
    for index in range(len(answer_attempts) + 1):
        request, reply = next(calls)
        note = ("Answer call: only the scoped lines and the message. Gemma decides: <explanation> when the user asked "
                "for one, <found/> when they only asked where something is.") if not index else \
            "The same context, asked again: sampled (temperature 0.7)."
        events += [request_event(request, code_context(request.context), note, "answer"), {"type": "reply", "text": reply}]
        if index < len(answer_attempts):
            events.append(harness("fail", "Check failed", f"{answer_attempts[index]['problem']}. Asked again once.", "check2"))
    wrong = "" if turn["route_correct"] else f" (the request called for {item['kind']}: a wrong decision, kept as data)"
    if turn["route"] == "find":
        events.append(harness("route", "Decision: found" + wrong, "The user only asked where it is: the scope is the answer.",
                              "answer"))
    else:
        events.append(harness("route", "Decision: explanation" + wrong, "The user asked for one.", "answer"))
        events.append(harness("fail", "Check failed again", f"{turn['check']}. Kept, flagged.", "check2") if turn.get("check") else
                      harness("pass", "Check passed", "Finished, and it names only things the project has.", "check2"))
    scoped = code_context(model.calls[-1][0].context)
    events.append({"type": "assistant", "format": "found" if turn["route"] == "find" else "code-explanation",
                   "text": turn["output"].replace("<explanation>", "").replace("</explanation>", "").strip(),
                   "files": scoped["files"]})
    score = scored_turn(item, version, turn)
    truth = ", ".join(f"{file_name}:{name}" for file_name, name in item["truth"])
    events.append({"type": "grade", "ok": score["places_hit"] == score["places"],
                   "text": f"Scored against the true place ({truth}), not part of the harness: true lines covered "
                           f"{score['recall']:.0%}, scope on target {score['precision']:.0%}, places hit "
                           f"{score['places_hit']}/{score['places']}."})
    return {"user": item["message"], "events": events}


def code_chat(title, item_ids):
    from library.domain.code_agent import CodeAgentDomain
    from library.domain.code_agent.harness import version_cache
    from library.domain.code_agent.requests import read_items
    saved = {entry["item_id"]: entry["turn"] for entry in read_jsonl(CODE_CONVERSATIONS)}
    items = {item["id"]: item for item in read_items()}
    versions = version_cache()
    first = items[item_ids[0]]
    version = versions(first["project"], first["version"])
    turns = [code_turn(items[item_id], saved[item_id], versions(items[item_id]["project"], items[item_id]["version"]),
                       index == 0) for index, item_id in enumerate(item_ids)]
    renames = [f"{old} → {new}" for old, new in sorted(version.renames.items())]
    return {"id": version.id, "agent": "code", "group": "Coding agent", "title": title,
            "subtitle": "Gemma 4 12B, 4-bit · trial pass",
            "detail": f"Gemma 4 12B (NF4) as the agent in the pilot2_nf4 trial: {first['split']} project {version.id}",
            "instructions": CodeAgentDomain().teacher_prompt("").strip(),
            "project": {"name": first["project"], "version": first["version"], "files": list(version.files),
                        "lines": sum(len(lines) for lines in version.files.values()), "renames": renames},
            "turns": turns}


def build():
    chats = [math_chat(item_id, title) for item_id, title in MATH_CHATS] + [code_chat(title, ids) for title, ids in CODE_CHATS]
    return {"chats": chats}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    data = build()
    for chat in data["chats"]:
        steps = sum(len(turn["events"]) for turn in chat["turns"])
        print(f"{chat['group']:<13} {chat['title']:<26} {len(chat['turns'])} messages, {steps} steps")
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    PAGE.write_text(TEMPLATE.replace("__DARK__", DARK).replace("__DATA__", payload), encoding="utf-8")
    print(f"-> {PAGE} ({PAGE.stat().st_size / 1000:.0f} KB)")


DARK = """
    --bg: #212121; --sidebar: #171717; --panel: #1b1b1b; --text: #ececec; --muted: #b4b4b4; --faint: #8b8b8b;
    --bubble: #303030; --border: #2f2f2f; --composer: #303030; --composer-border: #3a3a3a; --hover: #2a2a2a; --active: #343434;
    --code-bg: #151515; --send: #ececec; --send-text: #0d0d0d; --kind-text: #0d0d0d;
    --task: #60a5fa; --solution: #34d399; --question: #fbbf24; --explanation: #a78bfa; --message: #f472b6; --file: #22d3ee; --wrap: #8b8b8b;
    --pass: #34d399; --fail: #f87171; --route: #60a5fa; --info: #9ca3af;
    --enter: rgba(52, 211, 153, .16); --leave: rgba(248, 113, 113, .16); --scope: rgba(250, 204, 21, .18);
    --shadow: 0 0 0 1px rgba(255, 255, 255, .03), 0 10px 30px rgba(0, 0, 0, .45);
    color-scheme: dark;
"""

TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Harness Chat</title>
<style>
:root {
  --bg: #ffffff; --sidebar: #f9f9f9; --panel: #fafafa; --text: #0d0d0d; --muted: #5d5d5d; --faint: #8f8f8f;
  --bubble: #f1f1f1; --border: #e6e6e6; --composer: #ffffff; --composer-border: #dcdcdc; --hover: #ececec; --active: #e3e3e3;
  --code-bg: #f5f5f6; --accent: #10a37f; --send: #0d0d0d; --send-text: #ffffff; --kind-text: #ffffff;
  --task: #2563eb; --solution: #047857; --question: #b45309; --explanation: #7c3aed; --message: #be185d; --file: #0e7490; --wrap: #6b7280;
  --pass: #047857; --fail: #dc2626; --route: #2563eb; --info: #6b7280;
  --enter: rgba(16, 163, 127, .18); --leave: rgba(220, 38, 38, .14); --scope: rgba(234, 179, 8, .28);
  --shadow: 0 0 0 1px rgba(0, 0, 0, .03), 0 8px 26px rgba(0, 0, 0, .08);
  --mono: ui-monospace, "SF Mono", "Cascadia Code", Menlo, Consolas, monospace;
  --k: 1;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { __DARK__ } }
:root[data-theme="dark"] { __DARK__ }
* { box-sizing: border-box; }
html, body { height: 100%; margin: 0; }
body { background: var(--bg); color: var(--text); overflow: hidden; -webkit-font-smoothing: antialiased;
  font: 15px/1.6 ui-sans-serif, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
body.fast { --k: 1000; }
button { font: inherit; color: inherit; }
code { font-family: var(--mono); font-size: .9em; }
.app { display: grid; grid-template-columns: 260px minmax(0, 1fr) 0; grid-template-rows: minmax(0, 1fr); height: 100vh; height: 100dvh; }
.app.panel-open { grid-template-columns: 260px minmax(0, 1fr) minmax(360px, 40vw); }

/* sidebar */
.sidebar { background: var(--sidebar); display: flex; flex-direction: column; padding: 10px 8px; overflow-y: auto; min-width: 0; min-height: 0; }
.side-top { display: flex; align-items: center; justify-content: space-between; padding: 2px 4px 8px 8px; }
.brand { font-weight: 600; display: flex; align-items: center; gap: 9px; }
.logo { width: 26px; height: 26px; border-radius: 50%; background: var(--text); color: var(--bg); display: grid; place-items: center; font-size: 14px; }
.icon-btn { width: 36px; height: 36px; border-radius: 8px; border: 0; background: transparent; display: grid; place-items: center; cursor: pointer; color: var(--muted); flex: none; }
.icon-btn:hover { background: var(--hover); color: var(--text); }
.side-group { margin: 16px 0 4px; padding: 0 10px; font-size: 12px; color: var(--faint); font-weight: 500; }
.side-item { display: block; width: 100%; text-align: left; border: 0; background: transparent; padding: 7px 10px; border-radius: 8px; cursor: pointer; font-size: 14px; }
.side-item:hover { background: var(--hover); }
.side-item.active { background: var(--active); }
.side-item small { display: block; color: var(--faint); font-size: 12px; line-height: 1.4; }
.side-foot { margin-top: auto; padding: 12px 2px 2px; }
.side-foot p { font-size: 12px; color: var(--faint); margin: 8px 10px 0; line-height: 1.5; }

/* main column */
.main { display: flex; flex-direction: column; min-width: 0; min-height: 0; height: 100%; }
.topbar { height: 56px; display: flex; align-items: center; gap: 8px; padding: 0 12px 0 16px; flex: none; }
.title { font-size: 17px; font-weight: 600; white-space: nowrap; display: flex; gap: 6px; }
.title .light { color: var(--faint); font-weight: 400; }
.model-pill { font-size: 12px; color: var(--muted); border: 1px solid var(--border); border-radius: 999px; padding: 2px 10px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; min-width: 0; }
.spacer { flex: 1; }
.seg { display: flex; background: var(--hover); border-radius: 999px; padding: 2px; flex: none; }
.seg button { border: 0; background: transparent; border-radius: 999px; padding: 3px 9px; font-size: 12px; cursor: pointer; color: var(--muted); }
.seg button.on { background: var(--bg); color: var(--text); box-shadow: var(--shadow); }
.pill-btn { border: 1px solid var(--border); background: transparent; border-radius: 999px; padding: 6px 12px; font-size: 13px; cursor: pointer;
  display: flex; align-items: center; gap: 6px; flex: none; white-space: nowrap; }
.pill-btn:hover { background: var(--hover); }
.pill-btn.on { background: var(--active); border-color: var(--active); }
.badge-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--accent); display: none; }
.pill-btn.ping .badge-dot { display: inline-block; animation: pulse calc(1200ms / var(--k)) infinite; }
.chat-scroll { flex: 1; min-height: 0; overflow-y: auto; scroll-behavior: smooth; }
.chat { max-width: 768px; margin: 0 auto; padding: 4px 24px 32px; }

/* messages */
.msg { margin: 22px 0; display: flex; }
.msg.user { justify-content: flex-end; }
.msg.user .bubble { background: var(--bubble); border-radius: 18px; padding: 9px 16px; max-width: 78%; white-space: pre-wrap;
  animation: rise calc(300ms / var(--k)) ease-out; }
.msg.assistant { gap: 14px; align-items: flex-start; }
.avatar { width: 30px; height: 30px; border-radius: 50%; border: 1px solid var(--border); display: grid; place-items: center; flex: none;
  font: 600 12px/1 var(--mono); color: var(--muted); }
.a-body { flex: 1; min-width: 0; padding-top: 3px; }
.steps { border-left: 2px solid var(--border); margin: 2px 0 14px; padding: 2px 0 2px 12px; display: flex; flex-direction: column; gap: 5px; }
.steps.folded { display: none; }
.steps-toggle { font-size: 13px; color: var(--muted); background: none; border: 0; padding: 0; cursor: pointer; margin: 0 0 8px; }
.steps-toggle:hover { color: var(--text); }
.step { font-size: 13px; color: var(--muted); line-height: 1.5; display: flex; gap: 8px; animation: rise calc(300ms / var(--k)) ease-out; }
.step .ic { flex: none; width: 14px; text-align: center; font-weight: 700; }
.step b { color: var(--text); font-weight: 600; }
.step.pass .ic { color: var(--pass); } .step.fail .ic { color: var(--fail); } .step.route .ic { color: var(--route); }
.step.info .ic { color: var(--info); } .step.call .ic { color: var(--accent); }
.step.fail b { color: var(--fail); }
.thinking { display: flex; align-items: center; gap: 10px; color: var(--muted); font-size: 14px; }
.thinking .dot { width: 11px; height: 11px; border-radius: 50%; background: var(--text); animation: pulse calc(1100ms / var(--k)) infinite ease-in-out; }
.a-answer { animation: rise calc(350ms / var(--k)) ease-out; }
.a-answer p { margin: 0 0 10px; }
.a-answer code { background: var(--code-bg); border: 1px solid var(--border); padding: 0 5px; border-radius: 5px; }
.solution-card { background: var(--code-bg); border: 1px solid var(--border); border-radius: 12px; padding: 10px 14px; font: 13.5px/1.65 var(--mono); }
.solution-card .ans { font-weight: 700; color: var(--solution); }
.harness-badge { display: inline-block; font-size: 11px; font-weight: 600; color: var(--info); border: 1px solid var(--border);
  border-radius: 999px; padding: 0 8px; margin-left: 8px; vertical-align: 1px; }
.read-label { font-size: 12.5px; color: var(--faint); margin: 4px 0 0; }
.snippet { border: 1px solid var(--border); border-radius: 12px; overflow: hidden; margin: 6px 0 8px; }
.snippet .head { font: 12px/1.4 var(--mono); color: var(--muted); padding: 7px 12px; background: var(--hover); }
.snippet pre { margin: 0; padding: 8px 12px; font: 12.5px/1.55 var(--mono); overflow-x: auto; background: var(--code-bg); }
.grade { font-size: 12.5px; color: var(--faint); margin-top: 4px; display: flex; gap: 6px; }
.grade.ok::before { content: "\2713"; color: var(--pass); font-weight: 700; }
.grade.bad::before { content: "\2715"; color: var(--fail); font-weight: 700; }

/* empty state */
.empty { max-width: 640px; margin: 10vh auto 0; text-align: center; }
.empty h1 { font-size: 28px; font-weight: 600; margin: 0 0 12px; letter-spacing: -.01em; line-height: 1.25; }
.empty p { color: var(--muted); margin: 0 auto 16px; max-width: 580px; }
.empty .hint { font-size: 13.5px; color: var(--faint); margin-top: 20px; }
.legend { display: flex; flex-wrap: wrap; justify-content: center; gap: 6px; font-size: 12.5px; }
.legend span { border: 1px solid var(--border); border-radius: 999px; padding: 2px 10px; color: var(--muted); }
.legend b { font: 600 12px var(--mono); }
.project-card { text-align: left; border: 1px solid var(--border); border-radius: 14px; padding: 12px 16px; margin: 6px auto 0; font-size: 13.5px; color: var(--muted); }
.project-card b { color: var(--text); }
.project-card .renames, .project-card .files { font: 12px/1.6 var(--mono); margin-top: 8px; color: var(--faint); }

/* composer */
.composer-wrap { padding: 0 24px 12px; flex: none; }
.composer { max-width: 768px; margin: 0 auto; background: var(--composer); border: 1px solid var(--composer-border); border-radius: 28px;
  box-shadow: var(--shadow); display: flex; align-items: flex-end; gap: 10px; padding: 11px 11px 11px 20px; cursor: pointer; min-height: 56px; }
.composer:hover:not(.disabled) { border-color: var(--faint); }
.composer:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.composer.disabled { cursor: default; }
.composer-text { flex: 1; min-height: 28px; padding: 4px 0; white-space: pre-wrap; word-break: break-word; }
.composer-text.placeholder { color: var(--faint); }
.caret-inline { display: inline-block; width: 2px; height: 1.1em; background: var(--text); vertical-align: -3px; margin-left: 1px; animation: blink 1s steps(1) infinite; }
.send { width: 36px; height: 36px; border-radius: 50%; border: 0; background: var(--send); color: var(--send-text); display: grid; place-items: center;
  cursor: pointer; flex: none; transition: opacity .2s; }
.send:disabled { opacity: .22; cursor: default; }
.send.ready { animation: ready 1.4s ease-in-out infinite; }
.skip { border: 1px solid var(--border); background: transparent; border-radius: 999px; padding: 5px 12px; font-size: 12.5px; color: var(--muted); cursor: pointer; flex: none; align-self: center; }
.skip:hover { background: var(--hover); color: var(--text); }
.footnote { text-align: center; font-size: 12px; color: var(--faint); padding-top: 8px; }

/* the context panel */
.panel { background: var(--panel); border-left: 1px solid var(--border); display: none; flex-direction: column; min-width: 0; min-height: 0; height: 100%; overflow: hidden; }
.app.panel-open .panel { display: flex; }
.p-head { padding: 12px 16px 12px; border-bottom: 1px solid var(--border); flex: none; }
.p-title { display: flex; align-items: center; gap: 8px; font-weight: 600; }
.p-title small { font-weight: 400; color: var(--faint); font-size: 12.5px; }
.req-row { display: flex; align-items: baseline; gap: 10px; margin-top: 6px; }
.kind { font: 700 11px/1 var(--mono); letter-spacing: .05em; padding: 5px 8px; border-radius: 6px; color: var(--kind-text); background: var(--info);
  text-transform: uppercase; white-space: nowrap; flex: none; }
.kind.route { background: var(--route); } .kind.solution { background: var(--solution); } .kind.explanation { background: var(--explanation); }
.kind.scope { background: var(--file); } .kind.answer { background: var(--message); }
.p-head.flash .kind { animation: pop calc(450ms / var(--k)) ease-out; }
.note { font-size: 13px; color: var(--muted); line-height: 1.45; }
.meta { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 8px; }
.meta span { font-size: 12px; border: 1px solid var(--border); border-radius: 999px; padding: 0 9px; color: var(--muted); white-space: nowrap; }
.meta code { color: var(--text); }
.pipeline { display: flex; align-items: center; flex-wrap: wrap; gap: 4px; margin-top: 10px; font-size: 12px; }
.pipe-group { display: inline-flex; align-items: center; gap: 3px; }
.pipe { border: 1px solid var(--border); border-radius: 999px; padding: 1px 9px; color: var(--faint); transition: all calc(250ms / var(--k)); white-space: nowrap; }
.pipe.active { border-color: var(--accent); color: var(--text); background: var(--enter); }
.pipe.done { color: var(--muted); }
.pipe.done::before { content: "\2713\00a0"; color: var(--pass); }
.pipe.failed { border-color: var(--fail); color: var(--text); background: var(--leave); }
.pipe.skipped { opacity: .45; text-decoration: line-through; }
.pipe-or, .arrow { color: var(--faint); }
details.instr { margin-top: 8px; font-size: 12.5px; color: var(--muted); }
details.instr summary { cursor: pointer; }
details.instr pre { white-space: pre-wrap; font: 12px/1.5 var(--mono); background: var(--code-bg); border: 1px solid var(--border); border-radius: 8px;
  padding: 8px 10px; max-height: 220px; overflow: auto; margin: 6px 0 0; }
.ctx { flex: 1; min-height: 0; overflow: auto; padding: 12px 16px 28px; font: 12.5px/1.55 var(--mono); position: relative; }
.ctx.same { animation: same calc(1100ms / var(--k)) ease-out; }
.ctx.fading > * { opacity: .12; transition: opacity calc(350ms / var(--k)); }
.ctx-empty { font: 13.5px/1.55 ui-sans-serif, -apple-system, "Segoe UI", sans-serif; color: var(--faint); padding: 18px 4px; }
.blk { white-space: pre-wrap; word-break: break-word; padding: 3px 8px; margin: 0 -8px 4px; border-radius: 6px; box-shadow: inset 2px 0 0 var(--border); }
.blk[data-tag="task"] { box-shadow: inset 2px 0 0 var(--task); } .blk[data-tag="solution"] { box-shadow: inset 2px 0 0 var(--solution); }
.blk[data-tag="question"] { box-shadow: inset 2px 0 0 var(--question); } .blk[data-tag="explanation"] { box-shadow: inset 2px 0 0 var(--explanation); }
.blk[data-tag="message"] { box-shadow: inset 2px 0 0 var(--message); }
.tag { font-weight: 700; }
.tag.task { color: var(--task); } .tag.solution { color: var(--solution); } .tag.question { color: var(--question); }
.tag.explanation { color: var(--explanation); } .tag.message { color: var(--message); }
.pending { display: none !important; }
.entering { animation: rise calc(420ms / var(--k)) ease-out, glow calc(2600ms / var(--k)) ease-out; }
.morph { animation: morph calc(1800ms / var(--k)) ease-out; }
.blk.leaving { background: var(--leave); }
.blk.leaving .inner { text-decoration: line-through; text-decoration-color: var(--fail); }
.drop-note { display: block; font: 11.5px/1.4 ui-sans-serif, -apple-system, "Segoe UI", sans-serif; color: var(--fail); margin-top: 2px; }
.collapsing { overflow: hidden; opacity: 0; margin-top: 0 !important; margin-bottom: 0 !important; padding-top: 0 !important; padding-bottom: 0 !important;
  transition: height calc(450ms / var(--k)) ease, opacity calc(450ms / var(--k)) ease, margin calc(450ms / var(--k)), padding calc(450ms / var(--k)); }
.wtag { color: var(--wrap); font-weight: 700; }
.file { margin: 2px 0 6px; overflow: hidden; transition: max-height calc(650ms / var(--k)) ease, opacity calc(500ms / var(--k)) ease, margin calc(650ms / var(--k)); }
.file.gone { max-height: 0 !important; opacity: 0; margin: 0; }
.ftag { color: var(--file); font-weight: 700; }
.ln { display: flex; max-height: 6em; overflow: hidden; border-radius: 3px;
  transition: max-height calc(550ms / var(--k)) ease, opacity calc(450ms / var(--k)) ease, background-color calc(300ms / var(--k)); }
.ln .no { color: var(--faint); min-width: 3.4ch; text-align: right; padding-right: 1.2ch; user-select: none; flex: none; }
.ln .code { white-space: pre-wrap; word-break: break-word; min-width: 0; }
.ln.in-scope { background: var(--scope); }
.ln.gone { max-height: 0; opacity: 0; }
.reply { flex: none; border-top: 1px solid var(--border); padding: 9px 16px 14px; max-height: 36%; overflow: auto; }
.reply-label { font-size: 12px; color: var(--faint); margin-bottom: 4px; }
.reply pre { margin: 0; white-space: pre-wrap; word-break: break-word; font: 12.5px/1.55 var(--mono); min-height: 1.55em; }
.reply .prefix { color: var(--faint); }
.reply .discard { color: var(--faint); text-decoration: line-through; opacity: .75; }
.caret { display: inline-block; width: 7px; height: 1.05em; background: var(--text); vertical-align: -2px; animation: blink 1s steps(1) infinite; }
.scrim { display: none; }

@keyframes rise { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
@keyframes glow { from { background-color: var(--enter); } to { background-color: transparent; } }
@keyframes morph { from { background-color: var(--scope); } to { background-color: transparent; } }
@keyframes same { from { box-shadow: inset 0 0 0 1px var(--accent); } to { box-shadow: inset 0 0 0 1px transparent; } }
@keyframes pulse { 0%, 100% { transform: scale(.75); opacity: .55; } 50% { transform: scale(1); opacity: 1; } }
@keyframes blink { 50% { opacity: 0; } }
@keyframes pop { 0% { transform: scale(.85); } 60% { transform: scale(1.06); } 100% { transform: none; } }
@keyframes ready { 0%, 100% { box-shadow: 0 0 0 0 rgba(16, 163, 127, .0); } 50% { box-shadow: 0 0 0 6px rgba(16, 163, 127, .25); } }

@media (max-width: 1180px) {
  .app, .app.panel-open { grid-template-columns: 260px minmax(0, 1fr); }
  .panel, .app.panel-open .panel { display: flex; position: fixed; top: 0; right: 0; bottom: 0; width: min(620px, 94vw); z-index: 30;
    box-shadow: var(--shadow); transform: translateX(105%); transition: transform .3s ease; }
  .app.panel-open .panel { transform: none; }
  .scrim.show { display: block; position: fixed; inset: 0; background: rgba(0, 0, 0, .35); z-index: 20; }
}
@media (max-width: 760px) {
  .app, .app.panel-open { grid-template-columns: minmax(0, 1fr); }
  .sidebar { position: fixed; top: 0; bottom: 0; left: 0; width: 280px; z-index: 40; transform: translateX(-105%); transition: transform .3s ease; }
  .app.sidebar-open .sidebar { transform: none; }
  .model-pill, .seg, .title .light { display: none; }
  .title { font-size: 16px; min-width: 0; overflow: hidden; text-overflow: ellipsis; }
  .pill-btn { padding: 5px 10px; font-size: 12.5px; }
  .chat { padding: 4px 16px 28px; }
  .composer-wrap { padding: 0 12px 10px; }
  .empty h1 { font-size: 23px; }
  .topbar { padding: 0 8px; }
}
.menu-btn { display: none; }
@media (prefers-reduced-motion: reduce) { .chat-scroll { scroll-behavior: auto; } }
@media (max-width: 1600px) { .app.panel-open .model-pill { display: none; } }
@media (max-width: 760px) { .menu-btn { display: grid; } }
</style>
</head>
<body>
<div class="app" id="app">
  <aside class="sidebar" id="sidebar">
    <div class="side-top">
      <div class="brand"><span class="logo">&#9881;</span>Harness chat</div>
      <button class="icon-btn" id="restart" title="Start this chat again" aria-label="Start this chat again">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"/></svg>
      </button>
    </div>
    <nav id="side-list"></nav>
    <div class="side-foot">
      <button class="side-item" id="theme">&#9680;&nbsp; Light / dark</button>
      <p>A local page. The replies are Gemma 4 12B's own, saved from the data passes; the harness code rebuilt every context.</p>
    </div>
  </aside>
  <div class="scrim" id="scrim"></div>
  <main class="main">
    <header class="topbar">
      <button class="icon-btn menu-btn" id="menu" aria-label="Chats">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M4 7h16M4 12h16M4 17h16"/></svg>
      </button>
      <div class="title"><span id="title-text">Math tutor</span><span class="light">harness</span></div>
      <span class="model-pill" id="model-pill"></span>
      <span class="spacer"></span>
      <div class="seg" id="speed" role="group" aria-label="Speed"></div>
      <button class="pill-btn" id="auto" aria-pressed="false">Auto-play</button>
      <button class="pill-btn" id="ctx-toggle" aria-pressed="false"><span class="badge-dot"></span>Context</button>
    </header>
    <div class="chat-scroll" id="chat-scroll"><div class="chat" id="chat"></div></div>
    <div class="composer-wrap">
      <div class="composer" id="composer" role="button" tabindex="0" aria-label="Write the next message">
        <div class="composer-text placeholder" id="composer-text"></div>
        <button class="skip" id="skip" hidden>Skip &#9656;&#9656;</button>
        <button class="send" id="send" disabled aria-label="Send">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V5M5 12l7-7 7 7"/></svg>
        </button>
      </div>
      <div class="footnote">Gemma 4 12B (4-bit, local) wrote these replies in the data passes. The distilled student plugs into the same harness.</div>
    </div>
  </main>
  <section class="panel" id="panel" aria-label="Context">
    <div class="p-head" id="p-head">
      <div class="p-title"><span>Context</span><small>what the model reads in this call</small><span class="spacer"></span>
        <button class="icon-btn" id="ctx-close" aria-label="Close the context panel">&#10005;</button></div>
      <div class="req-row"><span class="kind" id="kind">No call yet</span><span class="note" id="note"></span></div>
      <div class="meta" id="meta"></div>
      <div class="pipeline" id="pipeline"></div>
      <details class="instr"><summary>The instructions Gemma also gets (the student does not)</summary><pre id="instr"></pre></details>
    </div>
    <div class="ctx" id="ctx"></div>
    <div class="reply" id="reply"><div class="reply-label" id="reply-label">Gemma's reply</div><pre id="reply-text"></pre></div>
  </section>
</div>
<script id="data" type="application/json">__DATA__</script>
<script>
(() => {
"use strict";
const DATA = JSON.parse(document.getElementById("data").textContent);
const $ = (selector) => document.querySelector(selector);
const app = $("#app");

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

const saved = {
  get(key, fallback) {
    try { const value = localStorage.getItem("harness-chat:" + key); return value === null ? fallback : JSON.parse(value); }
    catch (error) { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem("harness-chat:" + key, JSON.stringify(value)); } catch (error) { /* storage blocked: fine */ }
  },
};

const KIND_LABEL = { route: "Route", solution: "Solution", explanation: "Explanation", scope: "Scope", answer: "Answer" };
const BRANCHES = ["solution", "explanation", "oos"];
const PIPES = {
  math: [["route", "Route"], [["solution", "Solution"], ["explanation", "Explanation"], ["oos", "Off topic"]], ["check", "Check"], ["store", "Context updated"]],
  code: [["scope", "Scope"], ["check1", "Check"], ["narrow", "Narrow"], ["answer", "Answer"], ["check2", "Check"]],
};
const INTRO = {
  math: "Every pupil message is one agent turn. A route call reads the conversation and picks the next step with its first tag. " +
        "Then comes a solution call, an explanation call, or the harness's own reply to an off-topic message; a check; and the " +
        "harness's bookkeeping of what the next call will see.",
  code: "Every request is two model calls. The scope call reads the whole project and writes the lines that matter. The harness " +
        "checks them and narrows the context to them. The answer call explains the code, or says it found it.",
};
const ICONS = { call: "↗", route: "↪", pass: "✓", fail: "✕", info: "•" };

const state = { chat: 0, turn: 0, mode: "idle", speed: Number(saved.get("speed", 1)) || 1, auto: false, skip: false, run: 0, calls: 0,
                panelOpen: window.innerWidth > 1180, view: null, stream: null, caret: null };

class Aborted extends Error {}
function alive(run) { if (run !== state.run) throw new Aborted(); }
async function wait(ms, run) {
  alive(run);
  if (!state.skip && ms > 0) await new Promise((resolve) => setTimeout(resolve, ms / state.speed));
  alive(run);
}

/* ---- sidebar, top bar ---------------------------------------------------------------------- */
function renderSidebar() {
  const list = $("#side-list");
  list.textContent = "";
  let group = null;
  DATA.chats.forEach((chat, index) => {
    if (chat.group !== group) { group = chat.group; list.append(el("div", "side-group", group)); }
    const item = el("button", "side-item" + (index === state.chat ? " active" : ""));
    item.append(el("span", null, chat.title), el("small", null, chat.turns.length + (chat.turns.length === 1 ? " message" : " messages")));
    item.addEventListener("click", () => { openChat(index); setSidebar(false); });
    list.append(item);
  });
}

function renderSpeed() {
  const box = $("#speed");
  box.textContent = "";
  [0.5, 1, 2, 4].forEach((speed) => {
    const button = el("button", speed === state.speed ? "on" : null, speed + "×");
    button.addEventListener("click", () => setSpeed(speed));
    box.append(button);
  });
}

function setSpeed(speed) {
  state.speed = speed;
  saved.set("speed", speed);
  document.documentElement.style.setProperty("--k", String(speed));
  renderSpeed();
}

function setPanel(open) {
  state.panelOpen = open;
  app.classList.toggle("panel-open", open);
  const toggle = $("#ctx-toggle");
  toggle.classList.toggle("on", open);
  toggle.setAttribute("aria-pressed", String(open));
  if (open) toggle.classList.remove("ping");
  updateScrim();
}

function setSidebar(open) { app.classList.toggle("sidebar-open", open); updateScrim(); }

function updateScrim() {
  const drawer = window.innerWidth <= 1180;
  $("#scrim").classList.toggle("show", app.classList.contains("sidebar-open") || (drawer && state.panelOpen));
}

function pingContext() { if (!state.panelOpen) $("#ctx-toggle").classList.add("ping"); }

function toggleTheme() {
  const current = document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const next = current === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  saved.set("theme", next);
}

/* ---- a chat ---------------------------------------------------------------------------------- */
function openChat(index) {
  state.run += 1;
  Object.assign(state, { chat: index, turn: 0, skip: false, calls: 0, view: null });
  document.body.classList.remove("fast");
  const chat = DATA.chats[index];
  renderSidebar();
  $("#title-text").textContent = chat.group;
  $("#model-pill").textContent = chat.subtitle;
  $("#model-pill").title = chat.detail;
  renderEmpty(chat);
  resetPanel(chat);
  setComposer("idle");
  if (state.auto) autoNext(state.run);
}

function renderEmpty(chat) {
  const box = $("#chat");
  box.textContent = "";
  const empty = el("div", "empty");
  empty.append(el("h1", null, chat.agent === "math" ? "How the math tutor's harness works" : "How the coding agent's harness works"));
  empty.append(el("p", null, INTRO[chat.agent]));
  if (chat.agent === "math") {
    const legend = el("div", "legend");
    [["task", "a task"], ["solution", "its solution"], ["question", "a question"], ["explanation", "its explanation"], ["message", "the new message"]]
      .forEach(([tag, words]) => {
        const chip = el("span");
        chip.append(el("b", "tag " + tag, "<" + tag + ">"), document.createTextNode(" " + words));
        legend.append(chip);
      });
    empty.append(legend);
    empty.append(el("p", "hint", "Tasks and solutions are the history. Questions and explanations are the side thread: a new task drops it."));
  } else {
    const project = chat.project, card = el("div", "project-card");
    card.append(el("b", null, "Project " + project.name + ", version " + project.version),
                document.createTextNode(" · " + project.files.length + " files, " + project.lines + " lines. A randomized copy of the " +
                                        "base project: names renamed, functions, methods and files shuffled."));
    if (project.renames.length) {
      card.append(el("div", "renames", "Renamed here: " + project.renames.slice(0, 10).join(", ") + (project.renames.length > 10 ? ", …" : "")));
    }
    card.append(el("div", "files", project.files.join("   ")));
    empty.append(card);
  }
  empty.append(el("p", "hint", "Click the message box below to write the next message, then send it. The Context panel shows each model call as the harness builds it."));
  box.append(empty);
}

function setComposer(mode) {
  state.mode = mode;
  const chat = DATA.chats[state.chat], box = $("#composer"), text = $("#composer-text"), send = $("#send");
  box.classList.toggle("disabled", mode === "busy" || mode === "typing");
  send.disabled = mode !== "ready";
  send.classList.toggle("ready", mode === "ready");
  $("#skip").hidden = mode !== "busy" && mode !== "typing";
  if (mode === "idle" || mode === "busy" || mode === "done") {
    text.classList.add("placeholder");
    const position = " (" + (state.turn + 1) + " of " + chat.turns.length + ")";
    text.textContent = mode === "idle" ? (chat.agent === "math" ? "Click to write the pupil's next message" : "Click to write the next request") + position
      : mode === "busy" ? "The harness is working…" : "End of the saved conversation. Click to start it again.";
  }
}

async function typeNext() {
  if (state.mode === "done") { openChat(state.chat); return; }
  if (state.mode !== "idle") return;
  const run = state.run, message = DATA.chats[state.chat].turns[state.turn].user;
  setComposer("typing");
  const text = $("#composer-text");
  text.textContent = "";
  text.classList.remove("placeholder");
  const typed = el("span"), caret = el("span", "caret-inline");
  text.append(typed, caret);
  try {
    for (let index = 1; index <= message.length; index += 1) { typed.textContent = message.slice(0, index); await wait(14, run); }
  } catch (error) { if (error instanceof Aborted) return; throw error; }
  caret.remove();
  state.skip = false;
  document.body.classList.remove("fast");
  setComposer("ready");
}

async function send() {
  if (state.mode !== "ready") return;
  const run = state.run, chat = DATA.chats[state.chat], turn = chat.turns[state.turn];
  setComposer("busy");
  try { await playTurn(chat, turn, run); }
  catch (error) { if (error instanceof Aborted) return; throw error; }
  state.skip = false;
  document.body.classList.remove("fast");
  state.turn += 1;
  setComposer(state.turn < chat.turns.length ? "idle" : "done");
  if (state.auto && state.mode === "idle") autoNext(run);
}

async function autoNext(run) {
  try { await wait(1100, run); } catch (error) { return; }
  if (!state.auto || state.mode !== "idle") return;
  await typeNext();
  if (run === state.run && state.mode === "ready") send();
}

function scrollChat() { const box = $("#chat-scroll"); box.scrollTop = box.scrollHeight; }

function foldSteps() {
  document.querySelectorAll("#chat .steps:not(.folded)").forEach((steps) => {
    steps.classList.add("folded");
    const count = steps.childElementCount;
    const toggle = el("button", "steps-toggle", "Show the " + count + " harness steps");
    toggle.addEventListener("click", () => {
      const folded = steps.classList.toggle("folded");
      toggle.textContent = (folded ? "Show" : "Hide") + " the " + count + " harness steps";
    });
    steps.before(toggle);
  });
}

async function playTurn(chat, turn, run) {
  const box = $("#chat");
  if (state.turn === 0) box.textContent = "";
  foldSteps();
  const user = el("div", "msg user");
  user.append(el("div", "bubble", turn.user));
  box.append(user);
  const reply = el("div", "msg assistant"), body = el("div", "a-body"), steps = el("div", "steps"), thinking = el("div", "thinking");
  thinking.append(el("span", "dot"), el("span", null, "The harness is running"));
  body.append(steps, thinking);
  reply.append(el("div", "avatar", chat.agent === "math" ? "∑" : "</>"), body);
  box.append(reply);
  scrollChat();
  await wait(450, run);
  const view = { steps, body, thinking };
  for (const event of turn.events) await play(chat, event, view, run);
  thinking.remove();
  scrollChat();
}

async function play(chat, event, view, run) {
  pipelineFor(chat, event);
  if (event.type === "request") {
    state.calls += 1;
    addStep(view.steps, "call", "Model call " + state.calls + ": " + KIND_LABEL[event.kind].toLowerCase(),
            event.chars.toLocaleString("en") + " characters of context" + (event.temperature > 0 ? ", sampled" : "") + ".");
    await showRequest(chat, event, run);
  } else if (event.type === "reply") {
    await streamReply(event, run);
  } else if (event.type === "harness") {
    addStep(view.steps, event.status, event.title + (event.text ? "." : ""), event.text);
    await wait(event.status === "fail" ? 1400 : 850, run);
  } else if (event.type === "narrow") {
    await narrow(event.ranges, run);
  } else if (event.type === "drop_message") {
    await dropMessage(run);
  } else if (event.type === "assistant") {
    view.thinking.remove();
    view.body.append(renderAnswer(event));
    scrollChat();
    await wait(550, run);
  } else if (event.type === "grade") {
    view.body.append(el("div", "grade " + (event.ok ? "ok" : "bad"), event.text));
    scrollChat();
    await wait(300, run);
  }
}

function addStep(steps, status, title, text) {
  const step = el("div", "step " + status), words = el("span");
  step.append(el("span", "ic", ICONS[status] || "•"));
  words.append(el("b", null, title));
  if (text) words.append(document.createTextNode(" " + text));
  step.append(words);
  steps.append(step);
  scrollChat();
}

/* ---- the pipeline ---------------------------------------------------------------------------- */
function renderPipeline(agent) {
  const box = $("#pipeline");
  box.textContent = "";
  PIPES[agent].forEach((entry, index) => {
    if (index) box.append(el("span", "arrow", "→"));
    if (Array.isArray(entry[0])) {
      const group = el("span", "pipe-group");
      entry.forEach(([stage, label], position) => {
        if (position) group.append(el("span", "pipe-or", "/"));
        const pipe = el("span", "pipe", label);
        pipe.dataset.stage = stage;
        group.append(pipe);
      });
      box.append(group);
    } else {
      const pipe = el("span", "pipe", entry[1]);
      pipe.dataset.stage = entry[0];
      box.append(pipe);
    }
  });
}

function pipes() { return [...document.querySelectorAll("#pipeline .pipe")]; }
function resetPipeline() { pipes().forEach((pipe) => { pipe.className = "pipe"; }); }

function setStage(stage, status) {
  const all = pipes(), target = all.find((pipe) => pipe.dataset.stage === stage);
  if (!target) return;
  all.forEach((pipe) => {
    if (pipe !== target && pipe.classList.contains("active")) { pipe.classList.remove("active"); pipe.classList.add("done"); }
  });
  target.classList.remove("active", "done", "failed", "skipped");
  target.classList.add(status);
  if (status === "active" && BRANCHES.includes(stage)) {
    all.filter((pipe) => BRANCHES.includes(pipe.dataset.stage) && pipe !== target).forEach((pipe) => { pipe.className = "pipe skipped"; });
  }
}

function finishPipeline() {
  pipes().forEach((pipe) => {
    if (pipe.classList.contains("active")) { pipe.classList.remove("active"); pipe.classList.add("done"); }
    else if (pipe.className === "pipe") pipe.classList.add("skipped");
  });
}

function pipelineFor(chat, event) {
  if (event.type === "request") {
    if ((chat.agent === "math" && event.kind === "route") || (chat.agent === "code" && event.kind === "scope")) resetPipeline();
    setStage(event.stage, "active");
  } else if (event.type === "harness") {
    if (event.title === "A new request") resetPipeline();
    else if (event.status === "route" && chat.agent === "math") setStage(event.stage, "active");
    else setStage(event.stage, event.status === "fail" ? "failed" : "done");
  } else if (event.type === "narrow") {
    setStage("narrow", "active");
  } else if (event.type === "assistant") {
    finishPipeline();
  }
}

/* ---- the context panel ----------------------------------------------------------------------- */
function resetPanel(chat) {
  const kind = $("#kind");
  kind.className = "kind";
  kind.textContent = "No call yet";
  $("#note").textContent = "Send a message: each model call's context is built here.";
  $("#meta").textContent = "";
  $("#instr").textContent = chat.instructions;
  renderPipeline(chat.agent);
  const ctx = $("#ctx");
  ctx.textContent = "";
  ctx.append(el("div", "ctx-empty", chat.agent === "math"
    ? "The conversation's blocks appear here as the harness builds them: the history, the side thread and the new message."
    : "The project appears here: every file with its lines numbered, then the user's message."));
  $("#reply-text").textContent = "";
  $("#reply-label").textContent = "Gemma's reply";
}

function chip(label, code) {
  const span = el("span", null, label);
  if (code !== undefined) { span.append(document.createTextNode(" ")); span.append(el("code", null, code)); }
  return span;
}

async function showRequest(chat, event, run) {
  const head = $("#p-head"), kind = $("#kind");
  kind.className = "kind " + event.kind;
  kind.textContent = KIND_LABEL[event.kind] + " call · " + state.calls;
  head.classList.remove("flash");
  void head.offsetWidth;
  head.classList.add("flash");
  $("#note").textContent = event.note;
  const meta = $("#meta");
  meta.textContent = "";
  if (event.prefix) meta.append(chip("reply starts with", event.prefix.replace(/\n/g, "⏎")));
  if (event.stop.length) meta.append(chip("stops at", event.stop.join("  ")));
  meta.append(chip("at most " + event.maxTokens + " tokens"),
              chip(event.temperature > 0 ? "sampled, temperature " + event.temperature : "greedy"),
              chip(event.chars.toLocaleString("en") + " characters"));
  resetReply(event.prefix);
  pingContext();
  if (chat.agent === "math") await showBlocks(event.context.blocks, run);
  else await showProject(event.context, run);
}

function innerOf(tag, text) { return text.slice(tag.length + 2, text.length - tag.length - 3); }

function fillBlock(node, tag, text) {
  node.textContent = "";
  node.dataset.tag = tag;
  node.append(el("span", "tag " + tag, "<" + tag + ">"), el("span", "inner", innerOf(tag, text)), el("span", "tag " + tag, "</" + tag + ">"));
}

function blockNode(tag, text) { const node = el("div", "blk"); fillBlock(node, tag, text); return node; }

function animateOnce(node, className) {
  // the class comes off when the animation is over (a timer: a hidden panel runs no animations and fires no
  // animationend), so a node moved later does not play it again
  node.classList.remove("entering", "morph");
  void node.offsetWidth;
  node.classList.add(className);
  clearTimeout(node.animationTimer);
  node.animationTimer = setTimeout(() => node.classList.remove(className), (className === "entering" ? 2600 : 1800) / state.speed + 80);
}

function keyed(blocks) {
  const seen = {};
  return blocks.map((block) => {
    const base = block.tag + "\u0001" + block.text;
    seen[base] = (seen[base] || 0) + 1;
    return { key: base + "\u0001" + seen[base], tag: block.tag, text: block.text, node: null, from: null };
  });
}

function collapse(node) {
  return new Promise((resolve) => {
    if (state.skip) { node.remove(); resolve(); return; }
    node.style.height = node.offsetHeight + "px";
    void node.offsetHeight;
    node.classList.add("collapsing");
    node.style.height = "0px";
    setTimeout(() => { node.remove(); resolve(); }, 480 / state.speed);
  });
}

async function typeInto(node, text, run, ms) {
  for (let index = 2; index < text.length; index += 2) { node.textContent = text.slice(0, index); await wait(ms, run); }
  node.textContent = text;
}

async function showBlocks(blocks, run) {
  const ctx = $("#ctx"), next = keyed(blocks);
  const previous = state.view && state.view.type === "math" ? state.view.items : [];
  const nextKeys = new Set(next.map((item) => item.key)), oldKeys = new Set(previous.map((item) => item.key));
  const leaving = previous.filter((item) => !nextKeys.has(item.key));
  const entering = next.filter((item) => !oldKeys.has(item.key));
  // a message that moved into the history (a task) or the side thread (a question) keeps its place and changes its tag
  const moved = [];
  leaving.filter((item) => item.tag === "message").forEach((item) => {
    const target = entering.find((candidate) => (candidate.tag === "task" || candidate.tag === "question") && !candidate.from &&
                                                 innerOf(candidate.tag, candidate.text) === innerOf("message", item.text));
    if (target) { target.from = item; moved.push(item); }
  });
  const dying = leaving.filter((item) => !moved.includes(item));
  if (dying.length) {
    dying.forEach((item) => item.node.classList.add("leaving"));
    dying[0].node.scrollIntoView({ block: "nearest", behavior: "smooth" });
    await wait(950, run);
    await Promise.all(dying.map((item) => collapse(item.node)));
  }
  for (const item of entering.filter((candidate) => candidate.from)) {
    item.node = item.from.node;
    fillBlock(item.node, item.tag, item.text);
    animateOnce(item.node, "morph");
    item.node.scrollIntoView({ block: "nearest", behavior: "smooth" });
    await wait(700, run);
  }
  next.forEach((item) => { const kept = previous.find((candidate) => candidate.key === item.key); if (kept) item.node = kept.node; });
  const fresh = [];
  let cursor = ctx.firstChild;
  for (const item of next) {   // only nodes out of place move (a moved node would replay its animation)
    if (!item.node) {
      item.node = blockNode(item.tag, item.tag === "message" ? "<message></message>" : item.text);
      item.node.classList.add("pending");
      fresh.push(item);
    }
    if (item.node === cursor) cursor = cursor.nextSibling;
    else ctx.insertBefore(item.node, cursor);
  }
  [...ctx.children].forEach((child) => { if (!next.some((item) => item.node === child)) child.remove(); });
  state.view = { type: "math", items: next };
  if (!fresh.length && !dying.length && !moved.length && previous.length) {
    ctx.classList.remove("same");
    void ctx.offsetWidth;
    ctx.classList.add("same");
    await wait(500, run);
  }
  for (const item of fresh) {
    item.node.classList.remove("pending");
    animateOnce(item.node, "entering");
    item.node.scrollIntoView({ block: "nearest", behavior: "smooth" });
    if (item.tag === "message") await typeInto(item.node.querySelector(".inner"), innerOf("message", item.text), run, 14);
    else await wait(480, run);
  }
}

async function dropMessage(run) {
  if (!state.view || state.view.type !== "math") return;
  const item = [...state.view.items].reverse().find((candidate) => candidate.tag === "message");
  if (!item) return;
  harnessHeader("No more model calls this turn: the harness answers the off-topic message itself and does not keep it.");
  item.node.classList.add("leaving");
  item.node.append(el("span", "drop-note", "Not kept: the model never sees this message."));
  item.node.scrollIntoView({ block: "nearest", behavior: "smooth" });
  await wait(1500, run);
  await collapse(item.node);
  state.view.items = state.view.items.filter((candidate) => candidate !== item);
}

function lineNode(number, text) {
  const row = el("div", "ln");
  row.dataset.n = String(number);
  row.append(el("span", "no", number + ":"), el("span", "code", text));
  return row;
}

function fileNode(file) {
  const node = el("div", "file"), lines = el("div", "lines");
  node.dataset.name = file.name;
  node.append(el("div", "ftag", "<file name=\"" + file.name + "\">"));
  file.lines.forEach(([number, text]) => lines.append(lineNode(number, text)));
  node.append(lines, el("div", "ftag", "</file>"));
  return node;
}

async function showProject(context, run) {
  const ctx = $("#ctx");
  if (context.wrapper === "context" && state.view && state.view.type === "code" && state.view.narrowed) { renderExact(context); return; }
  if (ctx.querySelector(".file, .blk")) { ctx.classList.add("fading"); await wait(380, run); }
  ctx.textContent = "";
  ctx.classList.remove("fading");
  ctx.scrollTop = 0;
  ctx.append(el("div", "wtag", "<" + context.wrapper + ">"));
  for (const file of context.files) {
    const node = fileNode(file);
    ctx.append(node);
    animateOnce(node, "entering");
    ctx.scrollTop = ctx.scrollHeight;
    await wait(context.files.length > 4 ? 130 : 260, run);
  }
  ctx.append(el("div", "wtag", "</" + context.wrapper + ">"));
  const message = blockNode("message", "<message></message>");
  ctx.append(message);
  animateOnce(message, "entering");
  ctx.scrollTop = ctx.scrollHeight;
  await typeInto(message.querySelector(".inner"), context.message, run, 14);
  state.view = { type: "code", narrowed: false };
}

function renderExact(context) {
  const ctx = $("#ctx");
  ctx.textContent = "";
  ctx.append(el("div", "wtag", "<" + context.wrapper + ">"));
  context.files.forEach((file) => {
    const node = fileNode(file);
    node.querySelectorAll(".ln").forEach((row) => row.classList.add("in-scope"));
    ctx.append(node);
  });
  ctx.append(el("div", "wtag", "</" + context.wrapper + ">"), blockNode("message", "<message>" + context.message + "</message>"));
  ctx.scrollTop = 0;
  state.view = { type: "code", narrowed: false };
}

async function narrow(ranges, run) {
  harnessHeader("No model call: the harness keeps only the scoped lines for the next call, with the files' own line numbers.");
  const ctx = $("#ctx"), files = [...ctx.querySelectorAll(".file")], kept = [];
  files.forEach((file) => {
    const spans = ranges[file.dataset.name] || [];
    file.querySelectorAll(".ln").forEach((row) => {
      const number = Number(row.dataset.n);
      if (spans.some(([start, end]) => start <= number && number <= end)) kept.push(row);
    });
  });
  if (kept.length) kept[0].scrollIntoView({ block: "center", behavior: state.skip ? "auto" : "smooth" });
  await wait(650, run);
  for (const row of kept) { row.classList.add("in-scope"); await wait(45, run); }
  await wait(1500, run);
  const keep = new Set(kept);
  files.forEach((file) => {
    if (!ranges[file.dataset.name]) {
      file.style.maxHeight = file.scrollHeight + "px";
      void file.offsetHeight;
      file.classList.add("gone");
    } else {
      file.querySelectorAll(".ln").forEach((row) => { if (!keep.has(row)) row.classList.add("gone"); });
    }
  });
  ctx.querySelectorAll(".wtag").forEach((tag) => {
    tag.textContent = tag.textContent.replace("project", "context");
    animateOnce(tag, "morph");
  });
  await wait(1000, run);
  if (kept.length) kept[0].scrollIntoView({ block: "center", behavior: "auto" });
  state.view = { type: "code", narrowed: true };
}

function harnessHeader(note) {
  const head = $("#p-head"), kind = $("#kind");
  kind.className = "kind";
  kind.textContent = "Harness";
  head.classList.remove("flash");
  void head.offsetWidth;
  head.classList.add("flash");
  $("#note").textContent = note;
  $("#meta").textContent = "";
}

function resetReply(prefix) {
  const pre = $("#reply-text");
  pre.textContent = "";
  if (prefix) pre.append(el("span", "prefix", prefix));
  state.stream = el("span", "stream");
  state.caret = el("span", "caret");
  pre.append(state.stream, state.caret);
  $("#reply-label").textContent = "Gemma's reply · waiting for the model";
}

async function streamReply(event, run) {
  $("#reply-label").textContent = "Gemma's reply · writing";
  const pieces = event.text.match(/\s+|[A-Za-z_]+|\d|[^\sA-Za-z_\d]/g) || [];
  const box = $("#reply");
  let shown = "";
  for (const piece of pieces) {
    shown += piece;
    state.stream.textContent = shown;
    box.scrollTop = box.scrollHeight;
    await wait(24, run);
  }
  state.stream.textContent = event.text;
  state.caret.remove();
  if (event.keep !== undefined && event.keep < event.text.length) {
    state.stream.textContent = event.text.slice(0, event.keep);
    state.stream.after(el("span", "discard", event.text.slice(event.keep)));
    $("#reply-label").textContent = "Gemma's reply · only the first tag is used, the rest is thrown away";
  } else {
    $("#reply-label").textContent = "Gemma's reply · saved from the pass";
  }
  await wait(450, run);
}

/* ---- the answer in the chat -------------------------------------------------------------------- */
function richText(text) {
  const fragment = document.createDocumentFragment();
  text.split(/\n{2,}/).forEach((paragraphText) => {
    const paragraph = el("p");
    paragraphText.split(/(`[^`]+`)/).forEach((part) => {
      if (part.length > 1 && part.startsWith("`") && part.endsWith("`")) paragraph.append(el("code", null, part.slice(1, -1)));
      else paragraph.append(document.createTextNode(part));
    });
    fragment.append(paragraph);
  });
  return fragment;
}

function snippet(file) {
  const box = el("div", "snippet"), lines = file.lines;
  const where = lines.length === 1 ? "line " + lines[0][0] : "lines " + lines[0][0] + "–" + lines[lines.length - 1][0];
  box.append(el("div", "head", file.name + " · " + where));
  const pre = el("pre");
  pre.textContent = lines.map(([number, text]) => String(number).padStart(2, " ") + "  " + text).join("\n");
  box.append(pre);
  return box;
}

function renderAnswer(event) {
  const box = el("div", "a-answer");
  if (event.format === "solution") {
    const card = el("div", "solution-card");
    event.text.replace("<solution>", "").replace("</solution>", "").trim().split("\n")
      .forEach((line) => card.append(el("div", line.startsWith("Answer") ? "ans" : null, line)));
    box.append(card);
  } else if (event.format === "harness") {
    const paragraph = el("p", null, event.text);
    paragraph.append(el("span", "harness-badge", "written by the harness, not the model"));
    box.append(paragraph);
  } else if (event.format === "found") {
    box.append(el("p", null, "Found it:"));
    event.files.forEach((file) => box.append(snippet(file)));
  } else {
    box.append(richText(event.text));
    if (event.files) {
      box.append(el("div", "read-label", "The lines it read:"));
      event.files.forEach((file) => box.append(snippet(file)));
    }
  }
  return box;
}

/* ---- wiring ----------------------------------------------------------------------------------- */
$("#composer").addEventListener("click", (event) => {
  if (event.target.closest("#send") || event.target.closest("#skip")) return;
  typeNext();
});
$("#composer").addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") { event.preventDefault(); if (state.mode === "ready") send(); else typeNext(); }
});
$("#send").addEventListener("click", (event) => { event.stopPropagation(); send(); });
$("#skip").addEventListener("click", (event) => { event.stopPropagation(); state.skip = true; document.body.classList.add("fast"); });
document.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && state.mode === "ready" && document.activeElement !== $("#composer")) send();
  if (event.key === "Escape") { setSidebar(false); if (window.innerWidth <= 1180) setPanel(false); }
});
$("#auto").addEventListener("click", () => {
  state.auto = !state.auto;
  $("#auto").classList.toggle("on", state.auto);
  $("#auto").setAttribute("aria-pressed", String(state.auto));
  if (state.auto && state.mode === "idle") autoNext(state.run);
});
$("#ctx-toggle").addEventListener("click", () => setPanel(!state.panelOpen));
$("#ctx-close").addEventListener("click", () => setPanel(false));
$("#menu").addEventListener("click", () => setSidebar(true));
$("#scrim").addEventListener("click", () => { setSidebar(false); if (window.innerWidth <= 1180) setPanel(false); });
$("#restart").addEventListener("click", () => openChat(state.chat));
$("#theme").addEventListener("click", toggleTheme);
window.addEventListener("resize", updateScrim);

const theme = saved.get("theme", null);
if (theme === "light" || theme === "dark") document.documentElement.dataset.theme = theme;
const hash = new URLSearchParams(location.hash.slice(1));
if (hash.get("theme") === "light" || hash.get("theme") === "dark") document.documentElement.dataset.theme = hash.get("theme");
if (hash.has("speed")) state.speed = Number(hash.get("speed")) || 1;
setSpeed(state.speed);
setPanel(hash.has("panel") ? hash.get("panel") === "1" : state.panelOpen);
state.auto = hash.get("auto") === "1";
$("#auto").classList.toggle("on", state.auto);
openChat(Math.min(Math.max(Number(hash.get("chat")) || 0, 0), DATA.chats.length - 1));
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
