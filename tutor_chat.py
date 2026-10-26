"""The math-tutor student, live, in a local chat page: pick a held-out test problem (or type your own),
then follow-ups. Each reply shows the harness's steps (the route call, the solution or explanation
call, the check, a second try), and the context panel shows exactly what the student read in each
call. A test problem also shows the right answer and Gemma's reply to the same message in its own
saved conversation (full_v1.jsonl, the problem families no student trained on).

    python tutor_chat.py                          # the best student (math-full-kd-l-s1), http://127.0.0.1:8765
    python tutor_chat.py math-full-kd-s1          # another trained run
    python tutor_chat.py math-full-kd-l-s1 8800   # another port

Local only: the server listens on 127.0.0.1 and nothing is uploaded. The student runs on the GPU
when there is one (~1 s a reply on the 4070); Ctrl+C stops the server and frees the GPU.
"""

import csv
import json
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from library.domain.math_tutor.conversation import Conversation
from library.domain.math_tutor.demo import DEFAULT_RUN, load_student, take_turn

ROOT = Path(__file__).resolve().parent
TEST_CONVERSATIONS = ROOT / "library" / "domain" / "math_tutor" / "items" / "conversations" / "full_v1.jsonl"
RESULTS = ROOT / "experiments" / "results" / "math_full.csv"
PORT = 8765


def test_problems():
    """The saved test conversations: each pupil message with the route and answer it expects, and
    Gemma's reply in that conversation."""
    problems = []
    for line in open(TEST_CONVERSATIONS, encoding="utf-8"):
        conversation = json.loads(line)
        if conversation["split"] != "test":
            continue
        turns = [{"event": turn["event"], "message": turn["message"], "route": turn.get("expected_route"),
                  "answer": turn.get("expected_answer"), "gemma": turn.get("output") or "",
                  "harness": turn.get("answered_by") == "harness"}
                 for turn in conversation["turns"] if turn.get("message")]
        problems.append({"id": conversation["item_id"], "turns": turns})
    return problems


def run_summary(run):
    """Name, size and test accuracy of the student, for the page's header."""
    accuracy = None
    if RESULTS.exists():
        for row in csv.DictReader(open(RESULTS, encoding="utf-8")):
            if row["run"] == run.name:
                accuracy, baseline = float(row["task_accuracy_final"]), float(row["majority_baseline"])
    summary = {"name": run.name, "parameters": run.info["model_info"]["total_parameters"]}
    if accuracy is not None:
        summary.update(accuracy=accuracy, baseline=baseline)
    return summary


class TutorServer:
    """One student and one conversation per chat id (a page's chat); one reply at a time on the GPU."""

    def __init__(self, model):
        self.model = model
        self.conversations = {}
        self.lock = threading.Lock()

    def turn(self, chat_id: str, message: str):
        with self.lock:
            conversation = self.conversations.setdefault(chat_id, Conversation())
            start = time.time()
            turn = take_turn(self.model, conversation, message)
            turn["seconds"] = round(time.time() - start, 2)
            return turn

    def reset(self, chat_id: str):
        with self.lock:
            self.conversations[chat_id] = Conversation()


def make_handler(server: TutorServer, page: bytes, port: int):
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, content_type, body: bytes):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, status, data):
            self.send(status, "application/json; charset=utf-8", json.dumps(data, ensure_ascii=False).encode("utf-8"))

        def local_request(self) -> bool:
            if self.headers.get("Host") in allowed_hosts:
                return True
            self.send(403, "text/plain; charset=utf-8", b"local page only")
            return False

        def do_GET(self):
            if not self.local_request():
                return
            if self.path in ("/", "/index.html"):
                self.send(200, "text/html; charset=utf-8", page)
            else:
                self.send(404, "text/plain; charset=utf-8", b"not found")

        def do_POST(self):
            if not self.local_request():
                return
            request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            try:
                if self.path == "/api/turn":
                    self.send_json(200, server.turn(str(request["chat"]), str(request["message"])))
                elif self.path == "/api/reset":
                    server.reset(str(request["chat"]))
                    self.send_json(200, {"ok": True})
                else:
                    self.send_json(404, {"error": "not found"})
            except Exception as error:   # shown in the page, so a failed reply does not look like a hang
                self.send_json(500, {"error": f"{type(error).__name__}: {error}"})

        def log_message(self, format, *args):
            pass

    return Handler


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    run_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RUN
    port = int(sys.argv[2]) if len(sys.argv) > 2 else PORT
    run, model = load_student(run_name)
    data = {"run": run_summary(run), "problems": test_problems()}
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    page = TEMPLATE.replace("__DARK__", DARK).replace("__DATA__", payload).encode("utf-8")
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(TutorServer(model), page, port))
    url = f"http://127.0.0.1:{port}/"
    print(f"\n{run.name} on {model.student.device}, {len(data['problems'])} test problems -> {url}   (Ctrl+C stops)")
    webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("stopped")
    finally:
        httpd.server_close()


DARK = """
    --bg: #212121; --sidebar: #171717; --panel: #1b1b1b; --text: #ececec; --muted: #b4b4b4; --faint: #8b8b8b;
    --bubble: #303030; --border: #2f2f2f; --composer: #303030; --composer-border: #3a3a3a; --hover: #2a2a2a; --active: #343434;
    --code-bg: #151515; --send: #ececec; --send-text: #0d0d0d; --kind-text: #0d0d0d;
    --task: #60a5fa; --solution: #34d399; --question: #fbbf24; --explanation: #a78bfa; --message: #f472b6;
    --pass: #34d399; --fail: #f87171; --route: #60a5fa; --info: #9ca3af;
    --shadow: 0 0 0 1px rgba(255, 255, 255, .03), 0 10px 30px rgba(0, 0, 0, .45);
    color-scheme: dark;
"""

TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tutor Chat</title>
<style>
:root {
  --bg: #ffffff; --sidebar: #f9f9f9; --panel: #fafafa; --text: #0d0d0d; --muted: #5d5d5d; --faint: #8f8f8f;
  --bubble: #f1f1f1; --border: #e6e6e6; --composer: #ffffff; --composer-border: #dcdcdc; --hover: #ececec; --active: #e3e3e3;
  --code-bg: #f5f5f6; --accent: #10a37f; --send: #0d0d0d; --send-text: #ffffff; --kind-text: #ffffff;
  --task: #2563eb; --solution: #047857; --question: #b45309; --explanation: #7c3aed; --message: #be185d;
  --pass: #047857; --fail: #dc2626; --route: #2563eb; --info: #6b7280;
  --shadow: 0 0 0 1px rgba(0, 0, 0, .03), 0 8px 26px rgba(0, 0, 0, .08);
  --mono: ui-monospace, "SF Mono", "Cascadia Code", Menlo, Consolas, monospace;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { __DARK__ } }
:root[data-theme="dark"] { __DARK__ }
* { box-sizing: border-box; }
html, body { height: 100%; margin: 0; }
body { background: var(--bg); color: var(--text); overflow: hidden; -webkit-font-smoothing: antialiased;
  font: 15px/1.6 ui-sans-serif, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
button, input, textarea { font: inherit; color: inherit; }
.app { display: grid; grid-template-columns: 280px minmax(0, 1fr) 0; height: 100vh; height: 100dvh; }
.app.panel-open { grid-template-columns: 280px minmax(0, 1fr) minmax(360px, 40vw); }

/* sidebar: the test problems */
.sidebar { background: var(--sidebar); display: flex; flex-direction: column; padding: 10px 8px; min-width: 0; min-height: 0; }
.brand { font-weight: 600; display: flex; align-items: center; gap: 9px; padding: 4px 8px 10px; }
.logo { width: 26px; height: 26px; border-radius: 50%; background: var(--text); color: var(--bg); display: grid; place-items: center; font-size: 14px; }
.new-chat { display: flex; align-items: center; gap: 8px; width: 100%; border: 0; background: transparent; padding: 8px 10px; border-radius: 8px;
  cursor: pointer; font-size: 14px; text-align: left; }
.new-chat:hover { background: var(--hover); }
.search { margin: 8px 4px 4px; padding: 7px 10px; border: 1px solid var(--border); border-radius: 8px; background: var(--bg); font-size: 13.5px; }
.side-group { margin: 12px 0 4px; padding: 0 10px; font-size: 12px; color: var(--faint); font-weight: 500; }
.problem-list { overflow-y: auto; min-height: 0; flex: 1; }
.side-item { display: block; width: 100%; text-align: left; border: 0; background: transparent; padding: 7px 10px; border-radius: 8px; cursor: pointer; font-size: 13.5px; line-height: 1.4; }
.side-item:hover { background: var(--hover); }
.side-item.active { background: var(--active); }
.side-item small { display: block; color: var(--faint); font-size: 11.5px; }
.side-foot { padding: 10px 10px 2px; font-size: 12px; color: var(--faint); line-height: 1.5; border-top: 1px solid var(--border); }

/* main column */
.main { display: flex; flex-direction: column; min-width: 0; min-height: 0; }
.topbar { height: 56px; display: flex; align-items: center; gap: 8px; padding: 0 12px 0 16px; flex: none; }
.title { font-size: 16px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; min-width: 0; }
.model-pill { font-size: 12px; color: var(--muted); border: 1px solid var(--border); border-radius: 999px; padding: 2px 10px; white-space: nowrap; }
.spacer { flex: 1; }
.pill-btn { border: 1px solid var(--border); background: transparent; border-radius: 999px; padding: 6px 12px; font-size: 13px; cursor: pointer; white-space: nowrap; }
.pill-btn:hover { background: var(--hover); }
.pill-btn.on { background: var(--active); border-color: var(--active); }
.chat-scroll { flex: 1; min-height: 0; overflow-y: auto; }
.chat { max-width: 780px; margin: 0 auto; padding: 4px 24px 24px; }
.msg { margin: 22px 0; display: flex; }
.msg.user { justify-content: flex-end; }
.msg.user .bubble { background: var(--bubble); border-radius: 18px; padding: 9px 16px; max-width: 78%; white-space: pre-wrap; }
.msg.assistant { gap: 14px; align-items: flex-start; }
.avatar { width: 30px; height: 30px; border-radius: 50%; border: 1px solid var(--border); display: grid; place-items: center; flex: none;
  font: 600 11px/1 var(--mono); color: var(--muted); }
.a-body { flex: 1; min-width: 0; padding-top: 3px; }
.steps { border-left: 2px solid var(--border); margin: 2px 0 12px; padding: 2px 0 2px 12px; display: flex; flex-direction: column; gap: 4px; }
.step { font-size: 13px; color: var(--muted); line-height: 1.5; display: flex; gap: 8px; border-radius: 6px; padding: 1px 4px; margin-left: -4px; }
.step.clickable { cursor: pointer; }
.step.clickable:hover { background: var(--hover); }
.step.selected { background: var(--active); }
.step .ic { flex: none; width: 14px; text-align: center; font-weight: 700; }
.step b { color: var(--text); font-weight: 600; }
.step.pass .ic { color: var(--pass); } .step.fail .ic { color: var(--fail); } .step.route .ic { color: var(--route); } .step.call .ic { color: var(--accent); }
.step.info .ic { color: var(--info); } .step.fail b { color: var(--fail); }
.step code { font-family: var(--mono); font-size: 12px; }
.thinking { display: flex; align-items: center; gap: 10px; color: var(--muted); font-size: 14px; }
.thinking .dot { width: 11px; height: 11px; border-radius: 50%; background: var(--text); animation: pulse 1.1s infinite ease-in-out; }
.solution-card { background: var(--code-bg); border: 1px solid var(--border); border-radius: 12px; padding: 10px 14px; font: 13.5px/1.65 var(--mono); white-space: pre-wrap; }
.solution-card .ans { font-weight: 700; color: var(--solution); }
.explanation-text { margin: 0; white-space: pre-wrap; }
.harness-badge { display: inline-block; font-size: 11px; font-weight: 600; color: var(--info); border: 1px solid var(--border); border-radius: 999px; padding: 0 8px; margin-left: 8px; }
.error { color: var(--fail); font-size: 14px; }
.grades { display: flex; flex-wrap: wrap; gap: 6px 14px; font-size: 12.5px; color: var(--muted); margin-top: 8px; }
.grade.ok::before { content: "\2713  "; color: var(--pass); font-weight: 700; }
.grade.bad::before { content: "\2715  "; color: var(--fail); font-weight: 700; }
details.gemma { margin-top: 6px; font-size: 13px; color: var(--muted); }
details.gemma summary { cursor: pointer; }
details.gemma pre { background: var(--code-bg); border: 1px solid var(--border); border-radius: 10px; padding: 8px 12px; font: 12.5px/1.55 var(--mono);
  white-space: pre-wrap; color: var(--text); margin: 6px 0 0; }
.seconds { font-size: 12px; color: var(--faint); margin-top: 6px; }

/* empty state */
.empty { max-width: 640px; margin: 8vh auto 0; text-align: center; padding: 0 16px; }
.empty h1 { font-size: 26px; font-weight: 600; margin: 0 0 12px; line-height: 1.25; }
.empty p { color: var(--muted); margin: 0 auto 14px; }
.starters { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 8px; margin-top: 18px; text-align: left; }
.starter { border: 1px solid var(--border); background: transparent; border-radius: 14px; padding: 10px 14px; cursor: pointer; font-size: 13.5px; color: var(--muted); text-align: left; }
.starter:hover { background: var(--hover); color: var(--text); }
.problem-card { border: 1px solid var(--border); border-radius: 14px; padding: 12px 16px; margin: 18px 0 0; font-size: 14px; color: var(--muted); }
.problem-card b { color: var(--text); }

/* composer */
.composer-wrap { padding: 0 24px 12px; flex: none; }
.suggest { max-width: 780px; margin: 0 auto 8px; display: flex; gap: 6px; align-items: center; flex-wrap: wrap; font-size: 12.5px; color: var(--faint); }
.chip { border: 1px solid var(--border); background: var(--bg); border-radius: 999px; padding: 4px 12px; font-size: 13px; cursor: pointer; color: var(--text);
  max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; text-align: left; }
.chip:hover { background: var(--hover); }
.chip .kind-word { color: var(--faint); margin-right: 6px; }
.link-btn { border: 0; background: none; color: var(--faint); cursor: pointer; font-size: 12.5px; padding: 0 4px; text-decoration: underline; }
.composer { max-width: 780px; margin: 0 auto; background: var(--composer); border: 1px solid var(--composer-border); border-radius: 26px;
  box-shadow: var(--shadow); display: flex; align-items: flex-end; gap: 10px; padding: 10px 10px 10px 20px; }
.composer:focus-within { border-color: var(--faint); }
.composer textarea { flex: 1; border: 0; outline: 0; background: transparent; resize: none; min-height: 28px; max-height: 180px; padding: 4px 0; line-height: 1.5; }
.send { width: 36px; height: 36px; border-radius: 50%; border: 0; background: var(--send); color: var(--send-text); display: grid; place-items: center; cursor: pointer; flex: none; }
.send:disabled { opacity: .22; cursor: default; }
.footnote { text-align: center; font-size: 12px; color: var(--faint); padding-top: 8px; }

/* the context panel */
.panel { background: var(--panel); border-left: 1px solid var(--border); display: none; flex-direction: column; min-width: 0; min-height: 0; }
.app.panel-open .panel { display: flex; }
.p-head { padding: 12px 16px; border-bottom: 1px solid var(--border); flex: none; }
.p-title { display: flex; align-items: center; justify-content: space-between; font-weight: 600; }
.call-tabs { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 8px; }
.call-tab { border: 1px solid var(--border); background: transparent; border-radius: 999px; padding: 2px 10px; font-size: 12px; cursor: pointer; color: var(--muted); }
.call-tab.on { border-color: var(--accent); color: var(--text); background: var(--hover); }
.note { font-size: 13px; color: var(--muted); line-height: 1.45; margin-top: 8px; }
.p-body { overflow-y: auto; padding: 12px 16px 24px; flex: 1; min-height: 0; }
.p-label { font-size: 12px; font-weight: 600; color: var(--faint); text-transform: uppercase; letter-spacing: .05em; margin: 10px 0 6px; }
.block { border: 1px solid var(--border); border-left: 3px solid var(--info); border-radius: 8px; padding: 6px 10px; margin: 0 0 6px; background: var(--bg); }
.block .tag { font: 700 11px/1.4 var(--mono); text-transform: uppercase; letter-spacing: .04em; }
.block pre { margin: 2px 0 0; font: 12.5px/1.55 var(--mono); white-space: pre-wrap; word-break: break-word; }
.block.task { border-left-color: var(--task); } .block.task .tag { color: var(--task); }
.block.solution { border-left-color: var(--solution); } .block.solution .tag { color: var(--solution); }
.block.question { border-left-color: var(--question); } .block.question .tag { color: var(--question); }
.block.explanation { border-left-color: var(--explanation); } .block.explanation .tag { color: var(--explanation); }
.block.message { border-left-color: var(--message); } .block.message .tag { color: var(--message); }
.raw { background: var(--code-bg); border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; font: 12.5px/1.55 var(--mono); white-space: pre-wrap; word-break: break-word; margin: 0; }
.p-empty { color: var(--faint); font-size: 13.5px; }
@keyframes pulse { 0%, 100% { transform: scale(.75); opacity: .5; } 50% { transform: scale(1); opacity: 1; } }
@media (max-width: 860px) {
  .app, .app.panel-open { grid-template-columns: minmax(0, 1fr); }
  .sidebar, .panel, .app.panel-open .panel { display: none; }
  .chat, .composer-wrap { padding-left: 16px; padding-right: 16px; }
}
</style>
</head>
<body>
<div class="app" id="app">
  <aside class="sidebar">
    <div class="brand"><div class="logo">&#8721;</div>Tutor Chat</div>
    <button class="new-chat" id="newChat">&#9998;&nbsp; New chat (your own problem)</button>
    <input class="search" id="search" placeholder="Search the test problems" autocomplete="off">
    <div class="side-group" id="problemCount">Held-out test problems</div>
    <div class="problem-list" id="problemList"></div>
    <div class="side-foot" id="sideFoot"></div>
  </aside>
  <main class="main">
    <div class="topbar">
      <div class="title" id="title">New chat</div>
      <div class="model-pill" id="modelPill"></div>
      <div class="spacer"></div>
      <button class="pill-btn" id="themeBtn" title="Light or dark">&#9680;</button>
      <button class="pill-btn" id="panelBtn">Context</button>
    </div>
    <div class="chat-scroll" id="scroll"><div class="chat" id="chat"></div></div>
    <div class="composer-wrap">
      <div class="suggest" id="suggest"></div>
      <div class="composer">
        <textarea id="input" rows="1" placeholder="Type a word problem, a what-if, or a question about the working"></textarea>
        <button class="send" id="send" disabled title="Send (Enter)">&#8593;</button>
      </div>
      <div class="footnote">The student runs on this computer. The harness routes each message, checks the reply and asks again once.</div>
    </div>
  </main>
  <aside class="panel" id="panel">
    <div class="p-head">
      <div class="p-title">What the student read <button class="pill-btn" id="closePanel">Close</button></div>
      <div class="call-tabs" id="callTabs"></div>
      <div class="note" id="callNote"></div>
    </div>
    <div class="p-body" id="panelBody"><div class="p-empty">Send a message, then click a step to see the exact text of that call.</div></div>
  </aside>
</div>
<script>
const DATA = __DATA__;
const ROUTE_WORDS = { solution: "a solution", explanation: "an explanation", out_of_scope: "out of scope", unclear: "no tag (treated as a task)" };
const BLOCK = /<(task|solution|question|explanation|message)>([\s\S]*?)<\/\1>/g;
const $ = id => document.getElementById(id);
let chat = null;          // { id, problem, next, turns: [...] }
let busy = false;
let selected = null;      // { turnIndex, callIndex }

function escapeHtml(text) {
  return String(text).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
function finalAnswer(text) {
  const match = /Answer:\s*(-?\d+)/.exec(text || "");
  return match ? Number(match[1]) : null;
}
function shortTitle(problem) {
  const words = problem.turns[0].message.split(/\s+/);
  return words.slice(0, 9).join(" ") + (words.length > 9 ? "…" : "");
}

/* sidebar */
function renderProblems() {
  const query = $("search").value.trim().toLowerCase();
  const list = $("problemList");
  list.innerHTML = "";
  const shown = DATA.problems.filter(p => !query || p.turns.some(t => t.message.toLowerCase().includes(query)));
  $("problemCount").textContent = `Held-out test problems (${shown.length})`;
  for (const problem of shown) {
    const button = document.createElement("button");
    button.className = "side-item" + (chat && chat.problem === problem ? " active" : "");
    const tasks = problem.turns.filter(t => t.route === "solution").length;
    button.innerHTML = `${escapeHtml(shortTitle(problem))}<small>${problem.turns.length} messages, ${tasks} tasks · answer ${problem.turns[0].answer}</small>`;
    button.onclick = () => startChat(problem);
    list.appendChild(button);
  }
}

/* chats */
async function startChat(problem) {
  if (busy) return;
  chat = { id: Math.random().toString(36).slice(2), problem, next: 0, turns: [] };
  selected = null;
  await fetch("/api/reset", { method: "POST", body: JSON.stringify({ chat: chat.id }) });
  $("title").textContent = problem ? shortTitle(problem) : "New chat";
  renderProblems();
  renderChat();
  renderPanel();
  if (problem) setInput(problem.turns[0].message);
  $("input").focus();
}

function savedTurn() {
  return chat && chat.problem && chat.next < chat.problem.turns.length ? chat.problem.turns[chat.next] : null;
}

function renderSuggest() {
  const box = $("suggest");
  box.innerHTML = "";
  const saved = savedTurn();
  if (!saved) {
    if (chat && chat.problem) box.textContent = "End of the saved conversation: type your own follow-ups.";
    return;
  }
  const label = document.createElement("span");
  label.textContent = "Next in the saved conversation:";
  const chip = document.createElement("button");
  chip.className = "chip";
  chip.innerHTML = `<span class="kind-word">${escapeHtml(saved.event.replace("_", "-"))}</span>${escapeHtml(saved.message)}`;
  chip.title = saved.message;
  chip.onclick = () => setInput(saved.message);
  const skip = document.createElement("button");
  skip.className = "link-btn";
  skip.textContent = "skip";
  skip.onclick = () => { chat.next += 1; renderSuggest(); };
  box.append(label, chip, skip);
}

function setInput(text) {
  const input = $("input");
  input.value = text;
  autoSize();
  $("send").disabled = busy || !input.value.trim();
  input.focus();
}
function autoSize() {
  const input = $("input");
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 180) + "px";
}

async function send() {
  const message = $("input").value.trim();
  if (!message || busy) return;
  if (!chat) await startChat(null);
  const saved = savedTurn();
  const matches = saved && saved.message.trim() === message;
  const turn = { message, saved: matches ? saved : null, result: null, error: null };
  if (matches) chat.next += 1;
  chat.turns.push(turn);
  setInput("");
  busy = true;
  renderChat();
  try {
    const response = await fetch("/api/turn", { method: "POST", body: JSON.stringify({ chat: chat.id, message }) });
    const reply = await response.json();
    if (!response.ok) throw new Error(reply.error || response.statusText);
    turn.result = reply;
    selected = { turnIndex: chat.turns.length - 1, callIndex: reply.calls.length - 1 };
  } catch (error) {
    turn.error = String(error.message || error);
  }
  busy = false;
  renderChat();
  renderPanel();
  $("send").disabled = !$("input").value.trim();
}

/* the conversation */
function renderChat() {
  const box = $("chat");
  box.innerHTML = "";
  if (!chat || !chat.turns.length) {
    box.appendChild(emptyState());
  }
  if (chat) chat.turns.forEach((turn, index) => {
    box.insertAdjacentHTML("beforeend", `<div class="msg user"><div class="bubble">${escapeHtml(turn.message)}</div></div>`);
    box.appendChild(assistantMessage(turn, index));
  });
  renderSuggest();
  const scroll = $("scroll");
  scroll.scrollTop = scroll.scrollHeight;
}

function emptyState() {
  const run = DATA.run;
  const div = document.createElement("div");
  div.className = "empty";
  const accuracy = run.accuracy != null
    ? ` On the held-out test it gets <b>${(100 * run.accuracy).toFixed(1)}%</b> of the answers right (always giving the most common answer gets ${(100 * run.baseline).toFixed(1)}%).`
    : "";
  if (chat && chat.problem) {
    const first = chat.problem.turns[0];
    div.innerHTML = `<h1>${escapeHtml(shortTitle(chat.problem))}</h1>
      <p>A problem from a family no student trained on. The first message is in the box: send it, then follow the saved conversation or type your own follow-ups.</p>
      <div class="problem-card"><b>Right answer:</b> ${first.answer} · <b>Gemma</b> answered ${finalAnswer(first.gemma) ?? "–"} in its own conversation.</div>`;
    return div;
  }
  div.innerHTML = `<h1>Talk to the ${(run.parameters / 1e6).toFixed(0)}M-parameter student</h1>
    <p>A small transformer trained from scratch on Gemma 4 12B's tutoring replies.${accuracy}</p>
    <p>Pick a held-out test problem on the left, or type your own word problem with whole numbers.</p>`;
  const starters = document.createElement("div");
  starters.className = "starters";
  const picks = [...DATA.problems].sort(() => Math.random() - .5).slice(0, 4);
  for (const problem of picks) {
    const button = document.createElement("button");
    button.className = "starter";
    button.textContent = problem.turns[0].message;
    button.onclick = () => startChat(problem);
    starters.appendChild(button);
  }
  div.appendChild(starters);
  return div;
}

function stepRow(kind, icon, html, turnIndex, callIndex) {
  const clickable = callIndex != null;
  const isSelected = clickable && selected && selected.turnIndex === turnIndex && selected.callIndex === callIndex;
  return `<div class="step ${kind}${clickable ? " clickable" : ""}${isSelected ? " selected" : ""}"
    ${clickable ? `data-turn="${turnIndex}" data-call="${callIndex}"` : ""}><span class="ic">${icon}</span><span>${html}</span></div>`;
}

function stepsHtml(result, turnIndex) {
  const rows = [];
  result.calls.forEach((call, callIndex) => {
    if (call.kind === "route") {
      const tag = /<(solution|explanation|out_of_scope)/.exec(call.reply);
      rows.push(stepRow("route", "&#8594;", `<b>Route call</b>: the student starts with ${tag ? `<code>&lt;${tag[1]}&gt;</code>` : "no tag"}, so the harness reads it as <b>${ROUTE_WORDS[result.route]}</b>`, turnIndex, callIndex));
      return;
    }
    const retry = call.temperature > 0;
    const name = call.kind === "solution" ? "Solution call" : "Explanation call";
    rows.push(stepRow("call", "&#9679;", `<b>${name}</b>${retry ? " again, sampled (temperature 0.7)" : ""}`, turnIndex, callIndex));
    rows.push(call.problem
      ? stepRow("fail", "&#10005;", `<b>Check failed</b>: ${escapeHtml(call.problem)}`, turnIndex, callIndex)
      : stepRow("pass", "&#10003;", call.kind === "solution" ? "Check passed: well formed, every calculation true" : "Check passed", turnIndex, callIndex));
  });
  if (result.route === "out_of_scope") {
    rows.push(stepRow("info", "i", "The harness answers with its fixed sentence; the message does not enter the student's history"));
  } else if (result.problem) {
    rows.push(stepRow("info", "i", "Still failing after the second try: the harness keeps the reply as it is"));
  }
  return `<div class="steps">${rows.join("")}</div>`;
}

function replyHtml(result) {
  if (result.route === "out_of_scope") {
    return `<p class="explanation-text">${escapeHtml(result.reply)}<span class="harness-badge">harness</span></p>`;
  }
  if (result.reply.startsWith("<solution>")) {
    const body = result.reply.replace("<solution>", "").replace("</solution>", "").trim();
    const lines = body.split("\n").map(line => /^Answer:/.test(line.trim())
      ? `<span class="ans">${escapeHtml(line)}</span>` : escapeHtml(line));
    return `<div class="solution-card">${lines.join("\n")}</div>`;
  }
  const text = result.reply.replace("<explanation>", "").replace("</explanation>", "").trim();
  return `<p class="explanation-text">${escapeHtml(text)}</p>`;
}

function gradesHtml(turn) {
  const saved = turn.saved, result = turn.result;
  if (!saved) return "";
  const grades = [];
  const routeRight = result.route === saved.route || (saved.route === "solution" && result.route === "unclear");
  grades.push(`<span class="grade ${routeRight ? "ok" : "bad"}">route ${routeRight ? "right" : `wrong (expected ${ROUTE_WORDS[saved.route]})`}</span>`);
  if (saved.route === "solution" && saved.answer != null) {
    const answer = finalAnswer(result.reply);
    const right = answer === saved.answer;
    grades.push(`<span class="grade ${right ? "ok" : "bad"}">answer ${answer ?? "missing"}${right ? ", right" : `, the right one is ${saved.answer}`}</span>`);
  }
  const gemmaWord = saved.harness ? "The harness's reply in Gemma's saved conversation" : "Gemma's reply to this message in its own saved conversation";
  return `<div class="grades">${grades.join("")}</div>
    <details class="gemma"><summary>${gemmaWord}</summary><pre>${escapeHtml(saved.gemma)}</pre></details>`;
}

function assistantMessage(turn, turnIndex) {
  const div = document.createElement("div");
  div.className = "msg assistant";
  let body;
  if (turn.error) body = `<p class="error">The reply failed: ${escapeHtml(turn.error)}</p>`;
  else if (!turn.result) body = `<div class="thinking"><span class="dot"></span>The student is writing…</div>`;
  else body = stepsHtml(turn.result, turnIndex) + replyHtml(turn.result) + gradesHtml(turn) + `<div class="seconds">${turn.result.seconds} s</div>`;
  div.innerHTML = `<div class="avatar">${(DATA.run.parameters / 1e6).toFixed(0)}M</div><div class="a-body">${body}</div>`;
  div.querySelectorAll(".step.clickable").forEach(row => row.onclick = () => {
    selected = { turnIndex: Number(row.dataset.turn), callIndex: Number(row.dataset.call) };
    $("app").classList.add("panel-open");
    $("panelBtn").classList.add("on");
    renderChat();
    renderPanel();
  });
  return div;
}

/* the context panel */
function contextHtml(context) {
  let html = "", last = 0, match;
  BLOCK.lastIndex = 0;
  while ((match = BLOCK.exec(context))) {
    const between = context.slice(last, match.index).trim();
    if (between) html += `<pre class="raw">${escapeHtml(between)}</pre>`;
    html += `<div class="block ${match[1]}"><div class="tag">&lt;${match[1]}&gt;</div><pre>${escapeHtml(match[2].trim())}</pre></div>`;
    last = match.index + match[0].length;
  }
  const rest = context.slice(last).trim();
  if (rest) html += `<pre class="raw">${escapeHtml(rest)}</pre>`;
  return html;
}

const CALL_NOTES = {
  route: "The whole history with the side thread, then the new message. Only the start of the reply is read: its first tag is the route.",
  solution: "The main line only (tasks and solutions; the side thread is left out), then the new message.",
  explanation: "The main line and the side thread, then the question.",
};

function renderPanel() {
  const tabs = $("callTabs"), body = $("panelBody"), note = $("callNote");
  tabs.innerHTML = ""; note.textContent = "";
  const turn = selected && chat && chat.turns[selected.turnIndex];
  if (!turn || !turn.result) {
    body.innerHTML = `<div class="p-empty">Send a message, then click a step to see the exact text of that call.</div>`;
    return;
  }
  turn.result.calls.forEach((call, callIndex) => {
    const button = document.createElement("button");
    button.className = "call-tab" + (callIndex === selected.callIndex ? " on" : "");
    button.textContent = call.kind + (call.temperature > 0 ? " (second try)" : "");
    button.onclick = () => { selected.callIndex = callIndex; renderChat(); renderPanel(); };
    tabs.appendChild(button);
  });
  const call = turn.result.calls[selected.callIndex];
  note.textContent = CALL_NOTES[call.kind] + (call.temperature > 0 ? " Asked again with sampling (temperature 0.7)." : " Greedy (temperature 0).");
  body.innerHTML = `<div class="p-label">The student's input</div>${contextHtml(call.context)}
    <div class="p-label">What it wrote</div><pre class="raw">${escapeHtml(call.reply)}</pre>
    ${call.problem ? `<div class="p-label">The check</div><pre class="raw">${escapeHtml(call.problem)}</pre>` : ""}`;
}

/* wiring */
const run = DATA.run;
$("modelPill").textContent = `${run.name} · ${(run.parameters / 1e6).toFixed(1)}M` + (run.accuracy != null ? ` · ${(100 * run.accuracy).toFixed(1)}% on the test` : "");
$("sideFoot").innerHTML = `Test problems come from problem families no student trained on. The right answers come from each problem's code.`;
$("search").oninput = renderProblems;
$("newChat").onclick = () => startChat(null);
$("input").oninput = () => { autoSize(); $("send").disabled = busy || !$("input").value.trim(); };
$("input").onkeydown = event => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); send(); } };
$("send").onclick = send;
$("panelBtn").onclick = () => { $("app").classList.toggle("panel-open"); $("panelBtn").classList.toggle("on"); };
$("closePanel").onclick = () => { $("app").classList.remove("panel-open"); $("panelBtn").classList.remove("on"); };
$("themeBtn").onclick = () => {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
};
renderProblems();
renderChat();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
