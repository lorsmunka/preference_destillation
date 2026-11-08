"""The first QR Game Studio agent (2026-11-08): a local server the app's editor asks when the user types a `//`
command line. The trained student writes the block (live.py builds its input the way the training problems were
built); the block goes into the game and the app's own compiler and checks run on it (check.check_sources, as for
Gemma's blocks); when it adds a problem the game did not have, the student tries again with sampling, at most
ATTEMPTS times in all, and the first clean block wins (else the greedy one, with its problems listed).

    python -m library.domain.game_dsl.server [run name]        (default: game-agent-kd-m-s1; port 8765)

POST /complete  {"source": the code without the command line, "offset": where that line was, "command": "/insert
..." or "/edit ...", "indent": the line's indentation}
  -> {"ok", "start", "end", "replacement", "problems", "attempts"}: the app replaces source[start:end] with it.
"""

import json
import sys
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import GameDslDomain
from .check import check_sources
from .gemma_writes import HOLLOW, block_of
from .live import edit_input, insert_input

PORT = 8765
ATTEMPTS = 3
SAMPLING_TEMPERATURE = 0.7


def from_utf16(text: str, offset: int) -> int:
    """The app (Dart) counts UTF-16 units, Python code points: they differ after an emoji."""
    return len(text.encode("utf-16-le")[:offset * 2].decode("utf-16-le", errors="ignore"))


def to_utf16(text: str, index: int) -> int:
    return len(text[:index].encode("utf-16-le")) // 2


def indented(block: str, indent: str) -> str:
    """The block at the cursor's indentation (its own common indentation taken off first)."""
    import textwrap
    return "\n".join(indent + line if line.strip() else line for line in textwrap.dedent(block).splitlines())


def new_problems(source: str, changed: str):
    found = check_sources({"before": source, "after": changed})
    added = Counter(found["after"]) - Counter(found["before"])
    return [item for item in sorted(added.elements()) if not item.startswith(HOLLOW)]


class Agent:
    def __init__(self, run_name: str = "game-agent-kd-m-s1"):
        from library import RunStore
        from library.model.student import StudentModel
        self.domain = GameDslDomain("card")
        store = RunStore()
        _, checkpoint_path = store.resolve_checkpoint(run_name, "latest")
        self.student = StudentModel.from_run(store.get(run_name), checkpoint_path, domain=self.domain)
        self.run_name = run_name

    def complete(self, source: str, offset: int, command: str, indent: str = "") -> dict:
        command = command.strip()
        if command.startswith("/edit"):
            found = edit_input(source, offset, command)
            if found is None:
                return {"ok": False, "error": "the cursor is on no Entity or event to edit"}
            text, start, end = found
            first_line = source[start:end].split("\n", 1)[0]
            indent = first_line[:len(first_line) - len(first_line.lstrip(" \t"))]
        else:
            text, start = insert_input(source, offset, command, indent)
            end = start
        best = None
        for attempt in range(1, ATTEMPTS + 1):
            temperature = 0.0 if attempt == 1 else SAMPLING_TEMPERATURE
            reply = self.student.generate(text, max_new_tokens=self.domain.max_steps, temperature=temperature).text
            block = block_of(reply)
            if block is None:
                continue
            replacement = indented(block, indent) + "\n"
            problems = new_problems(source, source[:start] + replacement + source[end:])
            if best is None or len(problems) < len(best["problems"]):
                best = {"ok": True, "start": start, "end": end, "replacement": replacement, "problems": problems,
                        "attempts": attempt}
            if not problems:
                break
        return best or {"ok": False, "error": "no complete block in the student's replies"}


def serve(run_name: str = "game-agent-kd-m-s1", port: int = PORT):
    agent = Agent(run_name)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/complete":
                self.send_error(404)
                return
            try:
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                source = request["source"]
                answer = agent.complete(source, from_utf16(source, int(request["offset"])), request["command"],
                                        request.get("indent", ""))
                if answer.get("ok"):         # back to the app's UTF-16 positions
                    answer["start"], answer["end"] = to_utf16(source, answer["start"]), to_utf16(source, answer["end"])
            except Exception as error:      # noqa: BLE001 - reported to the app, the server stays up
                answer = {"ok": False, "error": f"{type(error).__name__}: {error}"}
            body = json.dumps(answer).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            print(f"{request.get('command', '?')[:60]!r} -> {answer.get('attempts')} attempt(s), "
                  f"{len(answer.get('problems', []))} problem(s)", flush=True)

    print(f"game agent ({run_name}) on http://127.0.0.1:{port}/complete", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    serve(sys.argv[1] if len(sys.argv) > 1 else "game-agent-kd-m-s1")
