"""On the rented card (library/domain/game_dsl/open_commands.py): every task in a tasks file sent to the vLLM server
(`vllm serve openai/gpt-oss-120b`, OpenAI-style API on port 8000), PARALLEL requests at a time; the replies are
appended to <tasks>_answers.jsonl (resumes: tasks already answered are skipped). Standard library only.

    python experiments/open_commands_run.py runs/open_commands/test_tasks.jsonl [effort: low | medium | high]
"""

import json
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "openai/gpt-oss-120b"
PARALLEL = 192


def ask(system, task, effort):
    body = {"model": MODEL, "messages": [{"role": "system", "content": system}, {"role": "user", "content": task["user"]}],
            "max_tokens": 3000, "temperature": 0.6, "reasoning_effort": effort}
    request = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                reply = json.loads(response.read())
            message = reply["choices"][0]["message"]
            return {"id": task["id"], "command": message.get("content") or "", "effort": effort,
                    "reasoning_chars": len(message.get("reasoning_content") or message.get("reasoning") or ""),
                    "prompt_tokens": reply["usage"]["prompt_tokens"],
                    "completion_tokens": reply["usage"]["completion_tokens"],
                    "finish": reply["choices"][0].get("finish_reason")}
        except Exception as error:      # noqa: BLE001 - a failed request is retried, then recorded
            last_error = repr(error)[:300]
            time.sleep(5)
    return {"id": task["id"], "command": "", "effort": effort, "error": last_error}


def main():
    tasks_path = Path(sys.argv[1])
    effort = sys.argv[2] if len(sys.argv) > 2 else "low"
    system = (tasks_path.parent / "system.txt").read_text(encoding="utf-8")
    answers_path = tasks_path.with_name(tasks_path.stem + "_answers.jsonl")
    done = set()
    if answers_path.exists():
        done = {json.loads(line)["id"] for line in open(answers_path, encoding="utf-8")
                if json.loads(line).get("command")}
    todo = [json.loads(line) for line in open(tasks_path, encoding="utf-8")]
    todo = [task for task in todo if task["id"] not in done]
    print(f"{len(todo):,} tasks to ask ({len(done):,} answered before), effort {effort}", flush=True)
    lock, finished, start = threading.Lock(), [0], time.time()

    def work(task):
        row = ask(system, task, effort)
        with lock:
            with open(answers_path, "a", encoding="utf-8") as file:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
            finished[0] += 1
            if finished[0] % 100 == 0 or finished[0] == len(todo):
                rate = finished[0] / (time.time() - start)
                print(f"{finished[0]:,}/{len(todo):,} answered, {rate:.1f} per s, "
                      f"~{(len(todo) - finished[0]) / rate / 60:.0f} min left", flush=True)

    with ThreadPoolExecutor(PARALLEL) as pool:
        list(pool.map(work, todo))


if __name__ == "__main__":
    main()
