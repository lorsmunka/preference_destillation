"""Pass 1 of Gemma's game-language run on a second rented card through vLLM (2026-11-07: the commands from gpt-oss
roughly tripled the blocks to write; vLLM batches continuously and caches the reference card shared by every
prompt, the framework's model.generate does neither). The same prompt as game_gemma_run.write: the domain's teacher
prompt as the user turn of Gemma's chat template, thinking off, greedy, the same token budget. Pass 2 (scoring)
stays on the framework. Standard library only (the card's Python has vLLM, not this repository's packages).

    vllm serve <gemma dir> --served-model-name google/gemma-4-12B-it ...      (port 8000)
    python experiments/gemma_vllm_run.py runs/game_gemma_run/vllm_prompts.jsonl   -> vllm_written.jsonl next to it
"""

import json
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "google/gemma-4-12B-it"
PARALLEL = 128


def write_one(row):
    body = {"model": MODEL, "messages": [{"role": "user", "content": row["prompt"]}], "max_tokens": row["budget"],
            "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
    request = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    last_error = ""
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                reply = json.loads(response.read())
            return {"id": row["id"], "command": row["command"], "reply": reply["choices"][0]["message"]["content"] or "",
                    "budget": row["budget"], "finish": reply["choices"][0].get("finish_reason"), "writer": "vllm"}
        except Exception as error:      # noqa: BLE001 - retried, then recorded with no reply
            last_error = repr(error)[:300]
            time.sleep(5)
    return {"id": row["id"], "command": row["command"], "reply": "", "budget": row["budget"], "error": last_error}


def main():
    prompts_path = Path(sys.argv[1])
    written_path = prompts_path.with_name("vllm_written.jsonl")
    done = set()
    if written_path.exists():
        done = {json.loads(line)["id"] for line in open(written_path, encoding="utf-8") if json.loads(line)["reply"]}
    todo = [json.loads(line) for line in open(prompts_path, encoding="utf-8")]
    todo = [row for row in todo if row["id"] not in done]
    print(f"{len(todo):,} blocks to write ({len(done):,} before)", flush=True)
    lock, finished, start = threading.Lock(), [0], time.time()

    def work(row):
        result = write_one(row)
        with lock:
            with open(written_path, "a", encoding="utf-8") as file:
                file.write(json.dumps(result, ensure_ascii=False) + "\n")
            finished[0] += 1
            if finished[0] % 200 == 0 or finished[0] == len(todo):
                rate = finished[0] / (time.time() - start)
                print(f"{finished[0]:,}/{len(todo):,} written, {rate:.1f} per s, "
                      f"~{(len(todo) - finished[0]) / rate / 60:.0f} min left", flush=True)

    with ThreadPoolExecutor(PARALLEL) as pool:
        list(pool.map(work, todo))


if __name__ == "__main__":
    main()
