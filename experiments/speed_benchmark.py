"""How much faster the distilled students are than their teachers on real hardware (thesis 9.6; user,
2026-11-09: "Measure actual speedtest on both the PC, and cloud rented unit, how much faster is the distilled
model on actual hardware ... math, sentiment, and maybe another one ... Maybe a speedup curve?").

What is timed: one model reply to one test input, batch 1, greedy, from the input text to the reply text
(tokenising, the model, decoding and the stop rule all inside the clock; torch.cuda.synchronize at both ends).
The inputs are the attention study's fixed test records (runs/attention_study/probes.pt, never trained on):
  math       80 replies of the math tutor (32 solutions, 32 explanations, 16 routing tags), Gemma 4 12B
  code       24 scope replies of the coding agent (6 per test project, 5,300-7,200-token inputs), Gemma 4 12B
  sentiment  48 comments, Gemma 3 4B
Each student writes its own reply up to its domain's stop rule (runaway guard: the domain's max_steps); the
teacher gets its own prompt (the domain's teacher prompt, Gemma 4 in its chat template, Gemma 3 plain, as when
the data was made) and writes at most as many tokens as its recorded reply has. Per model and kind: one
untimed warm-up pass over all the inputs (a 3-input warm-up left the first timed round twice as slow: the card was
still clocking up and meeting new input lengths), then 3 timed rounds over all of them.
Students: the reported runs (math tutor 16.5M, coding agent 12.0M, sentiment 3.4M) plus the math size sweep
(2.7M / 6.1M / 16.4M / 53.7M) for the curve of time against size. Teachers: Gemma 4 12B in bf16 when the card
has >= 30 GB, in 4 bits (NF4) when it has >= 11 GB, else not measured (it does not fit); Gemma 3 4B in bf16 when
the card has >= 11 GB. On the CPU only the students are timed.

    python experiments/speed_benchmark.py            # this machine's GPU (if any) and its CPU
    python experiments/speed_benchmark.py --no-cpu   # GPU only (rented cards: their CPUs are not the question)
    python experiments/speed_benchmark.py --students # students only (a rerun after a disturbed run; teachers kept)

Results: experiments/results/speed_replies.csv (every timed reply) and speed.csv (per hardware / device / model /
kind: median and quartiles of the time per reply, tokens per second, peak GPU memory)."""

import os
import platform
import statistics
import sys
import time
from pathlib import Path

from _common import REPO_ROOT, write_rows

sys.path.insert(0, str(REPO_ROOT))

import torch

PROBES_PATH = REPO_ROOT / "runs" / "attention_study" / "probes.pt"
ROUNDS = 3
DOMAINS = {"math": "math_tutor_full_v1", "code": "code_agent", "sentiment": "reddit_comment_sentiment"}
STUDENTS = {   # run -> the kind of input it answers
    "math-agent-kl-m-s1": "math", "math-full-kd-xs-s1": "math", "math-full-kd-s1": "math",
    "math-full-kd-m-s1": "math", "math-full-kd-l-s1": "math",
    "code-agent-kl-m-s1": "code", "kl-ce-30k-reddit-pure-kl-s1": "sentiment",
}
TEACHERS = {"google/gemma-4-12B-it": ("math", "code"), "google/gemma-3-4b-it": ("sentiment",)}
CHAT_TEMPLATE = {"google/gemma-4-12B-it": True, "google/gemma-3-4b-it": False}
REPLY_COLUMNS = ["key", "hardware", "device", "model", "precision", "kind", "probe", "round", "seconds",
                 "input_tokens", "reply_tokens"]
SUMMARY_COLUMNS = ["key", "hardware", "device", "model", "precision", "parameters", "kind", "replies", "median_ms",
                   "p25_ms", "p75_ms", "mean_reply_tokens", "tokens_per_second", "peak_memory_gb", "threads", "measured_at"]


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def hardware_name(device: str) -> str:
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    return platform.processor() or platform.machine()


def gpu_memory_gb() -> float:
    return torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0.0


def time_replies(name, precision, parameters, kind, device, probes, reply_once):
    """Warm-up, then ROUNDS timed passes; returns (reply rows, summary row)."""
    hardware = hardware_name(device)
    for probe in probes:   # warm-up, untimed
        reply_once(probe)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    rows = []
    for round_number in range(1, ROUNDS + 1):
        for probe in probes:
            synchronize(device)
            started = time.perf_counter()
            input_tokens, reply_tokens = reply_once(probe)
            synchronize(device)
            seconds = time.perf_counter() - started
            rows.append({"key": f"{hardware}|{device}|{name}|{kind}|{probe['index']}|{round_number}", "hardware": hardware,
                         "device": device, "model": name, "precision": precision, "kind": kind, "probe": probe["index"],
                         "round": round_number, "seconds": round(seconds, 5), "input_tokens": input_tokens,
                         "reply_tokens": reply_tokens})
        print(f"  {name} on {device}, {kind}, round {round_number}: "
              f"median {statistics.median(row['seconds'] for row in rows if row['round'] == round_number) * 1000:.1f} ms",
              flush=True)
    times = sorted(row["seconds"] * 1000 for row in rows)
    quartiles = statistics.quantiles(times, n=4)
    summary = {"key": f"{hardware}|{device}|{name}|{kind}", "hardware": hardware, "device": device, "model": name,
               "precision": precision, "parameters": parameters, "kind": kind, "replies": len(rows),
               "median_ms": round(statistics.median(times), 2), "p25_ms": round(quartiles[0], 2),
               "p75_ms": round(quartiles[2], 2),
               "mean_reply_tokens": round(sum(row["reply_tokens"] for row in rows) / len(rows), 1),
               "tokens_per_second": round(sum(row["reply_tokens"] for row in rows) / sum(row["seconds"] for row in rows), 1),
               "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device == "cuda" else "",
               "threads": torch.get_num_threads() if device == "cpu" else "",
               "measured_at": time.strftime("%Y-%m-%d %H:%M")}
    write_rows("speed_replies.csv", REPLY_COLUMNS, rows)
    write_rows("speed.csv", SUMMARY_COLUMNS, [summary])
    return summary


def time_student(run_name, kind, device, probes):
    from library import RunStore
    from library.domain import get_domain
    from library.model.student import StudentModel

    store = RunStore()
    run = store.get(run_name)
    domain = get_domain(run.domain)
    _, checkpoint_path = store.resolve_checkpoint(run.name, "latest")
    student = StudentModel.from_run(run, checkpoint_path, device=device, domain=domain)

    def reply_once(probe):
        result = student.generate(probe["sentence"], max_new_tokens=domain.max_steps)
        input_tokens = len(student.tokenizer.encode(domain.student_prompt(probe["sentence"]), add_special_tokens=False))
        return input_tokens, result.steps

    summary = time_replies(run_name, "fp32", run.total_parameters, kind, device, probes, reply_once)
    del student
    return summary


def time_teacher(model_name, load_in_4bit, kinds, probes_by_kind):
    from library.data_gen.model_handler import ModelHandler
    from library.domain import as_domain

    handler = ModelHandler(model_name, as_domain(DOMAINS[kinds[0]]), load_in_4bit=load_in_4bit)
    precision = "nf4" if load_in_4bit else "bf16"
    # NF4 keeps two 4-bit weights per byte (uint8 tensors): count them as weights, not bytes
    parameters = sum(parameter.numel() * (2 if parameter.dtype == torch.uint8 else 1)
                     for parameter in handler.model.parameters())
    end_ids = sorted(handler.end_token_ids)
    pad_id = handler.tokenizer.pad_token_id or handler.tokenizer.eos_token_id
    summaries = []
    for kind in kinds:
        domain = as_domain(DOMAINS[kind])

        def reply_once(probe, domain=domain):
            prompt = domain.teacher_prompt(probe["sentence"])
            if CHAT_TEMPLATE[model_name]:
                prompt = handler.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                               add_generation_prompt=True, enable_thinking=False)
            inputs = handler.encode_prompts([prompt])
            with torch.no_grad():
                output = handler.model.generate(**inputs, max_new_tokens=len(probe["reply_tokens"]), do_sample=False,
                                                eos_token_id=end_ids, pad_token_id=pad_id)
            input_length = inputs["input_ids"].shape[1]
            handler.tokenizer.decode(output[0, input_length:], skip_special_tokens=False)
            return input_length, int(output.shape[1] - input_length)

        summaries.append(time_replies(model_name.split("/")[-1], precision, parameters, kind, "cuda",
                                      probes_by_kind[kind], reply_once))
    del handler
    torch.cuda.empty_cache()
    return summaries


def main():
    measure_cpu = "--no-cpu" not in sys.argv[1:]
    probes = torch.load(PROBES_PATH, weights_only=False)
    probes_by_kind = {kind: [dict(probe, index=index) for index, probe in enumerate(probes) if probe["kind"] == kind]
                      for kind in DOMAINS}
    print({kind: len(items) for kind, items in probes_by_kind.items()}, "test inputs;",
          f"GPU: {hardware_name('cuda') if torch.cuda.is_available() else 'none'} ({gpu_memory_gb():.0f} GB);",
          f"CPU: {hardware_name('cpu')}, {torch.get_num_threads()} threads", flush=True)

    devices = (["cuda"] if torch.cuda.is_available() else []) + (["cpu"] if measure_cpu else [])
    for device in devices:
        for run_name, kind in STUDENTS.items():
            if (REPO_ROOT / "runs" / run_name).exists():
                time_student(run_name, kind, device, probes_by_kind[kind])
            else:
                print(f"  {run_name}: not on this machine, skipped")

    if torch.cuda.is_available() and "--students" not in sys.argv[1:]:
        memory = gpu_memory_gb()
        for model_name, kinds in TEACHERS.items():
            if model_name.endswith("12B-it"):
                load_in_4bit = None if memory < 11 else memory < 30
            else:
                load_in_4bit = None if memory < 11 else False
            if load_in_4bit is None:
                print(f"  {model_name}: does not fit in {memory:.0f} GB, not measured")
                continue
            time_teacher(model_name, load_in_4bit, kinds, probes_by_kind)


if __name__ == "__main__":
    main()
