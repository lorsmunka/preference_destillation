"""CPU only, read-only on the project: the coding agent students' attention on the attention study's 24 code
probes after every pass (KL m: passes 1-10, CE m: passes 1-4), plus Gemma's and the untrained student's
numbers that experiments/results/attention_models.csv lost (its "gemma" key is shared by math, code and
sentiment, so only the last row survived).

Reads runs/attention_study/probes.pt, teacher.pt, students/*.pt and the runs' checkpoints. Writes only into
this folder: code_rows/<key>.pt and code_attention_passes.csv / gemma_metrics.txt.

    python -X utf8 code_attention_passes.py rows      # student rows per pass (CPU, slow)
    python -X utf8 code_attention_passes.py metrics   # the tables
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""   # never touch the GPU (another session trains on it)

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, r"D:\preference_destillation\experiments")

import numpy as np
import torch

import attention_report as report
import attention_study as study

HERE = Path(__file__).parent
ROWS_DIR = HERE / "code_rows"
PASSES = {"kl": ("code-agent-kl-m-s1", 10), "ce": ("code-agent-ce-m-s1", 4)}


def pass_students():
    students = []
    for arm, (run, passes) in PASSES.items():
        for epoch in range(passes):
            students.append(study.Student(f"code-{arm}-m-p{epoch + 1}", run, "code", f"{arm.upper()} m after pass {epoch + 1}", epoch=epoch))
    return students


def load_probes():
    return torch.load(study.PROBES_PATH, weights_only=False)


def make_rows():
    ROWS_DIR.mkdir(exist_ok=True)
    probes = load_probes()
    torch.set_grad_enabled(False)
    only = sys.argv[2:] if len(sys.argv) > 2 else None
    for student in pass_students():
        if only and student.key not in only:
            continue
        path = ROWS_DIR / f"{student.key}.pt"
        if path.exists():
            continue
        started = time.time()
        with torch.no_grad():
            rows = study.student_rows(student, probes, "cpu")
        torch.save(rows, path)
        print(f"{student.key}: {len(rows)} probes, {time.time() - started:.0f} s", flush=True)


original_load_rows = report.load_rows


def load_rows(key):
    path = ROWS_DIR / f"{key}.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    return original_load_rows(key)


report.load_rows = load_rows


def uniform_shares(probes, kind):
    """What an even spread over the visible tokens gives each region (mean over reply positions)."""
    sums, count = {}, 0
    for probe in probes:
        if probe["kind"] != kind:
            continue
        even = report.pattern_rows(probe)[0, 1]   # [queries, groups]
        masks = report.probe_masks(probe)
        for name, mask in masks.items():
            sums[name] = sums.get(name, 0.0) + float((even * mask).sum())
        count += even.shape[0]
    return {name: value / count for name, value in sums.items()}


def head_hits(key, layer, head, probes):
    """Like attention_report.retrieval_heads for one head: in how many probes the head's most-attended line
    while writing line numbers is a target line, and while writing the file name a line of a target file."""
    rows_by_probe = load_rows(key)
    hits = file_hits = probe_count = 0
    import re
    for index, probe in enumerate(probes):
        if probe["kind"] != "code":
            continue
        rows = report.writing_digits(probe)
        lines = probe["sentence"].split("\n")
        target = [region == "target lines" for region in probe["group_regions"][:len(lines)]]
        if not rows or not any(target):
            continue
        probe_count += 1
        file_of_line, current_file = [], None
        for line in lines:
            match = re.match(r'<file name="([^"]+)">', line)
            current_file = match.group(1) if match else current_file
            file_of_line.append(current_file)
        target_files = set(study.scope_target_lines(probe["reply"]))
        newline = next((row for row, token in enumerate(probe["reply_tokens"]) if "\n" in token or "Ċ" in token), None)
        name_rows = list(range(newline + 1, rows[0])) if newline is not None and newline + 1 < rows[0] else []
        head_rows = report.view_of(rows_by_probe[index])[0][layer, head]
        hits += bool(target[int(head_rows[rows, :len(lines)].mean(0).argmax())])
        if name_rows:
            file_hits += file_of_line[int(head_rows[name_rows, :len(lines)].mean(0).argmax())] in target_files
    return hits, file_hits, probe_count


def metrics():
    probes = load_probes()
    keys = ["code-untrained-m"] + [student.key for student in pass_students() if (ROWS_DIR / f"{student.key}.pt").exists()] \
        + ["code-kl-m", "code-size-xl", report.GEMMA]
    names = ["target lift", "target lines when writing a line number", "target lines", "target file", "other files",
             "message", "markup", "end of input", "reply so far"]
    even = uniform_shares(probes, "code")
    print("even-spread shares:", {name: round(value, 4) for name, value in even.items()})
    out_rows = []
    for key in keys:
        values = report.head_metrics(key, probes, "code")
        row = {"model": key}
        for name in names:
            if name in values:
                row[f"{name} (mean head)"] = round(float(values[name].mean()), 4)
                row[f"{name} (best head)"] = round(float(values[name].max()), 4)
        best = np.unravel_index(values["target lines when writing a line number"].argmax(),
                                values["target lines when writing a line number"].shape)
        hits, file_hits, count = head_hits(key, int(best[0]), int(best[1]), probes)
        row["best head (layer, head)"] = f"{int(best[0]) + 1},{int(best[1]) + 1}"
        row["best head: line hits"] = f"{hits}/{count}"
        row["best head: file hits"] = f"{file_hits}/{count}"
        if "outside view" in values:
            row["outside view (mean head)"] = round(float(values["outside view"].mean()), 4)
        out_rows.append(row)
        print(row, flush=True)
    columns = []
    for row in out_rows:
        columns += [column for column in row if column not in columns]
    with open(HERE / "code_attention_passes.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(out_rows)

    # Gemma on the math probes (lost from the CSV): the same summary columns as attention_models.csv
    gemma_math = report.head_metrics(report.GEMMA, probes, "math")
    with open(HERE / "gemma_metrics.txt", "w", encoding="utf-8") as handle:
        for name in sorted(gemma_math):
            line = f"math {name}: mean head {float(gemma_math[name].mean()):.4f}, best head {float(gemma_math[name].max()):.4f}"
            handle.write(line + "\n")
            print(line)
        even_math = uniform_shares(probes, "math")
        handle.write("math even-spread shares: " + str({name: round(value, 4) for name, value in even_math.items()}) + "\n")
        print("math even-spread shares:", {name: round(value, 4) for name, value in even_math.items()})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if sys.argv[1] == "rows":
        make_rows()
    else:
        metrics()
