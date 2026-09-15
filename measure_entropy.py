"""Teacher entropy per domain, from the logits already stored in batches/ — no model, no GPU.

    python measure_entropy.py

Reads a sample of batch files per domain (first SAMPLE_FILES of each), computes the
teacher's per-step entropy from the stored reduced-vocabulary logits, and prints one row
per domain: mean entropy in nats, its effective branching factor exp(H), the median, and
the share of steps that are effectively one-hot. For reddit it also splits by JSON field
(tone / sentiment / safety / toxicity), scoring only the step that emits each field's value.

Results are also written to experiments/results/domain_entropy.csv (rows replaced by key).
"""

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

BATCHES_ROOT = Path("batches")
RESULTS_PATH = Path("experiments") / "results" / "domain_entropy.csv"
SAMPLE_FILES = int(sys.argv[1]) if len(sys.argv) > 1 else 8   # batch files per domain; override only
ONE_HOT_THRESHOLD = 0.1                                          # nats; below this the teacher had "one option"
REDDIT_FIELDS = ("tone", "sentiment", "safety", "toxicity")


def step_entropy(logits):
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max()
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()
    nonzero = probabilities[probabilities > 0]
    return float(-(nonzero * np.log(nonzero)).sum())


def clean(token):
    return token.replace("▁", "").strip().strip('"').lower()


def field_of_each_step(steps):
    """For reddit-style JSON output: label the step that emits each field's value with the
    field name; every other step gets None."""
    labels = [None] * len(steps)
    current_field = None
    awaiting_value = False
    for index, step in enumerate(steps):
        token = clean(step["token"])
        if token in REDDIT_FIELDS:
            current_field = token
            awaiting_value = False
        elif ":" in step["token"] and current_field:
            awaiting_value = True
        elif awaiting_value and token.isalpha():
            labels[index] = current_field
            current_field = None
            awaiting_value = False
    return labels


def iter_batch_files(domain_dir):
    for teacher_dir in sorted(p for p in domain_dir.iterdir() if p.is_dir()):
        files = sorted(teacher_dir.glob("batch_*.jsonl"), key=lambda p: int(p.stem.split("_")[-1]))
        yield teacher_dir.name, files[:SAMPLE_FILES]


def summarize(entropies):
    values = np.asarray(entropies)
    return {
        "steps": int(values.size),
        "mean_nats": float(values.mean()),
        "exp_H": float(math.exp(values.mean())),
        "median_nats": float(np.median(values)),
        "one_hot_share": float((values < ONE_HOT_THRESHOLD).mean()),
    }


def measure_domain(domain_dir):
    rows = []
    for teacher, files in iter_batch_files(domain_dir):
        all_entropies = []
        per_field = {field: [] for field in REDDIT_FIELDS}
        example_count = 0
        for path in files:
            with open(path, "r", encoding="utf-8") as file:
                for line in file:
                    record = json.loads(line)
                    steps = record["steps"]
                    if not steps:
                        continue
                    example_count += 1
                    entropies = [step_entropy(step["logits"]) for step in steps]
                    all_entropies.extend(entropies)
                    for label, entropy in zip(field_of_each_step(steps), entropies):
                        if label:
                            per_field[label].append(entropy)
        if not all_entropies:
            continue
        rows.append({"key": f"{domain_dir.name}", "domain": domain_dir.name, "field": "all",
                     "teacher": teacher, "examples": example_count, **summarize(all_entropies)})
        for field, entropies in per_field.items():
            if entropies:
                rows.append({"key": f"{domain_dir.name}/{field}", "domain": domain_dir.name,
                             "field": field, "teacher": teacher, "examples": example_count,
                             **summarize(entropies)})
    return rows


def write_rows(rows):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                existing[row["key"]] = row
    for row in rows:
        existing[row["key"]] = {k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()}
    columns = list(rows[0].keys())
    with open(RESULTS_PATH, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in existing.values():
            writer.writerow(row)


def main():
    rows = []
    for domain_dir in sorted(p for p in BATCHES_ROOT.iterdir() if p.is_dir()):
        rows.extend(measure_domain(domain_dir))
    if not rows:
        print("No batches found under", BATCHES_ROOT)
        return

    print(f"{'domain / field':34} {'teacher':22} {'examples':>8} {'steps':>7} {'H nats':>8} {'exp(H)':>7} {'median':>8} {'one-hot':>8}")
    for row in rows:
        label = row["domain"] if row["field"] == "all" else f"  {row['domain']} / {row['field']}"
        print(f"{label:34} {row['teacher'][:22]:22} {row['examples']:8d} {row['steps']:7d} "
              f"{row['mean_nats']:8.4f} {row['exp_H']:7.3f} {row['median_nats']:8.4f} {row['one_hot_share']:8.1%}")
    write_rows(rows)
    print(f"\n-> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
