"""Teacher entropy (nats per step, softmax over the stored reduced-vocabulary logits at T=1) split by
reply type, for both agents. Math tutor: every record of 200 randomly chosen batch files (the teacher's
entropy does not depend on the split). Coding agent: all test records (as in the 0.17 figure)."""
import json
import random
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, r"D:\preference_destillation")
from library.shared.batch_files import batch_paths, bfloat16_bits_to_float32

SOURCES = [
    ("math_tutor", r"D:\preference_destillation\batches\math_tutor_full_v1\google_gemma-4-12B-it", 200, None),
    ("code_agent", r"D:\preference_destillation\batches\code_agent\google_gemma-4-12B-it", None, "test"),
]


def step_entropies(rows):
    rows = rows.astype(np.float64)
    rows -= rows.max(axis=1, keepdims=True)
    probabilities = np.exp(rows)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return -(probabilities * np.log(np.clip(probabilities, 1e-30, None))).sum(axis=1)


for name, folder, file_sample, only_split in SOURCES:
    paths = batch_paths(folder)
    if file_sample:
        paths = random.Random(0).sample(paths, file_sample)
    totals = defaultdict(lambda: [0.0, 0, 0, 0])   # entropy sum, steps, near-certain steps, records
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(data["meta"].item())
            count = len(meta["sentences"])
            splits = meta.get("split") or [None] * count
            if only_split and only_split not in splits:
                continue
            types = meta.get("turn_type") or ["?"] * count
            logits = bfloat16_bits_to_float32(data["logits_bf16"])
            offsets = data["step_offsets"]
        for index in range(count):
            if only_split and splits[index] != only_split:
                continue
            entropies = step_entropies(logits[int(offsets[index]):int(offsets[index + 1])])
            total = totals[types[index]]
            total[0] += float(entropies.sum())
            total[1] += len(entropies)
            total[2] += int((entropies < 0.01).sum())
            total[3] += 1
    print(f"== {name}")
    all_sum = sum(total[0] for total in totals.values())
    all_steps = sum(total[1] for total in totals.values())
    for turn_type, (entropy_sum, steps, certain, records) in sorted(totals.items()):
        print(f"  {turn_type:10s} records {records:6d} steps {steps:8d} mean entropy {entropy_sum / steps:.4f} nats"
              f"  near-certain steps (<0.01) {certain / steps:.3f}")
    print(f"  all        mean entropy {all_sum / all_steps:.4f} nats")
