"""Convert every legacy `batch_N.jsonl` under batches/ to the compact `batch_N.npz`.

    python convert_batches.py

Writes the `.npz` beside the `.jsonl`, verifies the new file reads back identically
(logits, tokens, sentences, responses, predicted indices), and only then moves on. Files
that already have an `.npz` are skipped, so the script is re-runnable. Originals are
never deleted here — do that by hand once you are satisfied.
"""

import sys
from pathlib import Path
from time import time

import numpy as np

from library.shared.batch_files import BATCH_FILE, read_batch, write_batch

BATCHES_ROOT = Path("batches")


def verify(original, converted):
    assert len(original) == len(converted), "example count"
    for a, b in zip(original, converted):
        assert a["sentence"] == b["sentence"], "sentence"
        assert a.get("model_response", "") == b["model_response"], "model_response"
        assert len(a["steps"]) == len(b["steps"]), "step count"
        for step_a, step_b in zip(a["steps"], b["steps"]):
            assert step_a["token"] == step_b["token"], "token"
            assert step_a["predicted_token_index"] == step_b["predicted_token_index"], "predicted index"
            if not np.array_equal(np.asarray(step_a["logits"], dtype=np.float32),
                                  np.asarray(step_b["logits"], dtype=np.float32)):
                raise AssertionError("logits differ after bfloat16 round-trip")


def convert_directory(directory):
    jsonl_files = sorted((p for p in directory.iterdir() if BATCH_FILE.match(p.name) and p.suffix == ".jsonl"),
                         key=lambda p: int(BATCH_FILE.match(p.name).group(1)))
    pending = [p for p in jsonl_files if not p.with_suffix(".npz").exists()]
    print(f"{directory}: {len(jsonl_files)} jsonl, {len(pending)} to convert")
    before = after = 0
    started = time()
    for index, path in enumerate(pending, 1):
        records = read_batch(path)
        target = write_batch(path.with_suffix(".npz"), records)
        verify(records, read_batch(target))
        before += path.stat().st_size
        after += target.stat().st_size
        if index % 200 == 0 or index == len(pending):
            elapsed = time() - started
            rate = index / elapsed if elapsed else 0
            remaining = (len(pending) - index) / rate if rate else 0
            print(f"  {index}/{len(pending)}  {before / 1e9:.2f} GB -> {after / 1e9:.2f} GB  "
                  f"({before / max(after, 1):.1f}x)  ~{remaining / 60:.0f} min left", flush=True)
    return before, after


def main():
    total_before = total_after = 0
    domain_dirs = sorted(p for p in BATCHES_ROOT.iterdir() if p.is_dir())
    for domain_dir in domain_dirs:
        for teacher_dir in sorted(p for p in domain_dir.iterdir() if p.is_dir()):
            before, after = convert_directory(teacher_dir)
            total_before += before
            total_after += after
    if total_before:
        print(f"\nConverted {total_before / 1e9:.1f} GB of jsonl into {total_after / 1e9:.1f} GB of npz "
              f"({total_before / total_after:.1f}x smaller). Originals untouched.")
    else:
        print("\nNothing to convert.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
