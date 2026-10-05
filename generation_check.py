"""Five-minute check before a long teacher run: for every job in generate_data.py, 64
inputs from its corpus go through the real generator path (same settings, batched as
configured) into a temporary folder, and are read back.

    python generation_check.py

Prints, per job: examples kept, why the others were skipped, how many the teacher got
right (where the corpus has ground truth), seconds per example and peak GPU memory.
Nothing is written to batches/.
"""

import gc
import json
import tempfile
from collections import Counter
from dataclasses import replace
from pathlib import Path
from time import time

import torch

import library.data_gen.generator as generator_module
from generate_data import JOBS
from library.shared import ExitListener
from library.shared.batch_files import read_batch

CHECK_INPUTS = 64


def check(job, workdir: Path) -> None:
    corpus = workdir / f"{job.domain.name}_corpus.jsonl"
    with open(job.domain.corpus_path, encoding="utf-8") as source, open(corpus, "w", encoding="utf-8") as out:
        for _, line in zip(range(CHECK_INPUTS), source):
            out.write(line)
    domain = job.domain
    domain_class = type(domain)

    class CheckDomain(domain_class):
        @property
        def corpus_path(self):
            return str(corpus)

    check_domain = CheckDomain.__new__(CheckDomain)
    check_domain.__dict__.update(domain.__dict__)
    output = workdir / domain.name
    generator_module.get_output_dir = lambda name, model: str(output)
    check_job = replace(job, domain=check_domain, max_examples=CHECK_INPUTS, description="generation check")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time()
    listener = ExitListener()
    generator_module.DistillationDataGenerator(check_job, listener).run()
    listener.stop()
    seconds = time() - started

    records = [record for path in sorted(output.glob("batch_*.npz")) for record in read_batch(path)]
    info = json.load(open(output / "info.json", encoding="utf-8")) if (output / "info.json").exists() else {}
    correct = Counter(record.get("answer_correct") for record in records)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0
    print(f"\n== {domain.name}: {len(records)}/{CHECK_INPUTS} kept, answer_correct {dict(correct)}, "
          f"{seconds / CHECK_INPUTS:.2f} s per input incl. loading, peak GPU {peak:.1f} GB, info {info.get('status')}")
    if records:
        print("   example response:", repr(records[0]["model_response"][:120]))


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        for job in JOBS:
            check(job, Path(directory))
            gc.collect()                 # release this teacher before the next job loads one
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
