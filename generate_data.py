"""Data-generation entry point — edit the JOBS list and run `python generate_data.py`.

Jobs are typed `GenerationJob` objects: pass a Domain object as `domain=` (no magic
strings), everything else has a default. Completed jobs (by info.json status) are skipped.
"""

import json
import os

from library import GenerationJob
from library.domain import CODE_RECALL, MATH_TUTOR
from library.data_gen.generator import DistillationDataGenerator
from library.shared import ExitListener, get_output_dir
from library.shared.queue_file import write_queue

JOBS = [
    # Cloud RTX 5090 (32 GB), plan step 7: bf16 teacher, batched. thesis_work/cloud_runbook.md.
    # Build the corpora first (python -c "from library.domain import MATH_TUTOR, CODE_RECALL;
    # MATH_TUTOR.build_corpus(); CODE_RECALL.build_corpus()"). On the 12 GB 4070 use
    # load_in_4bit=True and generation_batch_size=16 instead.
    GenerationJob(
        domain=CODE_RECALL,
        teacher_model="google/gemma-4-12B-it",
        load_in_4bit=False,
        chat_template=True,
        max_examples=10000,
        batch_size=32,
        generation_batch_size=16,   # try 32 on the 5090 once memory is seen
        description="code_recall (code finder), Gemma 4 12B bf16, batched",
    ),
    GenerationJob(
        domain=MATH_TUTOR,
        teacher_model="google/gemma-4-12B-it",
        load_in_4bit=False,
        chat_template=True,
        max_examples=1_000_000,     # the whole corpus: train turns first, then test turns
        batch_size=32,
        generation_batch_size=16,
        description="math_tutor turns, working-lines format, Gemma 4 12B bf16, batched",
    ),
]


def main():
    exit_listener = ExitListener()
    write_queue("generation", [
        {"name": f"{job.domain.name} / {job.teacher_model}", "description": job.description,
         "info_path": os.path.join(get_output_dir(job.domain.name, job.teacher_model), "info.json")}
        for job in JOBS], kind="generation")
    for job in JOBS:
        info_path = os.path.join(get_output_dir(job.domain.name, job.teacher_model), "info.json")
        if os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as file:
                if json.load(file).get("status") == "completed":
                    print(f"Skipping completed: {job.domain.name} ({job.teacher_model})")
                    continue
        DistillationDataGenerator(job, exit_listener).run()
    exit_listener.stop()
    print("Done.")


if __name__ == "__main__":
    main()
