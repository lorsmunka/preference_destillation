"""Data-generation entry point — edit the JOBS list and run `python generate_data.py`.

Jobs are typed `GenerationJob` objects: pass a Domain object as `domain=` (no magic
strings), everything else has a default. Completed jobs (by info.json status) are skipped.
"""

import json
import os

from library import GenerationJob
from library.domain import LONG_CONTEXT_RECALL
from library.data_gen.generator import DistillationDataGenerator
from library.shared import ExitListener, get_output_dir
from library.shared.queue_file import write_queue

JOBS = [
    GenerationJob(
        domain=LONG_CONTEXT_RECALL,
        teacher_model="google/gemma-4-12B-it",
        load_in_4bit=True,          # 24 GB bf16 -> ~7 GB NF4; weights come from models/gemma-4-12B-it (re-sharded)
        chat_template=True,
        max_examples=5000,
        batch_size=32,
        description="Virág long-context recall, Gemma 4 12B teacher, no thinking",
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
