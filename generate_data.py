"""Data-generation entry point — edit the JOBS list and run `python generate_data.py`.

Jobs are typed `GenerationJob` objects: pass a Domain object as `domain=` (no magic
strings), everything else has a default. Completed jobs (by info.json status) are skipped.
"""

import json
import os

from library import GenerationJob
from library.domain import REDDIT_SENTIMENT
from library.data_gen.generator import DistillationDataGenerator
from library.shared import ExitListener, get_output_dir

JOBS = [
    GenerationJob(
        domain=REDDIT_SENTIMENT,
        max_examples=8000,
        batch_size=32,
        description="Example sentiment distillation-data generation",
    ),
]


def main():
    exit_listener = ExitListener()
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
