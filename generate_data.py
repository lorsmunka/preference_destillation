"""Data-generation entry point — edit the JOBS list and run `python generate_data.py`.

Replaces the old `distillation_data_generation/main.py` + `data_generation_queue.json`:
jobs are plain Python now. Completed jobs (by info.json status) are skipped.
"""

import json
import os

from distillation_data_generation.generator import DistillationDataGenerator
from shared import ExitListener, get_output_dir

JOBS = [
    {
        "domain": "reddit_comment_sentiment",
        "model_name": "google/gemma-3-4b-it",
        "max_examples": 8000,
        "batch_size": 32,
        "description": "Example sentiment distillation-data generation",
    },
]


def main():
    exit_listener = ExitListener()
    for job in JOBS:
        info_path = os.path.join(get_output_dir(job["domain"], job["model_name"]), "info.json")
        if os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as file:
                if json.load(file).get("status") == "completed":
                    print(f"Skipping completed: {job['domain']} ({job['model_name']})")
                    continue
        DistillationDataGenerator(job, exit_listener).run()
    exit_listener.stop()
    print("Done.")


if __name__ == "__main__":
    main()
