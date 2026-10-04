import os
import json
from time import time
from datetime import datetime, timezone

from library.shared.batch_files import write_batch
from library.shared.logging import RunLogger


class SavingHandler:
    def __init__(self, logger: RunLogger, output_dir: str):
        self.logger = logger
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

    def save_batch(self, batch_examples):
        start_time = time()

        # Numbered by the examples saved so far, rounded up: a final partial batch gets the next
        # number instead of overwriting the last full one (which cost Virag 32 of 5,000 examples).
        index = -(-self.logger.successful_sentence_count // self.logger.batch_size)
        path = write_batch(os.path.join(self.output_dir, f"batch_{index}.npz"), batch_examples)

        elapsed_time = time() - start_time
        print(f"\nSaved batch to {path} -> took {elapsed_time:.2f} seconds.\n")

    def write_info(self, queue_element: dict, status: str = "in_progress",
                   examples_generated: int = 0, batches_written: int = 0):
        info = {
            "domain": queue_element["domain"],
            "model_name": queue_element["model_name"],
            "description": queue_element["description"],
            "max_examples": queue_element["max_examples"],
            "batch_size": queue_element["batch_size"],
            "status": status,
            "started_at": queue_element.get("started_at", datetime.now(timezone.utc).isoformat()),
            "completed_at": datetime.now(timezone.utc).isoformat() if status == "completed" else None,
            "examples_generated": examples_generated,
            "batches_written": batches_written,
        }

        path = os.path.join(self.output_dir, "info.json")
        with open(path, "w", encoding="utf-8") as file:
            json.dump(info, file, indent=2)
