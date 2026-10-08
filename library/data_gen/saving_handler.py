import os
import json
from time import time
from datetime import datetime, timezone
from typing import Optional

from library.shared.batch_files import write_batch
from library.shared.logging import RunLogger


class SavingHandler:
    def __init__(self, logger: RunLogger, output_dir: str, vocabulary_sha1: Optional[str] = None):
        self.logger = logger
        self.output_dir = output_dir
        self.vocabulary_sha1 = vocabulary_sha1   # stored in every batch: which tokens the logit columns are
        self.info_path = os.path.join(self.output_dir, "info.json")

        os.makedirs(self.output_dir, exist_ok=True)

    def save_batch(self, batch_examples):
        start_time = time()

        # Numbered by the examples saved so far, rounded up: a final partial batch gets the next
        # number instead of overwriting the last full one (which cost Virag 32 of 5,000 examples).
        index = -(-self.logger.successful_sentence_count // self.logger.batch_size)
        path = write_batch(os.path.join(self.output_dir, f"batch_{index}.npz"), batch_examples,
                           vocabulary_sha1=self.vocabulary_sha1)

        elapsed_time = time() - start_time
        print(f"\nSaved batch to {path} -> took {elapsed_time:.2f} seconds.\n")

    def read_info(self) -> dict:
        """The info.json an earlier session wrote, or {} for a fresh job."""
        if not os.path.exists(self.info_path):
            return {}
        with open(self.info_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def write_info(self, fields: dict, status: str = "in_progress"):
        """Rewrite info.json with `fields` and `status`. The first session's started_at is
        kept on resume. Written under a temporary name and renamed, so a crash mid-write
        never leaves a truncated info.json."""
        info = self.read_info()
        info.update(fields)
        info["status"] = status
        info.setdefault("started_at", datetime.now(timezone.utc).isoformat())
        info["completed_at"] = datetime.now(timezone.utc).isoformat() if status == "completed" else None

        temporary_path = self.info_path + ".partial"
        with open(temporary_path, "w", encoding="utf-8") as file:
            json.dump(info, file, indent=2)
        os.replace(temporary_path, self.info_path)
