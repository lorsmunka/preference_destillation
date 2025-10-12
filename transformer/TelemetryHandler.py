import json
import os
from time import time, sleep
from typing import Dict
from pathlib import Path

TELEMETRY_DIR = "./telemetry"
TELEMETRY_FILE_PATH = f"{TELEMETRY_DIR}/training_telemetry.json"


class TelemetryHandler:
    def __init__(self):
        self.current_epoch = 0
        self.current_batch = 0
        self.total_batches_processed = 0
        self.total_runtime_seconds = 0
        self.session_count = 0
        self.session_start_time = None

        self.training_log_dir = Path(TELEMETRY_DIR)
        self.training_log_dir.mkdir(exist_ok=True)
        self.training_log_file = self.training_log_dir / "training.jsonl"

        self.load_save()

    def load_save(self):
        start_time = time()
        print("Loading telemetry...")
        if os.path.exists(TELEMETRY_FILE_PATH):
            with open(TELEMETRY_FILE_PATH, "r", encoding="utf-8") as file:
                data = json.load(file)
                self.current_epoch = data.get("current_epoch", 0)
                self.current_batch = data.get("current_batch", 0)
                self.total_batches_processed = data.get(
                    "total_batches_processed", 0)
                self.total_runtime_seconds = data.get(
                    "total_runtime_seconds", 0)
                self.session_count = data.get("session_count", 0)

        self.session_count += 1
        self.session_start_time = time()

        elapsed_time = time() - start_time
        print(f"Loaded telemetry -> took {elapsed_time:.2f} seconds.\n")
        print("== Telemetry ==")
        print(f"Current epoch: {self.current_epoch}")
        print(f"Current batch: {self.current_batch}")
        print(f"Total batches processed: {self.total_batches_processed:,}")
        print(f"Total runtime (seconds): {self.total_runtime_seconds:,.2f}")
        print(f"Session count: {self.session_count:,}")
        print(f"Training log file: {self.training_log_file}")
        print(f"Waiting 2 seconds before continuing... \n")
        sleep(2)

    def save(self):
        start_time = time()
        print("Saving telemetry...")

        if not os.path.exists(TELEMETRY_DIR):
            os.makedirs(TELEMETRY_DIR)

        self.total_runtime_seconds += time() - self.session_start_time
        with open(TELEMETRY_FILE_PATH, "w", encoding="utf-8") as file:
            json.dump({
                "current_epoch": self.current_epoch,
                "current_batch": self.current_batch,
                "total_batches_processed": self.total_batches_processed,
                "total_runtime_seconds": self.total_runtime_seconds,
                "session_count": self.session_count
            }, file, indent=4)

        elapsed_time = time() - start_time
        print(
            f"Saved telemetry to {TELEMETRY_FILE_PATH} -> took {elapsed_time:.2f} seconds.\n")

    def update_progress(self, epoch: int, batch: int):
        self.current_epoch = epoch
        self.current_batch = batch
        self.total_batches_processed += 1

    def should_resume(self):
        return self.current_epoch > 0 or self.current_batch > 0

    def log_training_example(self, epoch: int, batch: int, example: int, num_steps: int, loss: float, time_seconds: float):
        self._write_log({
            "type": "train_example",
            "epoch": epoch,
            "batch": batch,
            "example": example,
            "num_steps": num_steps,
            "loss": loss,
            "time_seconds": time_seconds
        })

    def log_train_epoch(self, epoch: int, avg_loss: float, total_steps: int):
        self._write_log({
            "type": "train_epoch",
            "epoch": epoch,
            "avg_loss": avg_loss,
            "total_steps": total_steps
        })

    def log_eval_epoch(self, epoch: int, avg_loss: float, accuracy: float, total_steps: int):
        self._write_log({
            "type": "eval_epoch",
            "epoch": epoch,
            "avg_loss": avg_loss,
            "accuracy": accuracy,
            "total_steps": total_steps
        })

    def _write_log(self, data: Dict):
        with open(self.training_log_file, 'a') as f:
            f.write(json.dumps(data) + '\n')
