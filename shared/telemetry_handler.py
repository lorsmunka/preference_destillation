import json
import os
from time import time, sleep
from typing import Dict
from pathlib import Path

from .config import TELEMETRY_DIR, BATCH_SIZE


TELEMETRY_FILE = f"{TELEMETRY_DIR}/telemetry.json"
TRAINING_LOG_FILE = f"{TELEMETRY_DIR}/training.jsonl"


class TelemetryHandler:
    def __init__(self):
        self.processed_sentence_count = 0
        self.successful_sentence_count = 0
        self.current_epoch = 0
        self.current_batch = 0
        self.total_batches_processed = 0
        self.total_runtime_seconds = 0
        self.session_count = 0
        self.session_start_time = None

        Path(TELEMETRY_DIR).mkdir(exist_ok=True)
        self._load()

    def _load(self):
        start_time = time()
        print("Loading telemetry...")

        if os.path.exists(TELEMETRY_FILE):
            with open(TELEMETRY_FILE, "r", encoding="utf-8") as file:
                data = json.load(file)
                self.processed_sentence_count = data.get(
                    "processed_sentence_count", 0)
                self.successful_sentence_count = data.get(
                    "successful_sentence_count", 0)
                self.current_epoch = data.get("current_epoch", 0)
                self.current_batch = data.get("current_batch", 0)
                self.total_batches_processed = data.get(
                    "total_batches_processed", 0)
                self.total_runtime_seconds = data.get(
                    "total_runtime_seconds", 0)
                self.session_count = data.get("session_count", 0)

        self.session_count += 1
        self.session_start_time = time()

        print(f"Loaded telemetry -> took {time() - start_time:.2f} seconds.\n")
        self._print_status()
        print(f"Waiting 2 seconds before continuing...\n")
        sleep(2)

    def _print_status(self):
        print("== Telemetry ==")
        print(
            f"Distillation: {self.successful_sentence_count:,}/{self.processed_sentence_count:,} sentences")
        print(
            f"Training: epoch {self.current_epoch}, batch {self.current_batch}, total batches {self.total_batches_processed:,}")
        print(
            f"Runtime: {self.total_runtime_seconds:,.2f}s over {self.session_count:,} sessions\n")

    def save(self):
        start_time = time()
        print("Saving telemetry...")

        self.total_runtime_seconds += time() - self.session_start_time
        self.session_start_time = time()

        with open(TELEMETRY_FILE, "w", encoding="utf-8") as file:
            json.dump({
                "processed_sentence_count": self.processed_sentence_count,
                "successful_sentence_count": self.successful_sentence_count,
                "current_epoch": self.current_epoch,
                "current_batch": self.current_batch,
                "total_batches_processed": self.total_batches_processed,
                "total_runtime_seconds": self.total_runtime_seconds,
                "session_count": self.session_count,
            }, file, indent=4)

        print(f"Saved telemetry -> took {time() - start_time:.2f} seconds.\n")

    @property
    def current_batch_sentence_count(self) -> int:
        return self.successful_sentence_count % BATCH_SIZE

    @property
    def batch_count(self) -> int:
        return self.successful_sentence_count // BATCH_SIZE

    def update_progress(self, epoch: int, batch: int):
        self.current_epoch = epoch
        self.current_batch = batch
        self.total_batches_processed += 1

    def should_resume(self) -> bool:
        should_resume = self.current_epoch > 0 or self.current_batch > 0
        if should_resume:
            print(
                f"Resuming from epoch {self.current_epoch + 1}, batch {self.current_batch}\n")
        else:
            print("Starting fresh training\n")
        return should_resume

    def log_training_example(self, epoch: int, batch: int, example: int, num_steps: int,
                             loss: float, time_seconds: float, kl_loss: float,
                             ce_loss: float, accuracy: float):
        self._write_log({
            "type": "train_example",
            "epoch": epoch,
            "batch": batch,
            "example": example,
            "num_steps": num_steps,
            "loss": loss,
            "kl_loss": kl_loss,
            "ce_loss": ce_loss,
            "accuracy": accuracy,
            "time_seconds": time_seconds
        })

    def log_train_epoch(self, epoch: int, avg_loss: float, total_steps: int,
                        kl_loss: float, ce_loss: float):
        self._write_log({
            "type": "train_epoch",
            "epoch": epoch,
            "avg_loss": avg_loss,
            "kl_loss": kl_loss,
            "ce_loss": ce_loss,
            "total_steps": total_steps
        })

    def log_eval_epoch(self, epoch: int, avg_loss: float, accuracy: float,
                       total_steps: int, kl_loss: float, ce_loss: float):
        self._write_log({
            "type": "eval_epoch",
            "epoch": epoch,
            "avg_loss": avg_loss,
            "kl_loss": kl_loss,
            "ce_loss": ce_loss,
            "accuracy": accuracy,
            "total_steps": total_steps
        })

    def _write_log(self, data: Dict):
        with open(TRAINING_LOG_FILE, 'a') as file:
            file.write(json.dumps(data) + '\n')
