import json
import os
from time import time, sleep
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path

from .config import LOGS_DIR, BATCH_SIZE


STATE_FILE = f"{LOGS_DIR}/state.json"
GENERATION_LOG_FILE = f"{LOGS_DIR}/generation.jsonl"
TRAINING_LOG_FILE = f"{LOGS_DIR}/training.jsonl"


class Logger:
    def __init__(self):
        self.processed_sentence_count = 0
        self.successful_sentence_count = 0

        self.current_epoch = 0
        self.current_batch = 0
        self.total_batches_processed = 0

        self.total_runtime_seconds = 0
        self.session_count = 0
        self.session_start_time = None
        self.session_id = None

        Path(LOGS_DIR).mkdir(exist_ok=True)
        self._load()

    def _generate_session_id(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _timestamp(self) -> str:
        return datetime.now().isoformat()

    def _load(self):
        start_time = time()
        print("Loading logger state...")

        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r", encoding="utf-8") as file:
                data = json.load(file)
                self.processed_sentence_count = data.get("processed_sentence_count", 0)
                self.successful_sentence_count = data.get("successful_sentence_count", 0)
                self.current_epoch = data.get("current_epoch", 0)
                self.current_batch = data.get("current_batch", 0)
                self.total_batches_processed = data.get("total_batches_processed", 0)
                self.total_runtime_seconds = data.get("total_runtime_seconds", 0)
                self.session_count = data.get("session_count", 0)

        self.session_count += 1
        self.session_start_time = time()
        self.session_id = self._generate_session_id()

        print(f"Loaded logger state -> took {time() - start_time:.2f} seconds.\n")
        self._print_status()
        print(f"Waiting 2 seconds before continuing...\n")
        sleep(2)

    def _print_status(self):
        print("== Logger State ==")
        print(f"Session: {self.session_id} (#{self.session_count})")
        print(f"Generation: {self.successful_sentence_count:,}/{self.processed_sentence_count:,} sentences ({self.batch_count} batches)")
        print(f"Training: epoch {self.current_epoch}, batch {self.current_batch}, total batches {self.total_batches_processed:,}")
        print(f"Runtime: {self.total_runtime_seconds:,.2f}s over {self.session_count:,} sessions")

    def save(self):
        start_time = time()
        print("Saving logger state...")

        self.total_runtime_seconds += time() - self.session_start_time
        self.session_start_time = time()

        with open(STATE_FILE, "w", encoding="utf-8") as file:
            json.dump({
                "processed_sentence_count": self.processed_sentence_count,
                "successful_sentence_count": self.successful_sentence_count,
                "current_epoch": self.current_epoch,
                "current_batch": self.current_batch,
                "total_batches_processed": self.total_batches_processed,
                "total_runtime_seconds": self.total_runtime_seconds,
                "session_count": self.session_count,
            }, file, indent=4)

        print(f"Saved logger state -> took {time() - start_time:.2f} seconds.\n")

    @property
    def current_batch_sentence_count(self) -> int:
        return self.successful_sentence_count % BATCH_SIZE

    @property
    def batch_count(self) -> int:
        return self.successful_sentence_count // BATCH_SIZE

    def log_generation_batch(self, batch_index: int, processed: int, successful: int,
                              skipped: int, time_seconds: float,
                              skip_reasons: Optional[Dict[str, int]] = None):
        self._write_generation_log({
            "timestamp": self._timestamp(),
            "session_id": self.session_id,
            "type": "batch",
            "batch": batch_index,
            "processed": processed,
            "successful": successful,
            "skipped": skipped,
            "time_seconds": round(time_seconds, 2),
            "skip_reasons": skip_reasons or {},
        })

    def _write_generation_log(self, data: Dict):
        with open(GENERATION_LOG_FILE, 'a', encoding='utf-8') as file:
            file.write(json.dumps(data) + '\n')

    def update_progress(self, epoch: int, batch: int):
        self.current_epoch = epoch
        self.current_batch = batch
        self.total_batches_processed += 1

    def should_resume(self) -> bool:
        should_resume = self.current_epoch > 0 or self.current_batch > 0
        if should_resume:
            print(f"Resuming from epoch {self.current_epoch + 1}, batch {self.current_batch}\n")
        else:
            print("Starting fresh training\n")
        return should_resume

    def log_training_batch(self, epoch: int, batch: int, steps: int,
                           loss: float, kl_loss: float, ce_loss: float,
                           accuracy: float, learning_rate: float, kl_ratio: float, time_seconds: float):
        self._write_training_log({
            "timestamp": self._timestamp(),
            "session_id": self.session_id,
            "type": "train_batch",
            "epoch": epoch,
            "batch": batch,
            "steps": steps,
            "loss": round(loss, 6),
            "kl_loss": round(kl_loss, 6),
            "ce_loss": round(ce_loss, 6),
            "accuracy": round(accuracy, 4),
            "learning_rate": learning_rate,
            "kl_ratio": round(kl_ratio, 6),
            "time_seconds": round(time_seconds, 2),
        })

    def log_train_epoch(self, epoch: int, avg_loss: float, total_steps: int,
                        kl_loss: float, ce_loss: float):
        self._write_training_log({
            "timestamp": self._timestamp(),
            "session_id": self.session_id,
            "type": "train_epoch",
            "epoch": epoch,
            "avg_loss": round(avg_loss, 6),
            "kl_loss": round(kl_loss, 6),
            "ce_loss": round(ce_loss, 6),
            "total_steps": total_steps,
        })

    def log_eval_epoch(self, epoch: int, avg_loss: float, teacher_forced_accuracy: float,
                       student_accuracy: float, total_steps: int, kl_loss: float, ce_loss: float):
        self._write_training_log({
            "timestamp": self._timestamp(),
            "session_id": self.session_id,
            "type": "eval_epoch",
            "epoch": epoch,
            "avg_loss": round(avg_loss, 6),
            "kl_loss": round(kl_loss, 6),
            "ce_loss": round(ce_loss, 6),
            "teacher_forced_accuracy": round(teacher_forced_accuracy, 4),
            "student_accuracy": round(student_accuracy, 4),
            "total_steps": total_steps,
        })

    def _write_training_log(self, data: Dict):
        with open(TRAINING_LOG_FILE, 'a', encoding='utf-8') as file:
            file.write(json.dumps(data) + '\n')
