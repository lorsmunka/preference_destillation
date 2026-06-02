"""RunLogger — writes the canonical schema to a run's own logs dir.

Ported from `shared/logger.py`, with two changes: it takes the run's `logs_dir` explicitly
(no global `LOGS_DIR`), and it writes the canonical `task_accuracy` field plus optional
eval-time distribution metrics. Drop-in for the training-side rewire (a later phase); it is
not wired into the live trainer in this change.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from time import time
from typing import Dict, Optional


class RunLogger:
    def __init__(self, logs_dir: str, batch_size: int = 32):
        self.logs_dir = str(logs_dir)
        self.batch_size = batch_size
        self.state_file = os.path.join(self.logs_dir, "state.json")
        self.generation_log_file = os.path.join(self.logs_dir, "generation.jsonl")
        self.training_log_file = os.path.join(self.logs_dir, "training.jsonl")

        self.processed_sentence_count = 0
        self.successful_sentence_count = 0
        self.current_epoch = 0
        self.current_batch = 0
        self.total_batches_processed = 0
        self.total_runtime_seconds = 0.0
        self.session_count = 0
        self.session_start_time: Optional[float] = None
        self.session_id: Optional[str] = None

        Path(self.logs_dir).mkdir(parents=True, exist_ok=True)
        self._load()

    # ── state / resume ────────────────────────────────────────────────
    def _load(self) -> None:
        if os.path.exists(self.state_file):
            with open(self.state_file, "r", encoding="utf-8") as file:
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
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def save(self) -> None:
        if self.session_start_time is not None:
            self.total_runtime_seconds += time() - self.session_start_time
            self.session_start_time = time()
        with open(self.state_file, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "processed_sentence_count": self.processed_sentence_count,
                    "successful_sentence_count": self.successful_sentence_count,
                    "current_epoch": self.current_epoch,
                    "current_batch": self.current_batch,
                    "total_batches_processed": self.total_batches_processed,
                    "total_runtime_seconds": self.total_runtime_seconds,
                    "session_count": self.session_count,
                },
                file,
                indent=4,
            )

    def should_resume(self) -> bool:
        return self.current_epoch > 0 or self.current_batch > 0

    def update_progress(self, epoch: int, batch: int) -> None:
        self.current_epoch = epoch
        self.current_batch = batch
        self.total_batches_processed += 1

    # ── writers ───────────────────────────────────────────────────────
    def _write(self, path: str, record: dict) -> None:
        record = {"timestamp": datetime.now().isoformat(), "session_id": self.session_id, **record}
        with open(path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")

    def log_generation_batch(self, batch_index, processed, successful, skipped,
                             time_seconds, skip_reasons: Optional[Dict[str, int]] = None) -> None:
        self._write(self.generation_log_file, {
            "type": "batch", "batch": batch_index, "processed": processed,
            "successful": successful, "skipped": skipped,
            "time_seconds": round(time_seconds, 2), "skip_reasons": skip_reasons or {},
        })

    def log_training_batch(self, epoch, batch, steps, loss, kl_loss, ce_loss, accuracy,
                           learning_rate, kl_ratio, temperature, time_seconds) -> None:
        self._write(self.training_log_file, {
            "type": "train_batch", "epoch": epoch, "batch": batch, "steps": steps,
            "loss": round(loss, 6), "kl_loss": round(kl_loss, 6), "ce_loss": round(ce_loss, 6),
            "accuracy": round(accuracy, 4), "learning_rate": learning_rate,
            "kl_ratio": round(kl_ratio, 6), "temperature": round(temperature, 6),
            "time_seconds": round(time_seconds, 2),
        })

    def log_train_epoch(self, epoch, avg_loss, total_steps, kl_loss, ce_loss) -> None:
        self._write(self.training_log_file, {
            "type": "train_epoch", "epoch": epoch, "avg_loss": round(avg_loss, 6),
            "kl_loss": round(kl_loss, 6), "ce_loss": round(ce_loss, 6), "total_steps": total_steps,
        })

    def log_eval_epoch(self, epoch, avg_loss, teacher_forced_accuracy, student_accuracy,
                       task_accuracy, confusion_matrices, total_steps, kl_loss, ce_loss,
                       distribution: Optional[Dict[str, float]] = None) -> None:
        record = {
            "type": "eval_epoch", "epoch": epoch, "avg_loss": round(avg_loss, 6),
            "kl_loss": round(kl_loss, 6), "ce_loss": round(ce_loss, 6),
            "teacher_forced_accuracy": round(teacher_forced_accuracy, 4),
            "student_accuracy": round(student_accuracy, 4),
            "task_accuracy": round(task_accuracy, 4),
            "confusion_matrices": confusion_matrices, "total_steps": total_steps,
        }
        if distribution:
            record.update(distribution)
        self._write(self.training_log_file, record)

    def log_mini_eval(self, epoch, batch, teacher_forced_accuracy, student_accuracy,
                      task_accuracy, total_steps, distribution: Optional[Dict[str, float]] = None) -> None:
        record = {
            "type": "mini_eval", "epoch": epoch, "batch": batch,
            "teacher_forced_accuracy": round(teacher_forced_accuracy, 4),
            "student_accuracy": round(student_accuracy, 4),
            "task_accuracy": round(task_accuracy, 4), "total_steps": total_steps,
        }
        if distribution:
            record.update(distribution)
        self._write(self.training_log_file, record)

    @property
    def batch_count(self) -> int:
        return self.successful_sentence_count // self.batch_size
