"""Typed log records — the single source of truth for log shapes.

Each record knows its `TYPE` string, how to parse a dict (tolerant of the old on-disk
schema), and how to serialize back. The key rename `classification_accuracy -> task_accuracy`
is handled at the read boundary so old logs and new logs both load.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Dict, Optional


def _first(record: dict, *names, default=None):
    for name in names:
        value = record.get(name)
        if value is not None:
            return value
    return default


@dataclass
class TrainBatch:
    TYPE: ClassVar[str] = "train_batch"
    epoch: int
    batch: int
    steps: int
    loss: float
    kl_loss: float
    ce_loss: float
    accuracy: float
    learning_rate: float
    kl_ratio: float
    temperature: float
    time_seconds: float
    session_id: Optional[str] = None
    timestamp: Optional[str] = None

    @classmethod
    def from_dict(cls, record: dict) -> "TrainBatch":
        return cls(
            epoch=record.get("epoch", 0),
            batch=record.get("batch", 0),
            steps=record.get("steps", 0),
            loss=record.get("loss", 0.0),
            kl_loss=record.get("kl_loss", 0.0),
            ce_loss=record.get("ce_loss", 0.0),
            accuracy=record.get("accuracy", 0.0),
            learning_rate=record.get("learning_rate", 0.0),
            kl_ratio=record.get("kl_ratio", 0.0),
            temperature=record.get("temperature", 0.0),
            time_seconds=record.get("time_seconds", 0.0),
            session_id=record.get("session_id"),
            timestamp=record.get("timestamp"),
        )


@dataclass
class TrainEpoch:
    TYPE: ClassVar[str] = "train_epoch"
    epoch: int
    avg_loss: float
    kl_loss: float
    ce_loss: float
    total_steps: int = 0

    @classmethod
    def from_dict(cls, record: dict) -> "TrainEpoch":
        return cls(
            epoch=record.get("epoch", 0),
            avg_loss=record.get("avg_loss", 0.0),
            kl_loss=record.get("kl_loss", 0.0),
            ce_loss=record.get("ce_loss", 0.0),
            total_steps=record.get("total_steps", 0),
        )


@dataclass
class EvalEpoch:
    TYPE: ClassVar[str] = "eval_epoch"
    epoch: int
    avg_loss: float
    kl_loss: float
    ce_loss: float
    teacher_forced_accuracy: float
    student_accuracy: float
    task_accuracy: float  # was logged as "classification_accuracy"
    confusion_matrices: dict = field(default_factory=dict)
    total_steps: int = 0
    # Eval-time distribution / generation metrics (None for legacy logs that predate them).
    topk_accuracy: Optional[float] = None
    teacher_student_overlap: Optional[float] = None
    mean_target_rank: Optional[float] = None
    teacher_entropy: Optional[float] = None
    student_entropy: Optional[float] = None
    perplexity: Optional[float] = None
    validity_rate: Optional[float] = None
    termination_rate: Optional[float] = None
    mean_length_ratio: Optional[float] = None

    @classmethod
    def from_dict(cls, record: dict) -> "EvalEpoch":
        return cls(
            epoch=record.get("epoch", 0),
            avg_loss=record.get("avg_loss", 0.0),
            kl_loss=record.get("kl_loss", 0.0),
            ce_loss=record.get("ce_loss", 0.0),
            teacher_forced_accuracy=_first(record, "teacher_forced_accuracy", "accuracy", default=0.0),
            student_accuracy=record.get("student_accuracy", 0.0),
            task_accuracy=_first(record, "task_accuracy", "classification_accuracy", default=0.0),
            confusion_matrices=record.get("confusion_matrices", {}) or {},
            total_steps=record.get("total_steps", 0),
            topk_accuracy=record.get("topk_accuracy"),
            teacher_student_overlap=record.get("teacher_student_overlap"),
            mean_target_rank=record.get("mean_target_rank"),
            teacher_entropy=record.get("teacher_entropy"),
            student_entropy=record.get("student_entropy"),
            perplexity=record.get("perplexity"),
            validity_rate=record.get("validity_rate"),
            termination_rate=record.get("termination_rate"),
            mean_length_ratio=record.get("mean_length_ratio"),
        )


@dataclass
class MiniEval:
    TYPE: ClassVar[str] = "mini_eval"
    epoch: int
    batch: int
    teacher_forced_accuracy: float
    student_accuracy: float
    task_accuracy: float  # was "classification_accuracy"
    total_steps: int = 0
    topk_accuracy: Optional[float] = None
    teacher_student_overlap: Optional[float] = None
    mean_target_rank: Optional[float] = None
    teacher_entropy: Optional[float] = None
    student_entropy: Optional[float] = None
    perplexity: Optional[float] = None
    validity_rate: Optional[float] = None
    termination_rate: Optional[float] = None
    mean_length_ratio: Optional[float] = None

    @classmethod
    def from_dict(cls, record: dict) -> "MiniEval":
        return cls(
            epoch=record.get("epoch", 0),
            batch=record.get("batch", 0),
            teacher_forced_accuracy=record.get("teacher_forced_accuracy", 0.0),
            student_accuracy=record.get("student_accuracy", 0.0),
            task_accuracy=_first(record, "task_accuracy", "classification_accuracy", default=0.0),
            total_steps=record.get("total_steps", 0),
            topk_accuracy=record.get("topk_accuracy"),
            teacher_student_overlap=record.get("teacher_student_overlap"),
            mean_target_rank=record.get("mean_target_rank"),
            teacher_entropy=record.get("teacher_entropy"),
            student_entropy=record.get("student_entropy"),
            perplexity=record.get("perplexity"),
            validity_rate=record.get("validity_rate"),
            termination_rate=record.get("termination_rate"),
            mean_length_ratio=record.get("mean_length_ratio"),
        )


@dataclass
class GenerationBatch:
    TYPE: ClassVar[str] = "batch"
    batch: int
    processed: int
    successful: int
    skipped: int
    time_seconds: float
    skip_reasons: Dict[str, int] = field(default_factory=dict)
    session_id: Optional[str] = None

    @classmethod
    def from_dict(cls, record: dict) -> "GenerationBatch":
        return cls(
            batch=record.get("batch", 0),
            processed=record.get("processed", 0),
            successful=record.get("successful", 0),
            skipped=record.get("skipped", 0),
            time_seconds=record.get("time_seconds", 0.0),
            skip_reasons=record.get("skip_reasons", {}) or {},
            session_id=record.get("session_id"),
        )


# type string -> dataclass. "batch" (generation) is parsed on demand by the generation reader,
# since it lives in a different file than the training records.
_TRAINING_RECORDS = {
    TrainBatch.TYPE: TrainBatch,
    TrainEpoch.TYPE: TrainEpoch,
    EvalEpoch.TYPE: EvalEpoch,
    MiniEval.TYPE: MiniEval,
}


def parse_record(record: dict):
    """Parse a training-log dict into its typed record, or None if the type is unknown."""
    record_class = _TRAINING_RECORDS.get(record.get("type", ""))
    return record_class.from_dict(record) if record_class else None
