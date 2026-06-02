"""Logging: one canonical typed schema shared by the writer (train side) and reader
(analysis side). Replaces the old split between `shared/logger.py` (dict writer) and
`experimental_analysis/run_data.py` (dict re-parser).

The reader tolerates the existing on-disk schema (incl. the old `classification_accuracy`
field name) so the 120 current logs load unchanged; the writer emits the new canonical
names (`task_accuracy`) plus optional eval-time distribution fields.
"""

from .records import (
    EvalEpoch,
    GenerationBatch,
    MiniEval,
    TrainBatch,
    TrainEpoch,
    parse_record,
)
from .reader import read_generation_batches, read_run_logs, read_training_records
from .writer import RunLogger

__all__ = [
    "TrainBatch",
    "TrainEpoch",
    "EvalEpoch",
    "MiniEval",
    "GenerationBatch",
    "parse_record",
    "read_run_logs",
    "read_training_records",
    "read_generation_batches",
    "RunLogger",
]
