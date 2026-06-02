"""Read run logs into typed records. Analysis never re-implements JSONL parsing."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from .records import (
    EvalEpoch,
    GenerationBatch,
    MiniEval,
    TrainBatch,
    TrainEpoch,
    parse_record,
)


@dataclass
class RunLogs:
    train_batches: List[TrainBatch] = field(default_factory=list)
    train_epochs: List[TrainEpoch] = field(default_factory=list)
    eval_epochs: List[EvalEpoch] = field(default_factory=list)
    mini_evals: List[MiniEval] = field(default_factory=list)


def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_training_records(training_jsonl: Path) -> RunLogs:
    logs = RunLogs()
    for record in _read_jsonl(Path(training_jsonl)):
        parsed = parse_record(record)
        if isinstance(parsed, TrainBatch):
            logs.train_batches.append(parsed)
        elif isinstance(parsed, TrainEpoch):
            logs.train_epochs.append(parsed)
        elif isinstance(parsed, EvalEpoch):
            logs.eval_epochs.append(parsed)
        elif isinstance(parsed, MiniEval):
            logs.mini_evals.append(parsed)
    return logs


def read_run_logs(run_dir: Path) -> RunLogs:
    return read_training_records(Path(run_dir) / "logs" / "training.jsonl")


def read_generation_batches(logs_dir: Path) -> List[GenerationBatch]:
    path = Path(logs_dir) / "generation.jsonl"
    return [
        GenerationBatch.from_dict(record)
        for record in _read_jsonl(path)
        if record.get("type") == GenerationBatch.TYPE
    ]
