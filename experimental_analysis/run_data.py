import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field


RUNS_DIR = Path(__file__).parent.parent / "runs"


@dataclass
class EpochEval:
    epoch: int
    avg_loss: float
    kl_loss: float
    ce_loss: float
    teacher_forced_accuracy: float
    student_accuracy: float
    classification_accuracy: float
    confusion_matrices: dict


@dataclass
class MiniEval:
    epoch: int
    batch: int
    teacher_forced_accuracy: float
    student_accuracy: float
    classification_accuracy: float


@dataclass
class TrainBatch:
    epoch: int
    batch: int
    loss: float
    kl_loss: float
    ce_loss: float
    accuracy: float
    learning_rate: float
    kl_ratio: float
    temperature: float
    time_seconds: float
    steps: int


@dataclass
class TrainEpoch:
    epoch: int
    avg_loss: float
    kl_loss: float
    ce_loss: float


@dataclass
class RunData:
    name: str
    info: dict
    train_batches: List[TrainBatch] = field(default_factory=list)
    train_epochs: List[TrainEpoch] = field(default_factory=list)
    eval_epochs: List[EpochEval] = field(default_factory=list)
    mini_evals: List[MiniEval] = field(default_factory=list)

    @property
    def status(self) -> str:
        return self.info.get("status", "unknown")

    @property
    def domain(self) -> str:
        return self.info.get("domain", "unknown")

    @property
    def hidden_dim(self) -> int:
        return self.info.get("hidden_dim", 0)

    @property
    def num_layers(self) -> int:
        return self.info.get("num_layers", 0)

    @property
    def num_heads(self) -> int:
        return self.info.get("num_heads", 0)

    @property
    def total_parameters(self) -> int:
        return self.info.get("model_info", {}).get("total_parameters", 0)

    @property
    def input_vocab_size(self) -> int:
        return self.info.get("model_info", {}).get("input_vocab_size", 0)

    @property
    def output_vocab_size(self) -> int:
        return self.info.get("model_info", {}).get("output_vocab_size", 0)

    @property
    def embedding_params(self) -> int:
        return self.info.get("model_info", {}).get("input_embedding_params", 0)

    @property
    def attention_params(self) -> int:
        return self.info.get("model_info", {}).get("attention_params", 0)

    @property
    def feedforward_params(self) -> int:
        return self.info.get("model_info", {}).get("feedforward_params", 0)

    @property
    def body_params(self) -> int:
        return self.attention_params + self.feedforward_params

    @property
    def kl_ratio_start(self) -> float:
        return self.info.get("kl_ratio_start", 0)

    @property
    def kl_ratio_end(self) -> float:
        return self.info.get("kl_ratio_end", 0)

    @property
    def temperature_start(self) -> float:
        return self.info.get("distillation_temperature_start", 0)

    @property
    def temperature_end(self) -> float:
        return self.info.get("distillation_temperature_end", 0)

    @property
    def learning_rate(self) -> float:
        return self.info.get("learning_rate", 0)

    @property
    def epoch_count(self) -> int:
        return self.info.get("epoch_count", 0)

    @property
    def batch_size(self) -> int:
        return self.info.get("batch_size", 0)

    @property
    def dropout(self) -> float:
        return self.info.get("dropout", 0)

    @property
    def started_at(self) -> Optional[str]:
        return self.info.get("started_at")

    @property
    def completed_at(self) -> Optional[str]:
        return self.info.get("completed_at")

    @property
    def training_hours(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            from datetime import datetime
            start = datetime.fromisoformat(self.started_at)
            end = datetime.fromisoformat(self.completed_at)
            return (end - start).total_seconds() / 3600
        if self.train_batches:
            return sum(batch.time_seconds for batch in self.train_batches) / 3600
        return None

    @property
    def total_train_batches(self) -> int:
        return len(self.train_batches)

    @property
    def final_eval(self) -> Optional[EpochEval]:
        if self.eval_epochs:
            return self.eval_epochs[-1]
        return None

    @property
    def final_train_loss(self) -> Optional[float]:
        if self.train_batches:
            last_n = self.train_batches[-200:] if len(self.train_batches) >= 200 else self.train_batches
            return sum(batch.loss for batch in last_n) / len(last_n)
        return None

    @property
    def final_train_accuracy(self) -> Optional[float]:
        if self.train_batches:
            last_n = self.train_batches[-200:] if len(self.train_batches) >= 200 else self.train_batches
            return sum(batch.accuracy for batch in last_n) / len(last_n)
        return None

    @property
    def has_reduced_input_vocab(self) -> bool:
        return self.input_vocab_size < 262144

    def params_millions(self) -> float:
        return self.total_parameters / 1_000_000

    def short_name(self) -> str:
        name = self.name
        for prefix in ["tsc-", "math-"]:
            if name.startswith(prefix):
                name = name[len(prefix):]
        return name


def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    entries = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def parse_train_batch(entry: dict) -> TrainBatch:
    return TrainBatch(
        epoch=entry.get("epoch", 0),
        batch=entry.get("batch", 0),
        loss=entry.get("loss", 0),
        kl_loss=entry.get("kl_loss", 0),
        ce_loss=entry.get("ce_loss", 0),
        accuracy=entry.get("accuracy", 0),
        learning_rate=entry.get("learning_rate", 0),
        kl_ratio=entry.get("kl_ratio", 0),
        temperature=entry.get("temperature", 0),
        time_seconds=entry.get("time_seconds", 0),
        steps=entry.get("steps", 0),
    )


def parse_train_epoch(entry: dict) -> TrainEpoch:
    return TrainEpoch(
        epoch=entry.get("epoch", 0),
        avg_loss=entry.get("avg_loss", 0),
        kl_loss=entry.get("kl_loss", 0),
        ce_loss=entry.get("ce_loss", 0),
    )


def parse_eval_epoch(entry: dict) -> EpochEval:
    return EpochEval(
        epoch=entry.get("epoch", 0),
        avg_loss=entry.get("avg_loss", 0),
        kl_loss=entry.get("kl_loss", 0),
        ce_loss=entry.get("ce_loss", 0),
        teacher_forced_accuracy=entry.get("teacher_forced_accuracy", entry.get("accuracy", 0)),
        student_accuracy=entry.get("student_accuracy", 0),
        classification_accuracy=entry.get("classification_accuracy", 0),
        confusion_matrices=entry.get("confusion_matrices", {}),
    )


def parse_mini_eval(entry: dict) -> MiniEval:
    return MiniEval(
        epoch=entry.get("epoch", 0),
        batch=entry.get("batch", 0),
        teacher_forced_accuracy=entry.get("teacher_forced_accuracy", 0),
        student_accuracy=entry.get("student_accuracy", 0),
        classification_accuracy=entry.get("classification_accuracy", 0),
    )


def load_run(run_dir: Path) -> Optional[RunData]:
    info_path = run_dir / "info.json"
    if not info_path.exists():
        return None

    with open(info_path, "r", encoding="utf-8") as file:
        info = json.load(file)

    run = RunData(name=info.get("run_name", run_dir.name), info=info)

    training_log = run_dir / "logs" / "training.jsonl"
    entries = load_jsonl(training_log)

    for entry in entries:
        entry_type = entry.get("type", "")
        if entry_type == "train_batch":
            run.train_batches.append(parse_train_batch(entry))
        elif entry_type == "train_epoch":
            run.train_epochs.append(parse_train_epoch(entry))
        elif entry_type == "eval_epoch":
            run.eval_epochs.append(parse_eval_epoch(entry))
        elif entry_type == "mini_eval":
            run.mini_evals.append(parse_mini_eval(entry))

    return run


def load_all_runs(runs_dir: Path = RUNS_DIR) -> List[RunData]:
    runs = []
    if not runs_dir.exists():
        return runs

    for run_dir in sorted(runs_dir.iterdir()):
        if run_dir.is_dir():
            run = load_run(run_dir)
            if run is not None:
                runs.append(run)

    return runs


def filter_runs(
    runs: List[RunData],
    prefix: Optional[str] = None,
    status: Optional[str] = None,
    names: Optional[List[str]] = None,
) -> List[RunData]:
    filtered = runs

    if names:
        filtered = [run for run in filtered if run.name in names]

    if prefix:
        filtered = [run for run in filtered if run.name.startswith(prefix)]

    if status:
        filtered = [run for run in filtered if run.status == status]

    return filtered


def sort_runs(runs: List[RunData], sort_by: str = "params") -> List[RunData]:
    sort_keys = {
        "params": lambda run: run.total_parameters,
        "name": lambda run: run.name,
        "layers": lambda run: run.num_layers,
        "hidden": lambda run: run.hidden_dim,
        "student_accuracy": lambda run: (run.final_eval.student_accuracy if run.final_eval else 0),
        "teacher_forced_accuracy": lambda run: (run.final_eval.teacher_forced_accuracy if run.final_eval else 0),
        "classification_accuracy": lambda run: (run.final_eval.classification_accuracy if run.final_eval else 0),
        "loss": lambda run: (run.final_eval.avg_loss if run.final_eval else float("inf")),
        "time": lambda run: (run.training_hours or float("inf")),
    }

    key_func = sort_keys.get(sort_by, sort_keys["params"])
    return sorted(runs, key=key_func)
