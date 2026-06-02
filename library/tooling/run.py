"""Run, RunStore, Experiment — the canonical run model and cohort aggregation.

`Run` promotes the old `experimental_analysis/run_data.py` `RunData` (the best existing
abstraction): info.json + lazily-loaded typed logs + derived properties.
`RunStore` is the one place that discovers runs and resolves checkpoints (logic that was
copy-pasted into 4 tools). `Experiment` aggregates seed-replicate cohorts (the `experiment`
tag) into mean ± std and significance — promoted out of `thesis_work/build_grafikonok.py`.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from library.shared.config import RUNS_DIR
from library.shared.logging.reader import RunLogs, read_run_logs

FULL_GEMMA_INPUT_VOCAB = 262144


class Run:
    def __init__(self, name: str, info: dict, run_dir: Path):
        self.name = name
        self.info = info
        self.run_dir = run_dir
        self._logs: Optional[RunLogs] = None

    # ── logs (lazy) ───────────────────────────────────────────────────
    @property
    def logs(self) -> RunLogs:
        if self._logs is None:
            self._logs = read_run_logs(self.run_dir)
        return self._logs

    @property
    def train_batches(self):
        return self.logs.train_batches

    @property
    def eval_epochs(self):
        return self.logs.eval_epochs

    @property
    def mini_evals(self):
        return self.logs.mini_evals

    # ── identity / config ─────────────────────────────────────────────
    @property
    def status(self) -> str:
        return self.info.get("status", "unknown")

    @property
    def domain(self) -> str:
        return self.info.get("domain", "unknown")

    @property
    def experiment(self) -> Optional[str]:
        return self.info.get("experiment")

    @property
    def teacher_model(self) -> str:
        return self.info.get("teacher_model", "")

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
    def dropout(self) -> float:
        return self.info.get("dropout", 0)

    # ── model info ────────────────────────────────────────────────────
    @property
    def _model_info(self) -> dict:
        return self.info.get("model_info", {})

    @property
    def total_parameters(self) -> int:
        return self._model_info.get("total_parameters", 0)

    @property
    def input_vocab_size(self) -> int:
        return self._model_info.get("input_vocab_size", 0)

    @property
    def output_vocab_size(self) -> int:
        return self._model_info.get("output_vocab_size", 0)

    @property
    def has_reduced_input_vocab(self) -> bool:
        return 0 < self.input_vocab_size < FULL_GEMMA_INPUT_VOCAB

    def params_millions(self) -> float:
        return self.total_parameters / 1_000_000

    # ── derived metrics ───────────────────────────────────────────────
    @property
    def final_eval(self):
        return self.eval_epochs[-1] if self.eval_epochs else None

    @property
    def started_at(self) -> Optional[str]:
        return self.info.get("started_at")

    @property
    def completed_at(self) -> Optional[str]:
        return self.info.get("completed_at")

    @property
    def training_hours(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            start = datetime.fromisoformat(self.started_at)
            end = datetime.fromisoformat(self.completed_at)
            return (end - start).total_seconds() / 3600
        if self.train_batches:
            return sum(batch.time_seconds for batch in self.train_batches) / 3600
        return None

    def _last_n_train(self, n: int = 200):
        batches = self.train_batches
        return batches[-n:] if len(batches) >= n else batches

    @property
    def final_train_loss(self) -> Optional[float]:
        recent = self._last_n_train()
        return sum(b.loss for b in recent) / len(recent) if recent else None

    @property
    def final_train_accuracy(self) -> Optional[float]:
        recent = self._last_n_train()
        return sum(b.accuracy for b in recent) / len(recent) if recent else None

    def short_name(self) -> str:
        name = self.name
        for prefix in ("tsc-", "math-", "exp-"):
            if name.startswith(prefix):
                name = name[len(prefix):]
        return name

    def __repr__(self) -> str:
        return f"Run({self.name!r}, {self.domain}, {self.status})"


# ── metric extraction (one place that maps metric names to a Run value) ──
_EVAL_METRICS = {
    "student_accuracy", "teacher_forced_accuracy", "task_accuracy", "avg_loss",
    "kl_loss", "ce_loss", "topk_accuracy", "teacher_student_overlap", "mean_target_rank",
    "teacher_entropy", "student_entropy", "perplexity", "validity_rate",
    "termination_rate", "mean_length_ratio",
}


def run_metric(run: Run, metric: str) -> Optional[float]:
    if metric in _EVAL_METRICS:
        return getattr(run.final_eval, metric, None) if run.final_eval else None
    if metric == "params_millions":
        return run.params_millions()
    if metric == "training_hours":
        return run.training_hours
    if metric == "final_train_loss":
        return run.final_train_loss
    if metric == "final_train_accuracy":
        return run.final_train_accuracy
    return None


@dataclass
class Stat:
    metric: str
    mean: float
    std: float
    n: int
    values: List[float]


@dataclass
class Comparison:
    metric: str
    mean_a: float
    mean_b: float
    cohens_d: float
    welch_t: Optional[float]


class Experiment:
    """A cohort of seed-replicate runs sharing one `experiment` tag."""

    def __init__(self, tag: str, runs: List[Run]):
        self.tag = tag
        self.runs = runs

    def __len__(self) -> int:
        return len(self.runs)

    def values(self, metric: str) -> List[float]:
        return [v for v in (run_metric(run, metric) for run in self.runs) if v is not None]

    def stats(self, metric: str, sample: bool = False) -> Stat:
        # Default = population std (ddof=0): the descriptive spread of the fixed seed cohort,
        # matching the thesis tables. `sample=True` gives the inferential (n-1) estimate.
        values = self.values(metric)
        mean = statistics.mean(values) if values else 0.0
        if len(values) >= 2:
            std = statistics.stdev(values) if sample else statistics.pstdev(values)
        else:
            std = 0.0
        return Stat(metric=metric, mean=mean, std=std, n=len(values), values=values)

    def compare(self, other: "Experiment", metric: str) -> Comparison:
        a, b = self.values(metric), other.values(metric)
        mean_a = statistics.mean(a) if a else 0.0
        mean_b = statistics.mean(b) if b else 0.0
        cohens_d = welch_t = None
        if len(a) >= 2 and len(b) >= 2:
            sa, sb = statistics.stdev(a), statistics.stdev(b)
            pooled = math.sqrt(((len(a) - 1) * sa ** 2 + (len(b) - 1) * sb ** 2) / (len(a) + len(b) - 2))
            cohens_d = (mean_a - mean_b) / pooled if pooled else 0.0
            denom = math.sqrt(sa ** 2 / len(a) + sb ** 2 / len(b))
            welch_t = (mean_a - mean_b) / denom if denom else None
        return Comparison(metric=metric, mean_a=mean_a, mean_b=mean_b,
                          cohens_d=cohens_d if cohens_d is not None else 0.0, welch_t=welch_t)


class RunStore:
    """Discovers runs and resolves checkpoints — the single source for both."""

    def __init__(self, runs_dir: Path = RUNS_DIR):
        self.runs_dir = Path(runs_dir)

    def all(self) -> List[Run]:
        runs = []
        if not self.runs_dir.exists():
            return runs
        for directory in sorted(self.runs_dir.iterdir()):
            run = self._load(directory)
            if run is not None:
                runs.append(run)
        return runs

    def _load(self, directory: Path) -> Optional[Run]:
        info_path = directory / "info.json"
        if not info_path.exists():
            return None
        with open(info_path, "r", encoding="utf-8") as file:
            info = json.load(file)
        return Run(name=info.get("run_name", directory.name), info=info, run_dir=directory)

    def get(self, name: str) -> Run:
        run = self._load(self.runs_dir / name)
        if run is None:
            raise FileNotFoundError(f"No run named {name!r} in {self.runs_dir}")
        return run

    def completed(self) -> List[Run]:
        return [run for run in self.all() if run.status == "completed"]

    def filter(self, prefix: Optional[str] = None, status: Optional[str] = None,
               domain: Optional[str] = None, experiment: Optional[str] = None,
               names: Optional[List[str]] = None) -> List[Run]:
        runs = self.all()
        if names:
            runs = [r for r in runs if r.name in names]
        if prefix:
            runs = [r for r in runs if r.name.startswith(prefix)]
        if status:
            runs = [r for r in runs if r.status == status]
        if domain:
            runs = [r for r in runs if r.domain == domain]
        if experiment:
            runs = [r for r in runs if r.experiment == experiment]
        return runs

    def by_experiment(self) -> Dict[str, Experiment]:
        cohorts: Dict[str, List[Run]] = {}
        for run in self.all():
            if run.experiment:
                cohorts.setdefault(run.experiment, []).append(run)
        return {tag: Experiment(tag, runs) for tag, runs in cohorts.items()}

    def experiment(self, tag: str) -> Experiment:
        return Experiment(tag, self.filter(experiment=tag))

    # ── checkpoint resolution (ported from the top-k tool, the most complete) ──
    def resolve_checkpoint(self, run_name: str, selection: str = "latest"):
        checkpoint_dir = self.runs_dir / run_name / "checkpoints"
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

        selection = str(selection).strip().lower()
        epoch_checkpoints = sorted(
            checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda path: int(path.stem.split("_")[-1]),
        )
        temp_path = checkpoint_dir / "temp_checkpoint.pt"

        if selection == "latest":
            if epoch_checkpoints:
                return epoch_checkpoints[-1].stem.replace("checkpoint_", ""), epoch_checkpoints[-1]
            if temp_path.exists():
                return "temp", temp_path
            raise FileNotFoundError(f"No checkpoints in {checkpoint_dir}")
        if selection == "temp":
            if temp_path.exists():
                return "temp", temp_path
            raise FileNotFoundError(f"Temp checkpoint not found: {temp_path}")
        if selection.isdigit():
            epoch_path = checkpoint_dir / f"checkpoint_epoch_{selection}.pt"
            if epoch_path.exists():
                return f"epoch_{selection}", epoch_path
            raise FileNotFoundError(f"Epoch checkpoint not found: {epoch_path}")
        named = checkpoint_dir / selection
        if named.exists():
            return named.stem, named
        if Path(selection).exists():
            return Path(selection).stem, Path(selection)
        raise FileNotFoundError(f"Checkpoint {selection!r} not found for run {run_name!r}")


def sort_runs(runs: List[Run], sort_by: str = "params") -> List[Run]:
    keys = {
        "params": lambda r: r.total_parameters,
        "name": lambda r: r.name,
        "layers": lambda r: r.num_layers,
        "hidden": lambda r: r.hidden_dim,
        "student_accuracy": lambda r: (r.final_eval.student_accuracy if r.final_eval else 0),
        "task_accuracy": lambda r: (r.final_eval.task_accuracy if r.final_eval else 0),
        "loss": lambda r: (r.final_eval.avg_loss if r.final_eval else float("inf")),
        "time": lambda r: (r.training_hours or float("inf")),
    }
    return sorted(runs, key=keys.get(sort_by, keys["params"]))
