"""Run, RunStore, Experiment — the canonical run model and cohort aggregation.

`Run` promotes the old `experimental_analysis/run_data.py` `RunData` (the best existing
abstraction): info.json + lazily-loaded typed logs + derived properties.
`RunStore` is the one place that discovers runs and resolves checkpoints (logic that was
copy-pasted into 4 tools). `Experiment` is one seed-replicate cohort — the completed runs
of one `experiment` tag that share a configuration — aggregated into mean ± std and
significance (promoted out of `thesis_work/build_grafikonok.py`).

Every task accuracy comes with its majority baseline (the accuracy of always giving the
most common answer) and the excess over it: on near-deterministic domains a student can
score the class prior without learning anything (code review X-C1).

Std convention, everywhere in reports: the sample standard deviation across seeds (n - 1),
None for a single run.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from library.shared.config import RUNS_DIR, batches_dir
from library.shared.logging.reader import RunLogs, read_run_logs
from library.shared.metrics.task import answer_type_accuracy
from library.specs import DEFAULT_MAX_TEST_EXAMPLES
from .significance import welch_t_test

FULL_GEMMA_INPUT_VOCAB = 262144

# info.json keys that are not configuration: identity, status, timing, what the trainer
# measured, and the replicate seeds. Runs equal on everything else are one cohort.
NON_CONFIGURATION_KEYS = {
    "run_name", "description", "experiment", "status", "started_at", "completed_at",
    "model_info", "longest_training_sequence", "split_rule", "training_batch_count",
    "test_batch_count", "vocabulary_sha1", "seed", "shuffle_seed",
}


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
        return self.info.get("hidden_dim") or 0   # None for a custom model

    @property
    def num_layers(self) -> int:
        return self.info.get("num_layers") or 0

    @property
    def num_heads(self) -> int:
        return self.info.get("num_heads") or 0

    @property
    def custom_model(self) -> Optional[str]:
        """The custom model's class name, or None for the built-in Transformer."""
        path = self.info.get("model")
        return path.rsplit(".", 1)[-1] if path else None

    @property
    def architecture(self) -> str:
        """hidden / layers / heads of the built-in Transformer, or the custom model's class name."""
        return self.custom_model or f"{self.hidden_dim}/{self.num_layers}/{self.num_heads}"

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

    def configuration(self) -> dict:
        """Everything in info.json that defines what was trained, without the seed."""
        return {key: value for key, value in self.info.items() if key not in NON_CONFIGURATION_KEYS}

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

    # ── data: which batch files the run trained and was tested on ─────
    def split_files(self) -> Tuple[List[Path], List[Path]]:
        """(training_files, test_files) by the trainer's own rule (split_batch_files), from
        this run's info.json. Read each file through select_records(records, role). Runs
        from before the fixed test tail recorded no max_test_examples and get the default,
        so a re-evaluation scores every run on the same fixed test set."""
        from library.training.batch_handler import split_batch_files
        return split_batch_files(
            batches_dir(self.domain, self.teacher_model),
            self.info.get("training_test_ratio", 0.8),
            max_training_examples=self.info.get("max_training_examples"),
            batch_size=self.info.get("batch_size", 32),
            max_test_examples=self.info.get("max_test_examples", DEFAULT_MAX_TEST_EXAMPLES),
            training_groups=self.info.get("training_groups"),
        )

    def logged_test_files(self) -> List[Path]:
        """The batch files this run's logged evaluations were scored on — to describe what
        a logged number was measured on (its baseline), never to choose evaluation data.
        Runs recording split_rule "fixed_tail" used split_files(). Earlier runs used the
        rule before it: the last (1 - training_test_ratio) of the first
        max_training_examples // batch_size files, so their test set moved with
        max_training_examples (code review X-H1)."""
        if self.info.get("split_rule") == "fixed_tail":
            return self.split_files()[1]
        from library.shared.batch_files import batch_paths
        all_files = batch_paths(batches_dir(self.domain, self.teacher_model))
        max_examples = self.info.get("max_training_examples")
        batch_size = self.info.get("batch_size", 32)
        used_count = min(len(all_files), max_examples // batch_size) if max_examples else len(all_files)
        ratio = self.info.get("training_test_ratio", 0.8)
        return all_files[int(used_count * ratio):used_count]

    # ── derived metrics ───────────────────────────────────────────────
    @property
    def final_eval(self):
        return self.eval_epochs[-1] if self.eval_epochs else None

    def majority_baseline(self, evaluation=None) -> Optional[float]:
        """The task accuracy of always giving the most common answer on the test set the
        evaluation was scored on: the floor task accuracy has to clear (code review X-C1).
        `evaluation` is one of eval_epochs (default: the final one). Read from the log when
        logged; for older records computed from the logged confusion matrices; for logs
        with none (math before the fix) from the true answers of the run's logged test set.
        None for free-form domains (post-gen: no answer set) or when the batches are absent."""
        evaluation = evaluation or self.final_eval
        if evaluation is None:
            return None
        if evaluation.majority_baseline is not None:
            return evaluation.majority_baseline
        from library.training.evaluator import majority_baseline
        if evaluation.confusion_matrices:
            return majority_baseline(evaluation.confusion_matrices)
        if self.info.get("split_rule") == "fixed_tail":
            return None   # the current evaluator logs a baseline whenever the task metric has one
        return majority_baseline(self.test_set_answer_counts())

    def task_accuracy_excess(self, evaluation=None) -> Optional[float]:
        """Task accuracy minus the majority baseline (positive = learned beyond the prior)."""
        evaluation = evaluation or self.final_eval
        if evaluation is None:
            return None
        if evaluation.task_accuracy_excess is not None:
            return evaluation.task_accuracy_excess
        baseline = self.majority_baseline(evaluation)
        return evaluation.task_accuracy - baseline if baseline is not None else None

    def answer_type_accuracy(self, evaluation=None) -> Dict[str, Optional[float]]:
        """Math: accuracy on the true/false and on the numeric questions — only where the
        log holds the "solution" confusion matrix (eval records written since the fix);
        {} otherwise. Analysis(run).evaluate() measures it for older runs."""
        evaluation = evaluation or self.final_eval
        return answer_type_accuracy(evaluation.confusion_matrices) if evaluation else {}

    def test_set_answer_counts(self) -> dict:
        """Confusion matrices whose rows count the true answers of the logged test set
        (every prediction counted invalid): enough for the majority baseline and the
        answer-type composition. Read from the batch texts (no logits), cached per test set."""
        from library.domain import get_domain
        files = tuple(self.logged_test_files())
        key = (self.domain, files)
        if key not in _TEST_SET_ANSWER_COUNTS:
            metric = get_domain(self.domain).task_metric()
            for path in files:
                for record in _test_records(path):
                    metric.update([], record["model_response"])
            _TEST_SET_ANSWER_COUNTS[key] = metric.result().confusion or {}
        return _TEST_SET_ANSWER_COUNTS[key]

    @property
    def started_at(self) -> Optional[str]:
        return self.info.get("started_at")

    @property
    def completed_at(self) -> Optional[str]:
        return self.info.get("completed_at")

    @property
    def training_hours(self) -> Optional[float]:
        """Wall time over every session of the run. state.json accumulates it across resumes;
        info.json's started_at is reset by each session, so started -> completed counts only
        the last one (the fallback for runs without a state.json)."""
        state_path = self.run_dir / "logs" / "state.json"
        if state_path.exists():
            with open(state_path, "r", encoding="utf-8") as file:
                runtime_seconds = json.load(file).get("total_runtime_seconds", 0)
            if runtime_seconds:
                return runtime_seconds / 3600
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


_TEST_SET_ANSWER_COUNTS: Dict[tuple, dict] = {}   # (domain, test files) -> answer-count matrices


def _test_records(path: Path) -> List[dict]:
    """The test records of one batch file, texts only (select_records keeps the "test"
    ones of a file carrying per-record split labels)."""
    from library.shared.batch_files import read_batch_meta, read_batch_texts
    from library.training.batch_handler import TEST, select_records
    texts = read_batch_texts(path)
    splits = read_batch_meta(path).get("split") or [None] * len(texts)
    return select_records([{**text, "split": split} for text, split in zip(texts, splits)], TEST)


def test_split_teacher_entropy(runs: Sequence[Run]) -> Optional[dict]:
    """Teacher entropy as every report quotes it (code review X-M8): the mean per-token
    entropy (nats) of the teacher's reduced-vocabulary distribution at T=1 over each run's
    test split, read from the final eval record, averaged over `runs`. The returned
    "sample" says which runs and how many tokens that is."""
    records = [run.final_eval for run in runs
               if run.final_eval is not None and run.final_eval.teacher_entropy is not None]
    if not records:
        return None
    nats = statistics.mean(record.teacher_entropy for record in records)
    tokens = statistics.mean(record.total_steps for record in records)
    return {"nats": nats, "exp_nats": math.exp(nats), "runs": len(records), "tokens_per_run": tokens,
            "sample": f"test split, final eval record of {len(records)} run(s), ~{tokens:,.0f} teacher tokens each"}


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
    if metric == "majority_baseline":
        return run.majority_baseline()
    if metric == "task_accuracy_excess":
        return run.task_accuracy_excess()
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
    mean: Optional[float]   # None for an empty cohort
    std: Optional[float]    # sample std (n - 1); None below two values
    n: int
    values: List[float]


@dataclass
class Comparison:
    metric: str
    mean_a: Optional[float]
    mean_b: Optional[float]
    cohens_d: Optional[float]   # None when a cohort has fewer than two values
    welch_t: Optional[float]
    p_value: Optional[float]    # two-sided Welch, uncorrected


class Experiment:
    """One seed-replicate cohort: completed runs of one `experiment` tag that share a
    configuration (RunStore.by_experiment). `tag` is the cohort's label."""

    def __init__(self, tag: str, runs: List[Run]):
        self.tag = tag
        self.runs = runs

    def __len__(self) -> int:
        return len(self.runs)

    def values(self, metric: str) -> List[float]:
        return [v for v in (run_metric(run, metric) for run in self.runs) if v is not None]

    def stats(self, metric: str) -> Stat:
        values = self.values(metric)
        mean = statistics.mean(values) if values else None
        std = statistics.stdev(values) if len(values) >= 2 else None
        return Stat(metric=metric, mean=mean, std=std, n=len(values), values=values)

    def compare(self, other: "Experiment", metric: str) -> Comparison:
        a, b = self.values(metric), other.values(metric)
        mean_a = statistics.mean(a) if a else None
        mean_b = statistics.mean(b) if b else None
        cohens_d = None
        if len(a) >= 2 and len(b) >= 2:
            sa, sb = statistics.stdev(a), statistics.stdev(b)
            pooled = math.sqrt(((len(a) - 1) * sa ** 2 + (len(b) - 1) * sb ** 2) / (len(a) + len(b) - 2))
            cohens_d = (mean_a - mean_b) / pooled if pooled else None
        welch = welch_t_test(a, b)
        return Comparison(metric=metric, mean_a=mean_a, mean_b=mean_b, cohens_d=cohens_d,
                          welch_t=welch.t if welch else None, p_value=welch.p_value if welch else None)


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
        """{label: Experiment} — one cohort per experiment tag and configuration, completed
        runs only. An experiment script tags its whole sweep (every domain and arm), so the
        tag alone would pool different configurations; runs differing only in their seed
        (run name, description, shuffle_seed) share a cohort. The label is the tag plus the
        settings that tell that tag's cohorts apart."""
        groups: Dict[Tuple[str, str], List[Run]] = {}
        for run in self.completed():
            if run.experiment:
                key = (run.experiment, json.dumps(run.configuration(), sort_keys=True))
                groups.setdefault(key, []).append(run)
        cohorts = {}
        for (tag, configuration_json), runs in groups.items():
            label = _cohort_label(tag, json.loads(configuration_json),
                                  [json.loads(key[1]) for key in groups if key[0] == tag])
            cohorts[label] = Experiment(label, runs)
        return cohorts

    def experiment(self, tag: str) -> Experiment:
        """Every completed run of one experiment tag, all configurations together."""
        return Experiment(tag, self.filter(experiment=tag, status="completed"))

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


def _cohort_label(tag: str, configuration: dict, sibling_configurations: List[dict]) -> str:
    """`tag` alone when the tag has one configuration, else `tag key=value ...` for the
    keys whose values differ between the tag's configurations."""
    varying = sorted({key for other in sibling_configurations for key in set(other) | set(configuration)
                      if other.get(key) != configuration.get(key)})
    if not varying:
        return tag
    return tag + " " + " ".join(f"{key}={configuration.get(key)}" for key in varying)


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
