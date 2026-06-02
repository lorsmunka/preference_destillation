"""Analysis — the single researcher-facing facade.

    from library import Analysis, RunStore
    a = Analysis("tsc-scale-128h-4L-36M")
    print(a.summary())
    a.plot_training()
    a.evaluate(metrics=["token", "task", "topk"])      # model path
    Analysis.compare(RunStore().completed()).table()
    Analysis.cohorts()                                  # {tag: Experiment}

Pure-log methods (summary/training/plot/compare/experiment) never touch the model. Model
methods (evaluate/examples/infer) lazily build a StudentModel and are the only things that
reload weights — the demo/eval/test path, never deployment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from . import render
from library.shared.config import batches_dir
from .run import Experiment, Run, RunStore, sort_runs

RunRef = Union[str, Run]


# ── evaluation result ─────────────────────────────────────────────────
@dataclass
class EvalResult:
    run: str
    checkpoint: str
    split: str
    examples: int
    steps: int
    teacher_forced_accuracy: float
    student_accuracy: float
    task_accuracy: float
    validity_rate: Optional[float]
    topk_accuracy: float
    teacher_student_overlap: float
    mean_target_rank: float
    teacher_entropy: float
    student_entropy: float
    perplexity: float
    termination_rate: Optional[float]
    mean_length_ratio: Optional[float]
    extra: Dict[str, float] = field(default_factory=dict)

    def distribution_fields(self) -> Dict[str, float]:
        """The subset to persist on the eval log record (compute-once principle)."""
        return {
            "topk_accuracy": self.topk_accuracy,
            "teacher_student_overlap": self.teacher_student_overlap,
            "mean_target_rank": self.mean_target_rank,
            "teacher_entropy": self.teacher_entropy,
            "student_entropy": self.student_entropy,
            "perplexity": self.perplexity,
            "validity_rate": self.validity_rate,
            "termination_rate": self.termination_rate,
            "mean_length_ratio": self.mean_length_ratio,
        }


def _batch_paths(run: Run, split: str, max_batches: Optional[int]) -> List[Path]:
    directory = batches_dir(run.domain, run.teacher_model)
    all_batches = sorted(directory.glob("batch_*.jsonl"), key=lambda p: int(p.stem.split("_")[-1]))
    total = len(all_batches)
    max_examples = run.info.get("max_training_examples")
    batch_size = run.info.get("batch_size", 32)
    effective = min(total, max_examples // batch_size) if max_examples else total
    ratio = run.info.get("training_test_ratio", 0.98)
    if split == "train":
        start, end = 0, int(effective * ratio)
    elif split == "test":
        start, end = int(effective * ratio), effective
    elif split == "all":
        start, end = 0, effective
    else:
        raise ValueError(f"Unknown split: {split!r}")
    selected = all_batches[start:end]
    return selected[:max_batches] if max_batches else selected


def _iter_examples(batch_path: Path):
    with open(batch_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


class Analysis:
    def __init__(self, run: RunRef, store: Optional[RunStore] = None):
        self.store = store or RunStore()
        self.run = run if isinstance(run, Run) else self.store.get(run)

    # ── log interpretation (no model) ─────────────────────────────────
    def summary(self) -> str:
        return render.run_summary(self.run)

    def print_summary(self) -> None:
        print(self.summary())

    def training(self):
        """Typed training logs (batches/epochs/evals/mini-evals)."""
        return self.run.logs

    def plot_training(self, out_path: Optional[Path] = None):
        from .render.plots import plot_training
        return plot_training(self.run, out_path)

    # ── multi-run / cohorts (no model) ────────────────────────────────
    @classmethod
    def compare(cls, runs: Sequence[RunRef], store: Optional[RunStore] = None) -> "Comparison":
        store = store or RunStore()
        resolved = [r if isinstance(r, Run) else store.get(r) for r in runs]
        return Comparison(resolved, store)

    @staticmethod
    def cohorts(store: Optional[RunStore] = None) -> Dict[str, Experiment]:
        return (store or RunStore()).by_experiment()

    @staticmethod
    def experiment(tag: str, store: Optional[RunStore] = None) -> Experiment:
        return (store or RunStore()).experiment(tag)

    # ── evaluation (model path) ───────────────────────────────────────
    def evaluate(self, split: str = "test", checkpoint: str = "latest", k: int = 20,
                 max_batches: Optional[int] = None, cap_multiple: float = 2.0) -> EvalResult:
        import torch
        import torch.nn.functional as F
        from library.shared.metrics.distribution import step_distribution_stats
        from library.shared.metrics.generation import TerminationStats
        from library.model.student import StudentModel

        label, path = self.store.resolve_checkpoint(self.run.name, checkpoint)
        student = StudentModel.from_run(self.run, path)
        domain = student.domain
        task_metric = domain.task_metric()
        termination = TerminationStats()

        totals = {"topk_hits": 0.0, "overlap_sum": 0.0, "target_rank_sum": 0.0,
                  "student_entropy_sum": 0.0, "teacher_entropy_sum": 0.0, "steps": 0}
        tf_correct = student_correct = ce_sum = 0.0
        examples = 0

        with torch.no_grad():
            for batch_path in _batch_paths(self.run, split, max_batches):
                for example in _iter_examples(batch_path):
                    forced = student.teacher_forced(example)
                    if forced is None:
                        continue
                    logits = forced["prediction_logits"]
                    targets = forced["target_indices"]
                    num_steps = targets.shape[0]

                    tf_correct += (logits.argmax(-1) == targets).sum().item()
                    ce_sum += F.cross_entropy(logits, targets, reduction="sum").item()
                    for key, value in step_distribution_stats(logits, forced["teacher_logits"], targets, k).items():
                        totals[key] += value

                    rollout = student.generate(example["sentence"], max_new_tokens=int(cap_multiple * num_steps))
                    task_metric.update(rollout.token_strings, example.get("model_response", ""))
                    termination.update(rollout.token_strings, domain.stop_token, num_steps)
                    student_correct += sum(
                        1 for i in range(num_steps)
                        if i < len(rollout.predicted_indices) and rollout.predicted_indices[i] == targets[i].item()
                    )
                    examples += 1

        steps = max(totals["steps"], 1)
        task = task_metric.result()
        return EvalResult(
            run=self.run.name, checkpoint=label, split=split, examples=examples, steps=totals["steps"],
            teacher_forced_accuracy=tf_correct / steps,
            student_accuracy=student_correct / steps,
            task_accuracy=task.accuracy,
            validity_rate=task.validity_rate,
            topk_accuracy=totals["topk_hits"] / steps,
            teacher_student_overlap=totals["overlap_sum"] / steps,
            mean_target_rank=totals["target_rank_sum"] / steps,
            teacher_entropy=totals["teacher_entropy_sum"] / steps,
            student_entropy=totals["student_entropy_sum"] / steps,
            perplexity=float(__import__("math").exp(ce_sum / steps)),
            termination_rate=termination.termination_rate,
            mean_length_ratio=termination.mean_length_ratio,
        )

    def infer(self, sentences: List[str], checkpoint: str = "latest", temperature: float = 0.0):
        """Demo/debug: generate the student's output for each sentence. Teacher comparison is
        intentionally left to a separate benchmark (loading Gemma is heavy)."""
        from library.model.student import StudentModel
        label, path = self.store.resolve_checkpoint(self.run.name, checkpoint)
        student = StudentModel.from_run(self.run, path)
        results = []
        for sentence in sentences:
            generation = student.generate(sentence, max_new_tokens=student.domain.max_steps, temperature=temperature)
            results.append({"sentence": sentence, "output": generation.text,
                            "terminated": generation.terminated, "steps": generation.steps})
        return {"run": self.run.name, "checkpoint": label, "results": results}


class Comparison:
    """Multi-run view (table / scaling / cohort plots)."""

    def __init__(self, runs: List[Run], store: Optional[RunStore] = None):
        self.runs = runs
        self.store = store or RunStore()

    def sort(self, by: str = "params") -> "Comparison":
        return Comparison(sort_runs(self.runs, by), self.store)

    def table(self) -> str:
        return render.comparison_table(self.runs)

    def print_table(self) -> None:
        print(self.table())

    def scaling(self, out_path: Path, metric: str = "student_accuracy"):
        from .render.plots import plot_scaling
        return plot_scaling(self.runs, out_path, metric)
