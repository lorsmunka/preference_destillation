"""Text tables and summaries built from Run / Experiment objects."""

from typing import List, Sequence

from ..run import Experiment, Run, run_metric


def _pct(value) -> str:
    return f"{value * 100:.2f}%" if value is not None else "—"


def _num(value, fmt="{:.4f}") -> str:
    return fmt.format(value) if value is not None else "—"


def run_summary(run: Run) -> str:
    logs = run.logs
    lines = [
        f"Run: {run.name}",
        f"  Domain: {run.domain}   Status: {run.status}   Experiment: {run.experiment or '—'}",
        f"  Arch: {run.hidden_dim}h / {run.num_layers}L / {run.num_heads} heads   "
        f"Params: {run.params_millions():.2f}M   reduced_input_vocab={run.has_reduced_input_vocab}",
        f"  Batches: {len(logs.train_batches):,}   Epochs: {len(logs.train_epochs)}   "
        f"Training hours: {_num(run.training_hours, '{:.2f}')}",
        f"  Final train: loss={_num(run.final_train_loss)} acc={_pct(run.final_train_accuracy)}",
    ]
    final = run.final_eval
    if final:
        lines.append(
            f"  Final eval: tf={_pct(final.teacher_forced_accuracy)} "
            f"student={_pct(final.student_accuracy)} task={_pct(final.task_accuracy)} "
            f"loss={_num(final.avg_loss)}"
        )
        if final.validity_rate is not None or final.topk_accuracy is not None:
            lines.append(
                f"  Eval-time extras: validity={_pct(final.validity_rate)} "
                f"topk={_pct(final.topk_accuracy)} teacher_entropy={_num(final.teacher_entropy, '{:.3f}')} "
                f"termination={_pct(final.termination_rate)}"
            )
    return "\n".join(lines)


_DEFAULT_COLUMNS = [
    ("run", lambda r: r.short_name(), 30),
    ("params(M)", lambda r: f"{r.params_millions():.1f}", 10),
    ("h/L/H", lambda r: f"{r.hidden_dim}/{r.num_layers}/{r.num_heads}", 12),
    ("student", lambda r: _pct(run_metric(r, "student_accuracy")), 9),
    ("task", lambda r: _pct(run_metric(r, "task_accuracy")), 9),
    ("tf", lambda r: _pct(run_metric(r, "teacher_forced_accuracy")), 9),
    ("eval_loss", lambda r: _num(run_metric(r, "avg_loss")), 10),
    ("hours", lambda r: _num(r.training_hours, "{:.1f}"), 7),
]


def comparison_table(runs: Sequence[Run]) -> str:
    header = "  ".join(label.ljust(width) for label, _, width in _DEFAULT_COLUMNS)
    rows = [header, "-" * len(header)]
    for run in runs:
        rows.append("  ".join(str(getter(run)).ljust(width) for _, getter, width in _DEFAULT_COLUMNS))
    return "\n".join(rows)


def experiment_summary(experiments: List[Experiment],
                       metrics: Sequence[str] = ("student_accuracy", "task_accuracy", "teacher_forced_accuracy")) -> str:
    rows = []
    for experiment in experiments:
        parts = [f"{experiment.tag:<28} n={len(experiment):<3}"]
        for metric in metrics:
            stat = experiment.stats(metric)
            parts.append(f"{metric}={stat.mean * 100:.2f}%±{stat.std * 100:.2f}")
        rows.append("  ".join(parts))
    return "\n".join(rows)
