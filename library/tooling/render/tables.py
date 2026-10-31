"""Text tables and summaries built from Run / Experiment objects.

Task accuracy is always shown beside its majority baseline (always giving the most common
answer) and the excess over it, in percentage points (code review X-C1). Cohort spreads are
the sample std across seeds (n - 1).
"""

from typing import List, Sequence

from ..run import Experiment, Run, run_metric


def _pct(value) -> str:
    return f"{value * 100:.2f}%" if value is not None else "—"


def _points(value) -> str:
    return f"{value * 100:+.2f}pp" if value is not None else "—"


def _num(value, fmt="{:.4f}") -> str:
    return fmt.format(value) if value is not None else "—"


def _architecture(run: Run) -> str:
    if run.custom_model:
        return f"custom {run.custom_model} {run.info.get('model_settings', {})}"
    return f"{run.hidden_dim}h / {run.num_layers}L / {run.num_heads} heads"


def run_summary(run: Run) -> str:
    logs = run.logs
    lines = [
        f"Run: {run.name}",
        f"  Domain: {run.domain}   Status: {run.status}   Experiment: {run.experiment or '—'}",
        f"  Arch: {_architecture(run)}   "
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
            f"(majority baseline {_pct(run.majority_baseline(final))}, "
            f"excess {_points(run.task_accuracy_excess(final))}) loss={_num(final.avg_loss)}"
        )
        answer_types = run.answer_type_accuracy(final)
        if answer_types:
            lines.append(
                f"  Answer types: true/false {_pct(answer_types['boolean_accuracy'])} of "
                f"{answer_types['boolean_examples']} (always the commoner value: "
                f"{_pct(answer_types['boolean_majority_baseline'])})   numeric "
                f"{_pct(answer_types['numeric_accuracy'])} of {answer_types['numeric_examples']}"
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
    ("h/L/H", lambda r: r.architecture, 12),
    ("student", lambda r: _pct(run_metric(r, "student_accuracy")), 9),
    ("task", lambda r: _pct(run_metric(r, "task_accuracy")), 9),
    ("baseline", lambda r: _pct(run_metric(r, "majority_baseline")), 9),
    ("excess", lambda r: _points(run_metric(r, "task_accuracy_excess")), 9),
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
                       metrics: Sequence[str] = ("task_accuracy", "majority_baseline", "task_accuracy_excess",
                                                 "student_accuracy", "teacher_forced_accuracy")) -> str:
    """One line per cohort: n, then mean ± sample std (in %) per metric."""
    rows = []
    for experiment in experiments:
        parts = [f"{experiment.tag:<28} n={len(experiment):<3}"]
        for metric in metrics:
            stat = experiment.stats(metric)
            mean = f"{stat.mean * 100:.2f}%" if stat.mean is not None else "—"
            spread = f"±{stat.std * 100:.2f}" if stat.std is not None else ""
            parts.append(f"{metric}={mean}{spread}")
        rows.append("  ".join(parts))
    return "\n".join(rows)
