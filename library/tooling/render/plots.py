"""Plots — write PNGs from Run / Experiment data. Matplotlib is imported lazily with the
Agg backend so this works headless and `import library` never pulls in matplotlib.
"""

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from ..run import Experiment, Run, run_metric


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def moving_average(values: Sequence[float], window: int) -> np.ndarray:
    if len(values) < window or window <= 1:
        return np.asarray(values, dtype=float)
    return np.convolve(values, np.ones(window) / window, mode="valid")


def _render_training(batches, title: str, out_path: Path):
    plt = _plt()
    if not batches:
        raise ValueError("No training batches to plot")
    index = list(range(1, len(batches) + 1))
    window = max(40, int(len(batches) * 0.03))
    series = [
        ("Accuracy (%)", [b.accuracy * 100 for b in batches], "green"),
        ("Loss", [b.loss for b in batches], "red"),
        ("CE loss", [b.ce_loss for b in batches], "blue"),
        ("KL loss", [b.kl_loss for b in batches], "purple"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(title, fontweight="bold")
    for ax, (label, values, color) in zip(axes.flat, series):
        ax.plot(index, values, color=color, alpha=0.3, linewidth=0.6)
        smoothed = moving_average(values, window)
        # moving_average returns the values untouched when there are fewer than `window` of
        # them, so derive the x offset from the lengths instead of assuming a valid-mode
        # convolution — otherwise short runs plot an empty x against a full y.
        offset = len(values) - len(smoothed)
        ax.plot(index[offset:], smoothed, color=color, linewidth=2)
        ax.set_title(label)
        ax.set_xlabel("Batch")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_training(run: Run, out_path: Optional[Path] = None):
    """Single-run training curves from a Run (loss/accuracy over batches)."""
    out_path = Path(out_path) if out_path else (run.run_dir / "logs" / "training_progress.png")
    title = f"{run.name}  ·  {run.params_millions():.1f}M  ·  {run.domain}"
    return _render_training(run.logs.train_batches, title, out_path)


def plot_training_logs(logs_dir, out_path: Optional[Path] = None, title: str = ""):
    """Plot directly from a logs dir's training.jsonl — used live during training (no Run
    object / info.json needed). Replaces analysis.visualize_logs.update_training_plot."""
    from library.shared.logging.reader import read_training_records
    logs_dir = Path(logs_dir)
    logs = read_training_records(logs_dir / "training.jsonl")
    out_path = Path(out_path) if out_path else (logs_dir / "training_progress.png")
    return _render_training(logs.train_batches, title or logs_dir.parent.name, out_path)


SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # fixed order, colour-blind checked
INK = "#0b0b0b"
MUTED_INK = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"


def _end_label_heights(values, minimum_gap):
    """Spread the end-of-line labels so close lines (e.g. 69% and 68%) do not print on top of each other."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    heights = list(values)
    for previous, current in zip(order, order[1:]):
        heights[current] = max(heights[current], heights[previous] + minimum_gap)
    return heights


def plot_tests_by_pass(runs_by_label, out_path: Path, title: str = ""):
    """One point per test (after every pass) for each run: task accuracy next to the most common answer,
    and the test error. Two panels with their own scales; the label order fixes each run's colour.
    A label may hold several runs (seeds of one setup): the thick line is their mean over the passes
    every one of them has, a thin line in the same colour shows each run."""
    plt = _plt()
    labelled = []
    for label, runs in runs_by_label.items():
        runs = [run for run in (runs if isinstance(runs, (list, tuple)) else [runs]) if run.eval_epochs]
        if runs:
            labelled.append((label, runs))
    if not labelled:
        raise ValueError("No run has a test yet.")
    fig, (accuracy_axis, error_axis) = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=SURFACE)
    panels = [
        (accuracy_axis, "Task accuracy (all test replies)", lambda evaluation: evaluation.task_accuracy * 100, "%"),
        (error_axis, "Test error (cross-entropy, lower is better)", lambda evaluation: evaluation.ce_loss, ""),
    ]
    most_passes = max(len(run.eval_epochs) for _, runs in labelled for run in runs)
    for axis, panel_title, read_value, unit in panels:
        axis.set_facecolor(SURFACE)
        end_values = []
        for (label, runs), color in zip(labelled, SERIES_COLORS):
            if len(runs) > 1:
                for run in runs:
                    axis.plot(range(1, len(run.eval_epochs) + 1), [read_value(evaluation) for evaluation in run.eval_epochs],
                              color=color, linewidth=1, alpha=0.35, zorder=2)
            shared_passes = min(len(run.eval_epochs) for run in runs)
            passes = list(range(1, shared_passes + 1))
            values = [float(np.mean([read_value(run.eval_epochs[index]) for run in runs])) for index in range(shared_passes)]
            axis.plot(passes, values, color=color, linewidth=2, marker="o", markersize=6,
                      markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)
            end_values.append((label, passes[-1], values[-1]))
        low, high = axis.get_ylim()
        heights = _end_label_heights([value for _, _, value in end_values], (high - low) * 0.06)
        for (label, last_pass, value), height in zip(end_values, heights):
            shown = f"{value:.1f}{unit}" if unit else f"{value:.3f}"
            axis.annotate(f"{label} {shown}", (last_pass, value), xytext=(last_pass + 0.25, height),
                          color=INK, fontsize=9, va="center")
        axis.set_title(panel_title, color=INK, fontsize=11, loc="left")
        axis.set_xlabel("Pass", color=MUTED_INK)
        axis.set_xticks(range(1, most_passes + 1))
        axis.set_xlim(0.6, most_passes + 1.6)
        axis.grid(True, axis="y", color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(colors=MUTED_INK, labelcolor=MUTED_INK)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(AXIS)
    first_run = labelled[0][1][0]
    baseline = first_run.majority_baseline(first_run.eval_epochs[-1])
    if baseline is not None:
        baseline = baseline * 100
        accuracy_axis.axhline(baseline, color=MUTED_INK, linewidth=1.2, linestyle="--", zorder=2)
        accuracy_axis.annotate(f"most common answer {baseline:.1f}%", (1, baseline), xytext=(0, 5),
                               textcoords="offset points", color=MUTED_INK, fontsize=9)
    accuracy_axis.set_ylim(bottom=0)
    accuracy_axis.legend(frameon=False, loc="upper left", labelcolor=INK)
    if title:
        fig.suptitle(title, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out_path


def plot_scaling(runs: Sequence[Run], out_path: Path, metric: str = "student_accuracy"):
    """Accuracy vs parameter count (the central scaling figure)."""
    plt = _plt()
    points = sorted(
        ((r.params_millions(), run_metric(r, metric), r) for r in runs if run_metric(r, metric) is not None),
        key=lambda p: p[0],
    )
    if not points:
        raise ValueError("No runs with the requested metric.")
    xs = [p[0] for p in points]
    ys = [p[1] * 100 for p in points]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(xs, ys, marker="o", color="#4285f4")
    for x, y, run in points:
        ax.annotate(run.short_name(), (x, y), fontsize=7, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (M, log)")
    ax.set_ylabel(f"{metric} (%)")
    ax.set_title(f"Scaling: {metric} vs parameters")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_experiment_comparison(experiments: List[Experiment], out_path: Path,
                               metric: str = "student_accuracy"):
    """Bar chart of per-cohort mean ± sample std (the thesis loss-strategy figures).
    Cohorts without a value for the metric are left out."""
    plt = _plt()
    measured = [(experiment, experiment.stats(metric)) for experiment in experiments]
    measured = [(experiment, stat) for experiment, stat in measured if stat.mean is not None]
    if not measured:
        raise ValueError(f"No cohort has a value for {metric!r}.")
    labels = [experiment.tag for experiment, _ in measured]
    means = [stat.mean * 100 for _, stat in measured]
    errors = [(stat.std or 0.0) * 100 for _, stat in measured]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.6), 6))
    ax.bar(range(len(labels)), means, yerr=errors, capsize=5, color="#4C78A8")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(f"{metric} (%)")
    ax.set_title(f"Cohort comparison: {metric} (mean ± std)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
