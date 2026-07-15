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
        ax.plot(index[window - 1:][: len(smoothed)], smoothed, color=color, linewidth=2)
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
    """Bar chart of per-cohort mean ± std (the thesis loss-strategy figures)."""
    plt = _plt()
    labels = [e.tag for e in experiments]
    stats = [e.stats(metric) for e in experiments]
    means = [s.mean * 100 for s in stats]
    errors = [s.std * 100 for s in stats]
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
