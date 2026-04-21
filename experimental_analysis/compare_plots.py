import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from experimental_analysis.run_data import (
    RunData,
    load_all_runs,
    filter_runs,
    sort_runs,
)


OUTPUT_DIR = Path(__file__).parent / "plots"


COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78",
]


def get_color(index: int) -> str:
    return COLORS[index % len(COLORS)]


def moving_average(values, window: int):
    if len(values) < window:
        return np.array(values)
    return np.convolve(values, np.ones(window) / window, mode="valid")


def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig, filename: str, show: bool = False):
    ensure_output_dir()
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    if show:
        import subprocess
        subprocess.Popen(["start", str(path)], shell=True)


def make_legend_label(run: RunData) -> str:
    params = f"{run.params_millions():.1f}M"
    return f"{run.short_name()} ({params})"


# ============================================================
# Chart 1: Accuracy vs Parameters (Scaling Curve)
# ============================================================
def plot_accuracy_vs_params(runs: List[RunData], show: bool = False):
    eval_runs = [run for run in runs if run.final_eval]
    if not eval_runs:
        print("  No eval data for accuracy vs params chart.")
        return

    eval_runs = sorted(eval_runs, key=lambda run: run.total_parameters)

    params = [run.params_millions() for run in eval_runs]
    student_accuracy = [run.final_eval.student_accuracy * 100 for run in eval_runs]
    teacher_forced_accuracy = [run.final_eval.teacher_forced_accuracy * 100 for run in eval_runs]
    classification_accuracy = [run.final_eval.classification_accuracy * 100 for run in eval_runs]

    fig, ax = plt.subplots(figsize=(12, 7))

    series = [
        (teacher_forced_accuracy, "Teacher-Forced", "#2ca02c", "o"),
        (student_accuracy, "Student", "#1f77b4", "s"),
        (classification_accuracy, "Classification", "#ff7f0e", "^"),
    ]

    for values, label, color, marker in series:
        ax.plot(params, values, marker=marker, color=color, linewidth=2,
                markersize=8, label=label, zorder=3)

        for x_value, y_value, run in zip(params, values, eval_runs):
            ax.annotate(
                f"{run.hidden_dim}h/{run.num_layers}L",
                (x_value, y_value),
                textcoords="offset points",
                xytext=(0, 12),
                ha="center",
                fontsize=7,
                color=color,
                alpha=0.8,
            )

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}M"))
    ax.set_xlabel("Parameters (millions)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy vs Model Size", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Mark reduced vocab runs differently
    reduced_runs = [run for run in eval_runs if run.has_reduced_input_vocab]
    if reduced_runs:
        for run in reduced_runs:
            index = eval_runs.index(run)
            for values, _, color, _ in series:
                ax.plot(params[index], values[index], marker="D", color=color,
                        markersize=12, fillstyle="none", linewidth=2, zorder=4)
        ax.plot([], [], marker="D", color="gray", markersize=10, fillstyle="none",
                linewidth=2, linestyle="none", label="Reduced input vocab")
        ax.legend(fontsize=11)

    fig.tight_layout()
    save_figure(fig, "accuracy_vs_params.png", show=show)


# ============================================================
# Chart 2: Training Accuracy Over Time (Overlay)
# ============================================================
def plot_accuracy_over_training(runs: List[RunData], show: bool = False):
    runs_with_data = [run for run in runs if run.train_batches]
    if not runs_with_data:
        print("  No training data for accuracy overlay chart.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for index, run in enumerate(runs_with_data):
        color = get_color(index)
        label = make_legend_label(run)
        accuracies = [batch.accuracy * 100 for batch in run.train_batches]
        batch_indices = list(range(1, len(accuracies) + 1))

        window = min(max(50, len(accuracies) // 50), len(accuracies))
        smoothed = moving_average(accuracies, window)

        # Left: full training
        axes[0].plot(batch_indices[window - 1:], smoothed, color=color,
                     linewidth=1.5, label=label, alpha=0.9)

        # Right: last 20% for convergence detail
        cutoff = int(len(accuracies) * 0.8)
        if cutoff < len(accuracies):
            late_accuracies = accuracies[cutoff:]
            late_indices = batch_indices[cutoff:]
            late_window = min(max(20, len(late_accuracies) // 20), len(late_accuracies))
            late_smoothed = moving_average(late_accuracies, late_window)
            axes[1].plot(late_indices[late_window - 1:], late_smoothed, color=color,
                         linewidth=1.5, label=label, alpha=0.9)

    axes[0].set_xlabel("Batch")
    axes[0].set_ylabel("Train Accuracy (%)")
    axes[0].set_title("Training Accuracy (Full)")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Batch")
    axes[1].set_ylabel("Train Accuracy (%)")
    axes[1].set_title("Training Accuracy (Last 20%)")
    axes[1].legend(fontsize=8, loc="lower right")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Training Accuracy Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "accuracy_over_training.png", show=show)


# ============================================================
# Chart 3: Loss Over Training (Overlay)
# ============================================================
def plot_loss_over_training(runs: List[RunData], show: bool = False):
    runs_with_data = [run for run in runs if run.train_batches]
    if not runs_with_data:
        print("  No training data for loss overlay chart.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    loss_types = [
        ("loss", "Combined Loss", lambda batch: batch.loss),
        ("ce_loss", "CE Loss", lambda batch: batch.ce_loss),
        ("kl_loss", "KL Loss", lambda batch: batch.kl_loss),
    ]

    for ax, (loss_name, title, getter) in zip(axes, loss_types):
        for index, run in enumerate(runs_with_data):
            color = get_color(index)
            label = make_legend_label(run)
            losses = [getter(batch) for batch in run.train_batches]
            batch_indices = list(range(1, len(losses) + 1))

            window = min(max(50, len(losses) // 50), len(losses))
            smoothed = moving_average(losses, window)

            ax.plot(batch_indices[window - 1:], smoothed, color=color,
                    linewidth=1.5, label=label, alpha=0.9)

        ax.set_xlabel("Batch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

        # Y-axis: clip to last 80% range for readability
        all_late_values = []
        for run in runs_with_data:
            losses = [getter(batch) for batch in run.train_batches]
            cutoff = int(len(losses) * 0.2)
            all_late_values.extend(losses[cutoff:])
        if all_late_values:
            low = max(0, min(all_late_values) - 0.1)
            high = np.percentile(all_late_values, 95) * 1.1
            ax.set_ylim([low, high])

    fig.suptitle("Loss Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "loss_over_training.png", show=show)


# ============================================================
# Chart 4: Epoch Eval Grouped Bar Chart
# ============================================================
def plot_epoch_eval_bars(runs: List[RunData], show: bool = False):
    eval_runs = [run for run in runs if run.final_eval]
    if not eval_runs:
        print("  No eval data for bar chart.")
        return

    eval_runs = sorted(eval_runs, key=lambda run: run.total_parameters)

    labels = [run.short_name() for run in eval_runs]
    teacher_forced = [run.final_eval.teacher_forced_accuracy * 100 for run in eval_runs]
    student = [run.final_eval.student_accuracy * 100 for run in eval_runs]
    classification = [run.final_eval.classification_accuracy * 100 for run in eval_runs]

    x_positions = np.arange(len(labels))
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.5), 7))

    bars_teacher_forced = ax.bar(x_positions - bar_width, teacher_forced, bar_width,
                                  label="Teacher-Forced", color="#2ca02c", alpha=0.85)
    bars_student = ax.bar(x_positions, student, bar_width,
                          label="Student", color="#1f77b4", alpha=0.85)
    bars_classification = ax.bar(x_positions + bar_width, classification, bar_width,
                                  label="Classification", color="#ff7f0e", alpha=0.85)

    for bars in [bars_teacher_forced, bars_student, bars_classification]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Run")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Final Evaluation Accuracy by Run", fontsize=14, fontweight="bold")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    all_values = teacher_forced + student + classification
    if all_values:
        ax.set_ylim([max(0, min(all_values) - 10), min(100, max(all_values) + 5)])

    fig.tight_layout()
    save_figure(fig, "epoch_eval_bars.png", show=show)


# ============================================================
# Chart 5: Training Efficiency (Accuracy vs Wall Time)
# ============================================================
def plot_training_efficiency(runs: List[RunData], show: bool = False):
    runs_with_time = [run for run in runs if run.train_batches and run.final_eval]
    if not runs_with_time:
        print("  No data for efficiency chart.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    for index, run in enumerate(runs_with_time):
        color = get_color(index)
        label = make_legend_label(run)

        cumulative_hours = []
        accuracies = []
        total_seconds = 0
        window = min(max(50, len(run.train_batches) // 50), len(run.train_batches))

        for batch in run.train_batches:
            total_seconds += batch.time_seconds
            cumulative_hours.append(total_seconds / 3600)
            accuracies.append(batch.accuracy * 100)

        smoothed = moving_average(accuracies, window)
        hours_smoothed = cumulative_hours[window - 1:]

        ax.plot(hours_smoothed, smoothed, color=color, linewidth=1.5,
                label=label, alpha=0.9)

    ax.set_xlabel("Wall Time (hours)", fontsize=12)
    ax.set_ylabel("Train Accuracy (%)", fontsize=12)
    ax.set_title("Training Efficiency: Accuracy vs Time", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, "training_efficiency.png", show=show)


# ============================================================
# Chart 6: Reduced vs Full Vocab Comparison
# ============================================================
def plot_vocab_comparison(runs: List[RunData], show: bool = False):
    reduced_runs = [run for run in runs if run.has_reduced_input_vocab and run.final_eval]
    full_runs = [run for run in runs if not run.has_reduced_input_vocab and run.final_eval]

    if not reduced_runs or not full_runs:
        print("  Need both reduced and full vocab runs for comparison.")
        return

    # Match by approximate param count
    pairs = []
    for reduced_run in reduced_runs:
        closest = min(full_runs, key=lambda full: abs(full.total_parameters - reduced_run.total_parameters))
        ratio = max(reduced_run.total_parameters, closest.total_parameters) / max(1, min(reduced_run.total_parameters, closest.total_parameters))
        if ratio < 3:
            pairs.append((closest, reduced_run))

    if not pairs:
        print("  No matching full/reduced vocab run pairs found.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    metrics = [
        ("Teacher-Forced", lambda run: run.final_eval.teacher_forced_accuracy * 100),
        ("Student", lambda run: run.final_eval.student_accuracy * 100),
        ("Classification", lambda run: run.final_eval.classification_accuracy * 100),
    ]

    x_positions = np.arange(len(pairs))
    bar_width = 0.35

    for ax, (metric_name, getter) in zip(axes, metrics):
        full_values = [getter(full) for full, _ in pairs]
        reduced_values = [getter(reduced) for _, reduced in pairs]
        labels = [f"~{full.params_millions():.0f}M" for full, _ in pairs]

        ax.bar(x_positions - bar_width / 2, full_values, bar_width,
               label="Full vocab", color="#1f77b4", alpha=0.85)
        ax.bar(x_positions + bar_width / 2, reduced_values, bar_width,
               label="Reduced vocab", color="#ff7f0e", alpha=0.85)

        for x_val, full_val, reduced_val in zip(x_positions, full_values, reduced_values):
            delta = reduced_val - full_val
            sign = "+" if delta >= 0 else ""
            ax.annotate(f"{sign}{delta:.1f}pp",
                        xy=(x_val, max(full_val, reduced_val)),
                        xytext=(0, 10), textcoords="offset points",
                        ha="center", fontsize=9, fontweight="bold",
                        color="green" if delta >= 0 else "red")

        ax.set_xlabel("Param Budget")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{metric_name} Accuracy")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

        all_values = full_values + reduced_values
        ax.set_ylim([max(0, min(all_values) - 10), min(100, max(all_values) + 8)])

    fig.suptitle("Reduced vs Full Input Vocabulary", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "vocab_comparison.png", show=show)


# ============================================================
# Chart 7: Annealing Strategy Comparison
# ============================================================
def plot_annealing_comparison(runs: List[RunData], show: bool = False):
    anneal_runs = [run for run in runs if run.mini_evals and len(run.mini_evals) > 1]
    if not anneal_runs:
        anneal_runs = [run for run in runs if run.eval_epochs]
    if not anneal_runs:
        print("  No annealing comparison data available.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: mini-eval student accuracy over training
    for index, run in enumerate(anneal_runs):
        color = get_color(index)
        label = make_legend_label(run)

        if run.mini_evals:
            batches = [mini_eval.batch for mini_eval in run.mini_evals]
            student_accuracy = [mini_eval.student_accuracy * 100 for mini_eval in run.mini_evals]
            axes[0].plot(batches, student_accuracy, color=color, linewidth=2,
                         marker="o", markersize=3, label=label, alpha=0.9)

    axes[0].set_xlabel("Batch")
    axes[0].set_ylabel("Student Accuracy (%)")
    axes[0].set_title("Mini-Eval Student Accuracy")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Right: epoch eval comparison
    for index, run in enumerate(anneal_runs):
        color = get_color(index)
        label = make_legend_label(run)

        if run.eval_epochs:
            epochs = [epoch_eval.epoch for epoch_eval in run.eval_epochs]
            student_accuracy = [epoch_eval.student_accuracy * 100 for epoch_eval in run.eval_epochs]
            axes[1].plot(epochs, student_accuracy, color=color, linewidth=2,
                         marker="s", markersize=6, label=label, alpha=0.9)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Student Accuracy (%)")
    axes[1].set_title("Epoch Eval Student Accuracy")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Annealing Strategy Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "annealing_comparison.png", show=show)


# ============================================================
# Chart 8: Mini-Eval Trajectory Overlay
# ============================================================
def plot_mini_eval_overlay(runs: List[RunData], show: bool = False):
    runs_with_mini = [run for run in runs if run.mini_evals]
    if not runs_with_mini:
        print("  No mini-eval data for overlay chart.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    metric_series = [
        ("Teacher-Forced", lambda mini_eval: mini_eval.teacher_forced_accuracy * 100),
        ("Student", lambda mini_eval: mini_eval.student_accuracy * 100),
        ("Classification", lambda mini_eval: mini_eval.classification_accuracy * 100),
    ]

    for ax, (metric_name, getter) in zip(axes, metric_series):
        for index, run in enumerate(runs_with_mini):
            color = get_color(index)
            label = make_legend_label(run)
            batches = [mini_eval.batch for mini_eval in run.mini_evals]
            values = [getter(mini_eval) for mini_eval in run.mini_evals]
            ax.plot(batches, values, color=color, linewidth=1.5,
                    marker="o", markersize=2, label=label, alpha=0.9)

        ax.set_xlabel("Batch")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"Mini-Eval: {metric_name}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Mini-Eval Trajectory Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "mini_eval_overlay.png", show=show)


# ============================================================
# Chart 9: Confusion Matrix Side-by-Side
# ============================================================
def plot_confusion_comparison(runs: List[RunData], category: str = "sentiment", show: bool = False):
    eval_runs = [run for run in runs if run.final_eval and run.final_eval.confusion_matrices]
    if not eval_runs:
        print("  No confusion matrix data.")
        return

    matching_runs = [
        run for run in eval_runs
        if category in run.final_eval.confusion_matrices
    ]
    if not matching_runs:
        print(f"  No runs have confusion data for '{category}'.")
        return

    count = len(matching_runs)
    cols = min(count, 4)
    rows_count = (count + cols - 1) // cols
    fig, axes = plt.subplots(rows_count, cols, figsize=(5 * cols, 5 * rows_count))
    if count == 1:
        axes = np.array([axes])
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("white_blue", ["white", "#4285f4"])

    for index, run in enumerate(matching_runs):
        ax = axes[index]
        matrix_data = run.final_eval.confusion_matrices[category]
        labels = list(matrix_data.keys())
        values = np.array([
            [matrix_data[true_label].get(pred_label, 0) for pred_label in labels]
            for true_label in labels
        ])

        ax.imshow(values, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground Truth")
        ax.set_title(f"{run.short_name()} ({run.params_millions():.0f}M)", fontsize=10)

        for row_index in range(len(labels)):
            for col_index in range(len(labels)):
                cell_value = values[row_index, col_index]
                if cell_value > 0:
                    text_color = "white" if cell_value > values.max() * 0.6 else "black"
                    ax.text(col_index, row_index, str(int(cell_value)),
                            ha="center", va="center", fontsize=8, color=text_color)

    for index in range(count, len(axes)):
        axes[index].axis("off")

    fig.suptitle(f"Confusion Matrices: {category.capitalize()}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, f"confusion_{category}.png", show=show)


# ============================================================
# Generate All Charts
# ============================================================
def generate_all_charts(runs: List[RunData], show: bool = False):
    print("\nGenerating comparison charts...")

    print("\n[1/9] Accuracy vs Parameters")
    plot_accuracy_vs_params(runs, show=show)

    print("[2/9] Training Accuracy Overlay")
    plot_accuracy_over_training(runs, show=show)

    print("[3/9] Loss Overlay")
    plot_loss_over_training(runs, show=show)

    print("[4/9] Epoch Eval Bars")
    plot_epoch_eval_bars(runs, show=show)

    print("[5/9] Training Efficiency")
    plot_training_efficiency(runs, show=show)

    print("[6/9] Vocab Comparison")
    plot_vocab_comparison(runs, show=show)

    print("[7/9] Annealing Comparison")
    plot_annealing_comparison(runs, show=show)

    print("[8/9] Mini-Eval Overlay")
    plot_mini_eval_overlay(runs, show=show)

    print("[9/9] Confusion Matrices")
    for category in ["tone", "sentiment", "safety", "toxicity"]:
        plot_confusion_comparison(runs, category=category, show=show)

    print(f"\nAll charts saved to: {OUTPUT_DIR}")


def main():
    args = sys.argv[1:]

    prefix = None
    names = []
    show_all = False
    completed_only = False
    show = False
    chart_type = None

    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--all":
            show_all = True
        elif arg == "--completed":
            completed_only = True
        elif arg == "--show":
            show = True
        elif arg == "--prefix" and index + 1 < len(args):
            index += 1
            prefix = args[index]
        elif arg == "--chart" and index + 1 < len(args):
            index += 1
            chart_type = args[index]
        elif not arg.startswith("--"):
            names.append(arg)
        index += 1

    print("Loading runs...")
    all_runs = load_all_runs()
    status_filter = "completed" if completed_only else None
    runs = filter_runs(all_runs, prefix=prefix, status=status_filter,
                       names=names if names else None)
    runs = sort_runs(runs, sort_by="params")

    if not runs:
        print("No runs match filters.")
        return

    print(f"Plotting {len(runs)} runs.")

    chart_functions = {
        "scaling": plot_accuracy_vs_params,
        "accuracy": plot_accuracy_over_training,
        "loss": plot_loss_over_training,
        "bars": plot_epoch_eval_bars,
        "efficiency": plot_training_efficiency,
        "vocab": plot_vocab_comparison,
        "annealing": plot_annealing_comparison,
        "mini_eval": plot_mini_eval_overlay,
    }

    if chart_type and chart_type in chart_functions:
        chart_functions[chart_type](runs, show=show)
    else:
        generate_all_charts(runs, show=show)


if __name__ == "__main__":
    main()
