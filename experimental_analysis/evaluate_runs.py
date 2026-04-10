import sys
import math
import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from experimental_analysis.run_data import (
    RunData,
    EpochEval,
    MiniEval,
    load_all_runs,
    filter_runs,
    sort_runs,
)


VIEWER_DIR = Path(__file__).parent / "templates"
JSON_OUTPUT_PATH = VIEWER_DIR / "run_evaluation.json"


# ── Metric helpers ──────────────────────────────────────────────────────────


def linear_regression_slope(x_values: list, y_values: list) -> Optional[float]:
    """Ordinary least squares slope. Returns None if too few points."""
    n = len(x_values)
    if n < 2:
        return None
    mean_x = sum(x_values) / n
    mean_y = sum(y_values) / n
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
    denominator = sum((x - mean_x) ** 2 for x in x_values)
    if denominator == 0:
        return None
    return numerator / denominator


def compute_mini_eval_slope(mini_evals: List[MiniEval], metric: str = "student_accuracy") -> Optional[float]:
    """Slope of mini-eval metric over the mini-eval sequence.
    Mini-evals are emitted at a fixed batch interval (MINI_EVAL_FREQUENCY in
    config, currently 1000 batches), so the index axis is uniform within and
    across runs. Units: accuracy-points per mini-eval checkpoint."""
    if len(mini_evals) < 3:
        return None
    x_values = list(range(len(mini_evals)))
    y_values = [getattr(mini_eval, metric) for mini_eval in mini_evals]
    return linear_regression_slope(x_values, y_values)


def compute_early_vs_late_gain(mini_evals: List[MiniEval], metric: str = "student_accuracy") -> Optional[float]:
    """Ratio of accuracy gain in first half vs second half of training.
    >1 means front-loaded learning, <1 means back-loaded."""
    if len(mini_evals) < 4:
        return None
    values = [getattr(me, metric) for me in mini_evals]
    midpoint = len(values) // 2
    first_half_gain = values[midpoint] - values[0]
    second_half_gain = values[-1] - values[midpoint]
    if second_half_gain == 0:
        return None
    return first_half_gain / second_half_gain


def compute_stability(mini_evals: List[MiniEval], metric: str = "student_accuracy") -> Optional[float]:
    """Standard deviation of the metric in the last epoch's mini-evals.
    Lower means more stable convergence."""
    if not mini_evals:
        return None
    last_epoch = mini_evals[-1].epoch
    last_epoch_values = [getattr(me, metric) for me in mini_evals if me.epoch == last_epoch]
    if len(last_epoch_values) < 2:
        return None
    mean = sum(last_epoch_values) / len(last_epoch_values)
    variance = sum((v - mean) ** 2 for v in last_epoch_values) / (len(last_epoch_values) - 1)
    return math.sqrt(variance)


def compute_convergence_speed(mini_evals: List[MiniEval], metric: str = "student_accuracy") -> Optional[float]:
    """Fraction of total training where 90% of the total accuracy gain has been achieved.
    Lower = faster convergence. Returns value between 0 and 1."""
    if len(mini_evals) < 3:
        return None
    values = [getattr(me, metric) for me in mini_evals]
    first_value = values[0]
    final_value = values[-1]
    total_gain = final_value - first_value
    if total_gain <= 0:
        return None
    target = first_value + total_gain * 0.9
    for index, value in enumerate(values):
        if value >= target:
            return (index + 1) / len(values)
    return 1.0


def compute_epoch_deltas(eval_epochs: List[EpochEval]) -> List[dict]:
    """Per-epoch accuracy improvements."""
    deltas = []
    for index in range(1, len(eval_epochs)):
        previous = eval_epochs[index - 1]
        current = eval_epochs[index]
        deltas.append({
            "epoch": current.epoch,
            "student_accuracy_delta": current.student_accuracy - previous.student_accuracy,
            "teacher_forced_accuracy_delta": current.teacher_forced_accuracy - previous.teacher_forced_accuracy,
            "classification_accuracy_delta": current.classification_accuracy - previous.classification_accuracy,
            "loss_delta": current.avg_loss - previous.avg_loss,
        })
    return deltas


def compute_f1_per_category(confusion_matrices: dict) -> dict:
    """Compute macro F1 score per classification category from confusion matrices."""
    results = {}
    for category, matrix in confusion_matrices.items():
        f1_scores = []
        for label in matrix:
            true_positive = matrix[label].get(label, 0)
            predicted_total = sum(
                row.get(label, 0) for row in matrix.values()
            )
            actual_total = sum(matrix[label].values())
            precision = true_positive / predicted_total if predicted_total > 0 else 0
            recall = true_positive / actual_total if actual_total > 0 else 0
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0
            f1_scores.append(f1)
        results[category] = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    return results


# ── Report building ─────────────────────────────────────────────────────────


def build_run_report(run: RunData) -> dict:
    """Extract all metrics for a single run."""
    report = {}

    # Identity
    report["name"] = run.name
    report["status"] = run.status
    report["params_millions"] = run.params_millions()
    report["hidden_dim"] = run.hidden_dim
    report["num_layers"] = run.num_layers
    report["num_heads"] = run.num_heads

    # Hyperparameters
    report["kl_ratio"] = f"{run.kl_ratio_start}->{run.kl_ratio_end}" if run.kl_ratio_start != run.kl_ratio_end else f"{run.kl_ratio_start}"
    report["temperature"] = f"{run.temperature_start}->{run.temperature_end}" if run.temperature_start != run.temperature_end else f"{run.temperature_start}"
    report["learning_rate"] = run.learning_rate
    report["dropout"] = run.dropout

    # Training stats
    report["total_batches"] = len(run.train_batches)
    report["training_hours"] = run.training_hours

    # Final train metrics (averaged over last 200 batches)
    report["final_train_loss"] = run.final_train_loss
    report["final_train_accuracy"] = run.final_train_accuracy

    # Final eval metrics — initialize all keys so downstream code can rely on
    # `.get(key)` returning None rather than missing entirely.
    report["final_eval_loss"] = None
    report["final_kl_loss"] = None
    report["final_ce_loss"] = None
    report["final_tf_accuracy"] = None
    report["final_student_accuracy"] = None
    report["final_classification_accuracy"] = None
    for category in ("tone", "sentiment", "safety", "toxicity"):
        report[f"f1_{category}"] = None
    report["f1_macro_avg"] = None

    if run.final_eval:
        report["final_eval_loss"] = run.final_eval.avg_loss
        report["final_kl_loss"] = run.final_eval.kl_loss
        report["final_ce_loss"] = run.final_eval.ce_loss
        report["final_tf_accuracy"] = run.final_eval.teacher_forced_accuracy
        report["final_student_accuracy"] = run.final_eval.student_accuracy
        report["final_classification_accuracy"] = run.final_eval.classification_accuracy

        if run.final_eval.confusion_matrices:
            f1_scores = compute_f1_per_category(run.final_eval.confusion_matrices)
            for category, score in f1_scores.items():
                report[f"f1_{category}"] = score
            if f1_scores:
                report["f1_macro_avg"] = sum(f1_scores.values()) / len(f1_scores)

    # Per-epoch progression
    epoch_deltas = compute_epoch_deltas(run.eval_epochs)
    for index, delta in enumerate(epoch_deltas):
        report[f"epoch{delta['epoch']}_student_delta"] = delta["student_accuracy_delta"]
        report[f"epoch{delta['epoch']}_class_delta"] = delta["classification_accuracy_delta"]

    # Mini-eval curve analysis
    report["student_slope"] = compute_mini_eval_slope(run.mini_evals, "student_accuracy")
    report["classification_slope"] = compute_mini_eval_slope(run.mini_evals, "classification_accuracy")
    report["early_vs_late_ratio"] = compute_early_vs_late_gain(run.mini_evals, "student_accuracy")
    report["convergence_speed_90"] = compute_convergence_speed(run.mini_evals, "student_accuracy")
    report["last_epoch_stability"] = compute_stability(run.mini_evals, "student_accuracy")

    # Efficiency metrics
    if run.final_eval:
        report["accuracy_per_million_params"] = run.final_eval.student_accuracy * 100 / run.params_millions() if run.params_millions() > 0 else None
        if run.training_hours and run.training_hours > 0:
            report["accuracy_per_hour"] = run.final_eval.student_accuracy * 100 / run.training_hours
        else:
            report["accuracy_per_hour"] = None
    else:
        report["accuracy_per_million_params"] = None
        report["accuracy_per_hour"] = None

    return report


# ── Column definitions ──────────────────────────────────────────────────────


# (label, report_key, format, higher_is_better, group)
# higher_is_better: True = best = max, False = best = min, None = no highlight
COLUMN_DEFINITIONS = [
    ("Status", "status", "auto", None, "Info"),
    ("Params (M)", "params_millions", "f1", None, "Architecture"),
    ("Hidden", "hidden_dim", "int", None, "Architecture"),
    ("Layers", "num_layers", "int", None, "Architecture"),
    ("Heads", "num_heads", "int", None, "Architecture"),
    ("KL ratio", "kl_ratio", "auto", None, "Hyperparams"),
    ("Temp", "temperature", "auto", None, "Hyperparams"),
    ("LR", "learning_rate", "sci", None, "Hyperparams"),
    ("Dropout", "dropout", "auto", None, "Hyperparams"),
    ("Batches", "total_batches", "int", None, "Training"),
    ("Wall time", "training_hours", "hours", None, "Training"),
    ("Train loss", "final_train_loss", "f4", False, "Training"),
    ("Train acc", "final_train_accuracy", "pct", True, "Training"),
    ("Eval loss", "final_eval_loss", "f4", False, "Evaluation"),
    ("KL loss", "final_kl_loss", "f4", False, "Evaluation"),
    ("CE loss", "final_ce_loss", "f4", False, "Evaluation"),
    ("TF acc", "final_tf_accuracy", "pct", True, "Evaluation"),
    ("Student acc", "final_student_accuracy", "pct", True, "Evaluation"),
    ("Class acc", "final_classification_accuracy", "pct", True, "Evaluation"),
    ("Tone F1", "f1_tone", "f4", True, "F1 Scores"),
    ("Sent F1", "f1_sentiment", "f4", True, "F1 Scores"),
    ("Safety F1", "f1_safety", "f4", True, "F1 Scores"),
    ("Toxic F1", "f1_toxicity", "f4", True, "F1 Scores"),
    ("Avg F1", "f1_macro_avg", "f4", True, "F1 Scores"),
    ("Stu slope", "student_slope", "slope", True, "Curves"),
    ("Cls slope", "classification_slope", "slope", True, "Curves"),
    ("Early/late", "early_vs_late_ratio", "f2", None, "Curves"),
    ("Conv (0-1)", "convergence_speed_90", "f2", False, "Curves"),
    ("Stdev", "last_epoch_stability", "f4", False, "Curves"),
    ("E1 stu \u0394", "epoch1_student_delta", "pct_delta", True, "Epoch deltas"),
    ("E1 cls \u0394", "epoch1_class_delta", "pct_delta", True, "Epoch deltas"),
    ("E2 stu \u0394", "epoch2_student_delta", "pct_delta", True, "Epoch deltas"),
    ("E2 cls \u0394", "epoch2_class_delta", "pct_delta", True, "Epoch deltas"),
    ("Acc/M par", "accuracy_per_million_params", "f2", True, "Efficiency"),
    ("Acc/hour", "accuracy_per_hour", "f2", True, "Efficiency"),
]


WINNER_METRICS = [
    ("Student accuracy", "final_student_accuracy", True, "pct"),
    ("Classification accuracy", "final_classification_accuracy", True, "pct"),
    ("Teacher-forced accuracy", "final_tf_accuracy", True, "pct"),
    ("Macro F1", "f1_macro_avg", True, "f4"),
    ("Eval loss (lowest)", "final_eval_loss", False, "f4"),
    ("Convergence (fastest)", "convergence_speed_90", False, "f2"),
    ("Stability (lowest stdev)", "last_epoch_stability", False, "f4"),
    ("Efficiency (acc/M params)", "accuracy_per_million_params", True, "f2"),
]


COLUMN_DESCRIPTIONS = {
    "status": "Run status: completed or in_progress.",
    "params_millions": "Total trainable parameters in millions.",
    "hidden_dim": "Transformer hidden / embedding dimension.",
    "num_layers": "Number of stacked transformer blocks.",
    "num_heads": "Number of attention heads per layer.",
    "kl_ratio": "Loss blend: 1.0 = pure KL divergence (match teacher distribution), 0.0 = pure cross-entropy (match hard label). Shown as start->end if annealed.",
    "temperature": "Distillation temperature for softening teacher logits. Higher = softer distribution. Shown as start->end if annealed.",
    "learning_rate": "Peak learning rate (cosine schedule with warmup).",
    "dropout": "Dropout rate applied across all transformer dropout layers during training.",
    "total_batches": "Total number of training batches processed across all epochs.",
    "training_hours": "Wall-clock training duration.",
    "final_train_loss": "Average training loss over the last 200 batches.",
    "final_train_accuracy": "Average training token accuracy over the last 200 batches.",
    "final_eval_loss": "Total evaluation loss (KL + CE blend) on test set after the final epoch.",
    "final_kl_loss": "KL divergence component of the final eval loss.",
    "final_ce_loss": "Cross-entropy component of the final eval loss.",
    "final_tf_accuracy": "Teacher-forced accuracy: ground truth tokens used as context. Optimistic upper bound.",
    "final_student_accuracy": "Student accuracy: model uses its own predictions as context (autoregressive). The realistic measure.",
    "final_classification_accuracy": "End-task accuracy: parses generated JSON output and compares all category labels against ground truth.",
    "f1_tone": "Macro-averaged F1 across the 5 tone classes.",
    "f1_sentiment": "Macro-averaged F1 across the 3 sentiment classes.",
    "f1_safety": "Macro-averaged F1 across the 2 safety classes.",
    "f1_toxicity": "Macro-averaged F1 across the 2 toxicity classes.",
    "f1_macro_avg": "Average of all four per-category macro F1 scores.",
    "student_slope": "Linear regression slope of student accuracy across mini-eval checkpoints, in percentage-points per mini-eval (~1000 batches). Indicates learning steepness.",
    "classification_slope": "Same slope but on classification accuracy. Often more sensitive than student accuracy.",
    "early_vs_late_ratio": "Ratio of accuracy gained in the first half of training vs the second half. >1 = front-loaded learning, <1 = back-loaded.",
    "convergence_speed_90": "Fraction of training (0-1) needed to reach 90% of total accuracy gain. Lower = faster convergence.",
    "last_epoch_stability": "Standard deviation of student accuracy across mini-evals in the final epoch. Lower = more stable convergence.",
    "epoch1_student_delta": "Change in full-eval student accuracy from epoch 0 to epoch 1.",
    "epoch1_class_delta": "Change in full-eval classification accuracy from epoch 0 to epoch 1.",
    "epoch2_student_delta": "Change in full-eval student accuracy from epoch 1 to epoch 2.",
    "epoch2_class_delta": "Change in full-eval classification accuracy from epoch 1 to epoch 2.",
    "accuracy_per_million_params": "Student accuracy (%) per million parameters. Parameter efficiency.",
    "accuracy_per_hour": "Student accuracy (%) per training hour. Compute efficiency.",
}


# ── JSON payload ────────────────────────────────────────────────────────────


def build_json_payload(reports: List[dict]) -> dict:
    columns = [
        {
            "label": label,
            "key": key,
            "format": fmt,
            "higher_is_better": higher_is_better,
            "group": group,
        }
        for label, key, fmt, higher_is_better, group in COLUMN_DEFINITIONS
    ]
    winner_metrics = [
        {
            "label": label,
            "key": key,
            "higher_is_better": higher_is_better,
            "format": fmt,
        }
        for label, key, higher_is_better, fmt in WINNER_METRICS
    ]
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "run_count": len(reports),
        "columns": columns,
        "winner_metrics": winner_metrics,
        "column_descriptions": COLUMN_DESCRIPTIONS,
        "runs": reports,
    }


# ── Entry point ─────────────────────────────────────────────────────────────


def evaluate_runs(
    names: Optional[List[str]] = None,
    prefix: Optional[str] = None,
    completed_only: bool = False,
    output_path: Optional[str] = None,
):
    all_runs = load_all_runs()
    print(f"Loaded {len(all_runs)} runs.")

    status_filter = "completed" if completed_only else None
    runs = filter_runs(all_runs, prefix=prefix, status=status_filter, names=names)

    if not runs:
        print("No runs match the filter criteria.")
        return

    runs = sort_runs(runs, sort_by="name")
    print(f"Evaluating {len(runs)} runs...")

    reports = [build_run_report(run) for run in runs]
    payload = build_json_payload(reports)

    destination = Path(output_path) if output_path else JSON_OUTPUT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    viewer_html = VIEWER_DIR / "run_evaluation.html"
    print(f"Wrote JSON: {destination}")
    print(f"Open viewer: {viewer_html}")
    print("(Serve the folder over HTTP so the viewer can fetch the JSON, e.g.")
    print(f"   python -m http.server --directory {VIEWER_DIR} 8000 )")


def main():
    args = sys.argv[1:]

    if "--help" in args or "-h" in args:
        print("Usage: python -m experimental_analysis.evaluate_runs [options]")
        print()
        print("Options:")
        print("  --all              Evaluate all runs")
        print("  --completed        Only completed runs")
        print("  --prefix PREFIX    Filter by run name prefix")
        print("  --output FILE      JSON output path (default: templates/run_evaluation.json)")
        print("  run_name ...       Specific run names to evaluate")
        print("  -h, --help         Show this help")
        return

    prefix = None
    output_path = None
    names = []
    completed_only = False

    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--all":
            pass  # no filter
        elif arg == "--completed":
            completed_only = True
        elif arg == "--prefix" and index + 1 < len(args):
            index += 1
            prefix = args[index]
        elif arg == "--output" and index + 1 < len(args):
            index += 1
            output_path = args[index]
        elif not arg.startswith("--"):
            names.append(arg)
        index += 1

    evaluate_runs(
        names=names if names else None,
        prefix=prefix,
        completed_only=completed_only,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
