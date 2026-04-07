import csv
import sys
from pathlib import Path
from typing import List, Optional

from experimental_analysis.run_data import (
    RunData,
    load_all_runs,
    filter_runs,
    sort_runs,
)


def format_number(value, decimals=4):
    if value is None:
        return "-"
    if isinstance(value, float):
        if abs(value) >= 1_000_000:
            return f"{value / 1_000_000:.1f}M"
        if abs(value) >= 1_000:
            return f"{value / 1_000:.1f}k"
        if abs(value) < 0.01:
            return f"{value:.2e}"
        return f"{value:.{decimals}f}"
    if isinstance(value, int):
        if value >= 1_000_000:
            return f"{value / 1_000_000:.1f}M"
        if value >= 1_000:
            return f"{value:,}"
        return str(value)
    return str(value)


def format_percentage(value):
    if value is None:
        return "-"
    return f"{value * 100:.2f}%"


def format_hours(value):
    if value is None:
        return "-"
    if value < 1:
        return f"{value * 60:.0f}m"
    return f"{value:.1f}h"


def build_comparison_table(runs: List[RunData]) -> dict:
    rows = []

    rows.append(("STATUS", [run.status for run in runs]))
    rows.append(("ARCHITECTURE", None))
    rows.append(("  Parameters", [format_number(run.total_parameters) for run in runs]))
    rows.append(("  Hidden dim", [str(run.hidden_dim) for run in runs]))
    rows.append(("  Layers", [str(run.num_layers) for run in runs]))
    rows.append(("  Heads", [str(run.num_heads) for run in runs]))
    rows.append(("  Head dim", [str(run.hidden_dim // run.num_heads) if run.num_heads > 0 else "-" for run in runs]))
    rows.append(("  Embedding params", [format_number(run.embedding_params) for run in runs]))
    rows.append(("  Body params", [format_number(run.body_params) for run in runs]))
    rows.append(("  Embed/Body ratio", [
        f"{run.embedding_params / run.body_params:.1f}x" if run.body_params > 0 else "-"
        for run in runs
    ]))
    rows.append(("  Input vocab", [format_number(run.input_vocab_size) for run in runs]))
    rows.append(("  Reduced input", ["Yes" if run.has_reduced_input_vocab else "No" for run in runs]))

    rows.append(("HYPERPARAMETERS", None))
    rows.append(("  KL ratio", [f"{run.kl_ratio_start}->{run.kl_ratio_end}" for run in runs]))
    rows.append(("  Temperature", [f"{run.temperature_start}->{run.temperature_end}" for run in runs]))
    rows.append(("  Learning rate", [f"{run.learning_rate:.0e}" for run in runs]))
    rows.append(("  Dropout", [str(run.dropout) for run in runs]))
    rows.append(("  Epochs", [str(run.epoch_count) for run in runs]))
    rows.append(("  Batch size", [str(run.batch_size) for run in runs]))

    rows.append(("TRAINING", None))
    rows.append(("  Wall time", [format_hours(run.training_hours) for run in runs]))
    rows.append(("  Total batches", [format_number(run.total_train_batches) for run in runs]))
    rows.append(("  Final train loss", [format_number(run.final_train_loss) for run in runs]))
    rows.append(("  Final train acc", [format_percentage(run.final_train_accuracy) for run in runs]))

    rows.append(("EVALUATION (final epoch)", None))
    for run in runs:
        if run.final_eval:
            break
    else:
        rows.append(("  (no eval data)", ["-" for _ in runs]))
        return {"headers": [run.short_name() for run in runs], "rows": rows}

    rows.append(("  Eval loss", [
        format_number(run.final_eval.avg_loss) if run.final_eval else "-" for run in runs
    ]))
    rows.append(("  Teacher-forced acc", [
        format_percentage(run.final_eval.teacher_forced_accuracy) if run.final_eval else "-" for run in runs
    ]))
    rows.append(("  Student acc", [
        format_percentage(run.final_eval.student_accuracy) if run.final_eval else "-" for run in runs
    ]))
    rows.append(("  Classification acc", [
        format_percentage(run.final_eval.classification_accuracy) if run.final_eval else "-" for run in runs
    ]))

    # Per-epoch breakdown
    max_epochs = max((len(run.eval_epochs) for run in runs), default=0)
    if max_epochs > 0:
        rows.append(("PER-EPOCH EVAL", None))
        for epoch_index in range(max_epochs):
            rows.append((f"  Epoch {epoch_index}", None))
            rows.append((f"    TF acc", [
                format_percentage(run.eval_epochs[epoch_index].teacher_forced_accuracy)
                if epoch_index < len(run.eval_epochs) else "-"
                for run in runs
            ]))
            rows.append((f"    Student acc", [
                format_percentage(run.eval_epochs[epoch_index].student_accuracy)
                if epoch_index < len(run.eval_epochs) else "-"
                for run in runs
            ]))
            rows.append((f"    Class acc", [
                format_percentage(run.eval_epochs[epoch_index].classification_accuracy)
                if epoch_index < len(run.eval_epochs) else "-"
                for run in runs
            ]))
            rows.append((f"    Loss", [
                format_number(run.eval_epochs[epoch_index].avg_loss)
                if epoch_index < len(run.eval_epochs) else "-"
                for run in runs
            ]))

    # Mini-eval summary
    mini_eval_runs = [run for run in runs if run.mini_evals]
    if mini_eval_runs:
        rows.append(("MINI-EVAL (latest)", None))
        rows.append(("  Count", [str(len(run.mini_evals)) if run.mini_evals else "-" for run in runs]))
        rows.append(("  TF acc", [
            format_percentage(run.mini_evals[-1].teacher_forced_accuracy)
            if run.mini_evals else "-" for run in runs
        ]))
        rows.append(("  Student acc", [
            format_percentage(run.mini_evals[-1].student_accuracy)
            if run.mini_evals else "-" for run in runs
        ]))
        rows.append(("  Class acc", [
            format_percentage(run.mini_evals[-1].classification_accuracy)
            if run.mini_evals else "-" for run in runs
        ]))

    return {"headers": [run.short_name() for run in runs], "rows": rows}


def find_best_values(runs: List[RunData]) -> dict:
    best = {}
    eval_runs = [run for run in runs if run.final_eval]
    if eval_runs:
        best["student_accuracy"] = max(run.final_eval.student_accuracy for run in eval_runs)
        best["teacher_forced_accuracy"] = max(run.final_eval.teacher_forced_accuracy for run in eval_runs)
        best["classification_accuracy"] = max(run.final_eval.classification_accuracy for run in eval_runs)
        best["eval_loss"] = min(run.final_eval.avg_loss for run in eval_runs)
    return best


def print_table(table: dict, highlight_best: bool = True):
    headers = table["headers"]
    rows = table["rows"]

    label_width = max(len(row[0]) for row in rows) + 2
    col_width = max(12, max((len(header) for header in headers), default=12) + 2)

    header_line = " " * label_width
    for header in headers:
        header_line += header.rjust(col_width)
    separator = "-" * len(header_line)

    print(separator)
    print(header_line)
    print(separator)

    for label, values in rows:
        if values is None:
            print(f"\n  {label}")
            continue
        line = label.ljust(label_width)
        for value in values:
            line += str(value).rjust(col_width)
        print(line)

    print(separator)


def print_winner_summary(runs: List[RunData]):
    eval_runs = [run for run in runs if run.final_eval]
    if not eval_runs:
        return

    print("\n  WINNERS")
    metrics = [
        ("Student accuracy", lambda r: r.final_eval.student_accuracy, True),
        ("Classification accuracy", lambda r: r.final_eval.classification_accuracy, True),
        ("Teacher-forced accuracy", lambda r: r.final_eval.teacher_forced_accuracy, True),
        ("Eval loss", lambda r: r.final_eval.avg_loss, False),
    ]

    for metric_name, getter, higher_is_better in metrics:
        if higher_is_better:
            winner = max(eval_runs, key=getter)
        else:
            winner = min(eval_runs, key=getter)
        value = getter(winner)
        formatted = format_percentage(value) if higher_is_better else format_number(value)
        print(f"    {metric_name}: {winner.short_name()} ({formatted})")


def export_csv(runs: List[RunData], output_path: str):
    table = build_comparison_table(runs)
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Metric"] + table["headers"])
        for label, values in table["rows"]:
            if values is None:
                writer.writerow([label])
            else:
                writer.writerow([label] + values)
    print(f"Exported to {output_path}")


def run_comparison(
    names: Optional[List[str]] = None,
    prefix: Optional[str] = None,
    sort_by: str = "params",
    csv_path: Optional[str] = None,
    completed_only: bool = False,
):
    print("Loading runs...")
    all_runs = load_all_runs()
    print(f"Found {len(all_runs)} runs total.")

    status_filter = "completed" if completed_only else None
    runs = filter_runs(all_runs, prefix=prefix, status=status_filter, names=names)

    if not runs:
        print("No runs match the filter criteria.")
        return

    runs = sort_runs(runs, sort_by=sort_by)
    print(f"Comparing {len(runs)} runs (sorted by {sort_by}):\n")

    table = build_comparison_table(runs)
    print_table(table)
    print_winner_summary(runs)

    if csv_path:
        export_csv(runs, csv_path)


def interactive_run_selection(all_runs: List[RunData]) -> List[RunData]:
    print("\nAvailable runs:")
    for index, run in enumerate(all_runs):
        status_marker = "+" if run.status == "completed" else "~" if run.status == "in_progress" else "?"
        params = format_number(run.total_parameters)
        print(f"  [{index + 1}] {status_marker} {run.name} ({params} params)")

    print(f"\n  [a] All runs")
    print(f"  [c] Completed only")
    print()

    selection = input("Select runs (comma-separated numbers, or a/c): ").strip().lower()

    if selection == "a":
        return all_runs
    if selection == "c":
        return [run for run in all_runs if run.status == "completed"]

    try:
        indices = [int(part.strip()) - 1 for part in selection.split(",")]
        return [all_runs[index] for index in indices if 0 <= index < len(all_runs)]
    except (ValueError, IndexError):
        print("Invalid selection.")
        return []


def main():
    args = sys.argv[1:]

    if "--help" in args or "-h" in args:
        print("Usage: python -m experimental_analysis.compare_runs [options] [run_names...]")
        print()
        print("Options:")
        print("  --all              Compare all runs")
        print("  --completed        Only completed runs")
        print("  --prefix PREFIX    Filter by run name prefix")
        print("  --sort-by METRIC   Sort by: params, name, student_accuracy, loss, time")
        print("  --csv FILE         Export comparison to CSV")
        print("  -h, --help         Show this help")
        print()
        print("If no arguments given, launches interactive picker.")
        return

    prefix = None
    sort_by = "params"
    csv_path = None
    names = []
    show_all = False
    completed_only = False

    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--all":
            show_all = True
        elif arg == "--completed":
            completed_only = True
        elif arg == "--prefix" and index + 1 < len(args):
            index += 1
            prefix = args[index]
        elif arg == "--sort-by" and index + 1 < len(args):
            index += 1
            sort_by = args[index]
        elif arg == "--csv" and index + 1 < len(args):
            index += 1
            csv_path = args[index]
        elif not arg.startswith("--"):
            names.append(arg)
        index += 1

    if not show_all and not prefix and not names and not completed_only:
        all_runs = load_all_runs()
        if not all_runs:
            print("No runs found.")
            return
        selected = interactive_run_selection(all_runs)
        if not selected:
            return
        selected = sort_runs(selected, sort_by=sort_by)
        print(f"\nComparing {len(selected)} runs (sorted by {sort_by}):\n")
        table = build_comparison_table(selected)
        print_table(table)
        print_winner_summary(selected)
        if csv_path:
            export_csv(selected, csv_path)
        return

    run_comparison(
        names=names if names else None,
        prefix=prefix,
        sort_by=sort_by,
        csv_path=csv_path,
        completed_only=completed_only,
    )


if __name__ == "__main__":
    main()
