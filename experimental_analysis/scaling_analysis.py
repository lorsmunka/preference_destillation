import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from experimental_analysis.run_data import (
    RunData,
    load_all_runs,
    filter_runs,
    sort_runs,
)
from experimental_analysis.compare_plots import OUTPUT_DIR, save_figure, ensure_output_dir


def power_law(x, a, b, c):
    return a * np.power(x, b) + c


def log_curve(x, a, b):
    return a * np.log(x) + b


def fit_scaling_curve(
    params: List[float],
    accuracies: List[float],
) -> Optional[dict]:
    params_array = np.array(params)
    accuracy_array = np.array(accuracies)

    results = {}

    # Power law fit: accuracy = a * params^b + c
    try:
        from scipy.optimize import curve_fit
        popt, pcov = curve_fit(
            power_law, params_array, accuracy_array,
            p0=[10, 0.3, 50],
            maxfev=10000,
        )
        predicted = power_law(params_array, *popt)
        residuals = accuracy_array - predicted
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((accuracy_array - np.mean(accuracy_array)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        results["power_law"] = {
            "params": {"a": popt[0], "b": popt[1], "c": popt[2]},
            "r_squared": r_squared,
            "predict": lambda x: power_law(x, *popt),
        }
    except Exception as error:
        print(f"  Power law fit failed: {error}")

    # Log fit: accuracy = a * log(params) + b
    try:
        from scipy.optimize import curve_fit
        popt, pcov = curve_fit(
            log_curve, params_array, accuracy_array,
            p0=[10, 0],
            maxfev=10000,
        )
        predicted = log_curve(params_array, *popt)
        residuals = accuracy_array - predicted
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((accuracy_array - np.mean(accuracy_array)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        results["log"] = {
            "params": {"a": popt[0], "b": popt[1]},
            "r_squared": r_squared,
            "predict": lambda x: log_curve(x, *popt),
        }
    except Exception as error:
        print(f"  Log fit failed: {error}")

    return results if results else None


def find_knee(params: List[float], accuracies: List[float]) -> Optional[Tuple[float, float]]:
    if len(params) < 3:
        return None

    params_array = np.array(params)
    accuracy_array = np.array(accuracies)

    # Marginal gain: accuracy improvement per million params
    marginal_gains = []
    for index in range(1, len(params_array)):
        delta_accuracy = accuracy_array[index] - accuracy_array[index - 1]
        delta_params = params_array[index] - params_array[index - 1]
        if delta_params > 0:
            marginal_gains.append(delta_accuracy / delta_params)
        else:
            marginal_gains.append(0)

    # Knee: where marginal gain drops below 50% of average
    avg_gain = np.mean(marginal_gains)
    for index, gain in enumerate(marginal_gains):
        if gain < avg_gain * 0.5:
            knee_index = index + 1
            return params_array[knee_index], accuracy_array[knee_index]

    return None


def print_scaling_summary(
    runs: List[RunData],
    fits: Optional[dict],
    knee: Optional[Tuple[float, float]],
):
    print("\n" + "=" * 60)
    print("SCALING ANALYSIS SUMMARY")
    print("=" * 60)

    eval_runs = sorted(
        [run for run in runs if run.final_eval],
        key=lambda run: run.total_parameters,
    )

    print(f"\nRuns analyzed: {len(eval_runs)}")
    print(f"Parameter range: {eval_runs[0].params_millions():.1f}M - {eval_runs[-1].params_millions():.1f}M")

    print("\n  Marginal gains:")
    for index in range(1, len(eval_runs)):
        previous = eval_runs[index - 1]
        current = eval_runs[index]
        delta_accuracy = (current.final_eval.student_accuracy - previous.final_eval.student_accuracy) * 100
        delta_params = current.params_millions() - previous.params_millions()
        gain_per_million = delta_accuracy / delta_params if delta_params > 0 else 0
        print(
            f"    {previous.params_millions():.0f}M -> {current.params_millions():.0f}M: "
            f"+{delta_accuracy:.2f}pp student acc "
            f"(+{delta_params:.0f}M params, {gain_per_million:.2f}pp/M)"
        )

    if knee:
        print(f"\n  Knee detected at ~{knee[0]:.0f}M params ({knee[1]:.1f}% student accuracy)")
        print(f"  Beyond this point, adding parameters gives diminishing returns.")
    else:
        print("\n  No clear knee detected — accuracy may still be scaling.")

    if fits:
        print("\n  Curve fits:")
        for fit_name, fit_data in fits.items():
            r_squared = fit_data["r_squared"]
            quality = "excellent" if r_squared > 0.95 else "good" if r_squared > 0.9 else "moderate" if r_squared > 0.8 else "poor"
            print(f"    {fit_name}: R^2 = {r_squared:.4f} ({quality})")

        # Extrapolations
        best_fit_name = max(fits.keys(), key=lambda name: fits[name]["r_squared"])
        best_fit = fits[best_fit_name]
        print(f"\n  Extrapolations (using {best_fit_name} fit, R^2={best_fit['r_squared']:.3f}):")
        for target_params in [200, 500, 1000]:
            predicted = best_fit["predict"](target_params)
            predicted = min(predicted, 100)
            print(f"    {target_params}M params -> ~{predicted:.1f}% student accuracy (projected)")

    # Reduced vs full vocab comparison
    reduced = [run for run in eval_runs if run.has_reduced_input_vocab]
    full = [run for run in eval_runs if not run.has_reduced_input_vocab]
    if reduced and full:
        print("\n  Reduced input vocab comparison:")
        for reduced_run in reduced:
            closest_full = min(full, key=lambda f: abs(f.total_parameters - reduced_run.total_parameters))
            delta = (reduced_run.final_eval.student_accuracy - closest_full.final_eval.student_accuracy) * 100
            sign = "+" if delta >= 0 else ""
            print(
                f"    {reduced_run.short_name()} vs {closest_full.short_name()}: "
                f"{sign}{delta:.2f}pp student accuracy "
                f"({reduced_run.params_millions():.0f}M vs {closest_full.params_millions():.0f}M params)"
            )


def plot_scaling_curve(runs: List[RunData], fits: Optional[dict], knee: Optional[Tuple[float, float]], show: bool = False):
    full_vocab_runs = sorted(
        [run for run in runs if not run.has_reduced_input_vocab and run.final_eval],
        key=lambda run: run.total_parameters,
    )
    reduced_vocab_runs = sorted(
        [run for run in runs if run.has_reduced_input_vocab and run.final_eval],
        key=lambda run: run.total_parameters,
    )

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Left: main scaling curve
    ax = axes[0]
    metrics = [
        ("Teacher-Forced", lambda run: run.final_eval.teacher_forced_accuracy * 100, "#2ca02c", "o"),
        ("Student", lambda run: run.final_eval.student_accuracy * 100, "#1f77b4", "s"),
        ("Classification", lambda run: run.final_eval.classification_accuracy * 100, "#ff7f0e", "^"),
    ]

    for metric_name, getter, color, marker in metrics:
        if full_vocab_runs:
            params = [run.params_millions() for run in full_vocab_runs]
            values = [getter(run) for run in full_vocab_runs]
            ax.plot(params, values, marker=marker, color=color, linewidth=2,
                    markersize=8, label=f"{metric_name} (full vocab)", zorder=3)

        if reduced_vocab_runs:
            params = [run.params_millions() for run in reduced_vocab_runs]
            values = [getter(run) for run in reduced_vocab_runs]
            ax.plot(params, values, marker=marker, color=color, linewidth=2,
                    markersize=8, linestyle="--", alpha=0.7,
                    label=f"{metric_name} (reduced vocab)", zorder=3)

    # Fitted curve overlay
    if fits and full_vocab_runs:
        best_fit_name = max(fits.keys(), key=lambda name: fits[name]["r_squared"])
        best_fit = fits[best_fit_name]
        fit_x = np.linspace(
            min(run.params_millions() for run in full_vocab_runs) * 0.8,
            max(run.params_millions() for run in full_vocab_runs) * 1.5,
            100,
        )
        fit_y = np.clip(best_fit["predict"](fit_x), 0, 100)
        ax.plot(fit_x, fit_y, color="#1f77b4", linewidth=1, linestyle=":",
                alpha=0.5, label=f"Fit ({best_fit_name}, R\u00b2={best_fit['r_squared']:.3f})")

    if knee:
        ax.axvline(x=knee[0], color="red", linewidth=1, linestyle="--", alpha=0.5)
        ax.annotate(
            f"Knee: ~{knee[0]:.0f}M\n({knee[1]:.1f}%)",
            xy=(knee[0], knee[1]),
            xytext=(knee[0] * 1.5, knee[1] - 5),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="red", alpha=0.7),
            color="red",
        )

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}M"))
    ax.set_xlabel("Parameters (millions)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Scaling Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    # Right: marginal gain analysis
    ax = axes[1]
    all_eval_runs = sorted(
        [run for run in runs if run.final_eval],
        key=lambda run: run.total_parameters,
    )

    if len(all_eval_runs) >= 2:
        midpoints = []
        gains_student = []
        gains_classification = []

        for index in range(1, len(all_eval_runs)):
            previous = all_eval_runs[index - 1]
            current = all_eval_runs[index]
            delta_params = current.params_millions() - previous.params_millions()
            if delta_params > 0:
                midpoint = (previous.params_millions() + current.params_millions()) / 2
                midpoints.append(midpoint)
                student_gain = (current.final_eval.student_accuracy - previous.final_eval.student_accuracy) * 100 / delta_params
                classification_gain = (current.final_eval.classification_accuracy - previous.final_eval.classification_accuracy) * 100 / delta_params
                gains_student.append(student_gain)
                gains_classification.append(classification_gain)

        ax.bar(
            [midpoint - 1 for midpoint in midpoints], gains_student,
            width=2, color="#1f77b4", alpha=0.8, label="Student acc gain/M params",
        )
        ax.bar(
            [midpoint + 1 for midpoint in midpoints], gains_classification,
            width=2, color="#ff7f0e", alpha=0.8, label="Classification acc gain/M params",
        )

        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_xlabel("Parameter Count (M)", fontsize=12)
        ax.set_ylabel("Accuracy Gain per Million Params (pp/M)", fontsize=12)
        ax.set_title("Marginal Returns", fontsize=14, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, "scaling_curve.png", show=show)


def run_scaling_analysis(
    runs: Optional[List[RunData]] = None,
    show: bool = False,
):
    if runs is None:
        all_runs = load_all_runs()
        runs = [run for run in all_runs if run.final_eval]

    if not runs:
        print("No runs with eval data found.")
        return

    runs = sort_runs(runs, sort_by="params")

    # Use full-vocab runs for curve fitting (cleaner comparison)
    full_vocab_runs = [run for run in runs if not run.has_reduced_input_vocab and run.final_eval]
    full_vocab_runs = sorted(full_vocab_runs, key=lambda run: run.total_parameters)

    fits = None
    knee = None

    if len(full_vocab_runs) >= 3:
        params = [run.params_millions() for run in full_vocab_runs]
        student_accuracy = [run.final_eval.student_accuracy * 100 for run in full_vocab_runs]

        fits = fit_scaling_curve(params, student_accuracy)
        knee = find_knee(params, student_accuracy)

    print_scaling_summary(runs, fits, knee)
    plot_scaling_curve(runs, fits, knee, show=show)


def main():
    show = "--show" in sys.argv
    run_scaling_analysis(show=show)


if __name__ == "__main__":
    main()
