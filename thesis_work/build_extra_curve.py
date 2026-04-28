"""Generate one additional training curve figure for the tsc-scale-192h-6L-53M run.

Reuses the same plotting style as plot_loss_strategy_training_curves() in
build_grafikonok.py without touching any existing outputs.
"""

import sys
from pathlib import Path

THESIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THESIS_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_grafikonok import training_entries, OUTPUT_DIR


def moving_average(values, window):
    if len(values) < window:
        return values
    return [
        sum(values[index - window + 1:index + 1]) / window
        for index in range(window - 1, len(values))
    ]


def plot_timeseries(axis, indices, data, light_color, dark_color, ylabel, title,
                    ma_window=40, ylim_data=None, default_padding=0.1,
                    is_percent=False, show_last_200_avg=True, ylim_padding_ratio=0.1):
    axis.plot(indices, data, color=light_color, alpha=0.5, linewidth=0.7)

    if len(data) >= ma_window:
        averaged = moving_average(data, ma_window)
        axis.plot(indices[ma_window - 1:], averaged, color=dark_color, linewidth=2,
                  label=f"MA({ma_window})")

    if show_last_200_avg and len(data) >= 200:
        avg = sum(data[-200:]) / 200
        fmt = f"{avg:.1f}%" if is_percent else f"{avg:.4f}"
        axis.axhline(y=avg, color=dark_color, linewidth=1, alpha=0.25, linestyle="-",
                     label=f"Last 200 avg: {fmt}")

    if ylim_data:
        min_value = min(ylim_data)
        max_value = max(ylim_data)
        padding = (max_value - min_value) * ylim_padding_ratio if max_value > min_value else default_padding
        low = max(0, min_value - padding)
        high = min(100, max_value + padding) if is_percent else max_value + padding
        axis.set_ylim([low, high])

    axis.set_xlabel("Batch")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.legend()
    axis.grid(True, alpha=0.3)


def main():
    run_name = "tsc-scale-192h-6L-53M"
    filename = "07g_tsc_scale_53M_training_curves.png"

    entries = training_entries(run_name)
    batches = [entry for entry in entries if entry.get("type") == "train_batch"]
    if not batches:
        raise SystemExit(f"No train_batch entries for {run_name}")

    batch_indices = list(range(1, len(batches) + 1))
    losses = [batch["loss"] for batch in batches]
    kl_losses = [batch.get("kl_loss", 0) for batch in batches]
    ce_losses = [batch.get("ce_loss", 0) for batch in batches]
    accuracies = [batch["accuracy"] * 100 for batch in batches]
    ma_window = max(40, int(len(losses) * 0.03))
    last_90_start = int(len(losses) * 0.1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    metrics = [
        ("Loss", losses, "lightcoral", "red", axes[0, 0], False),
        ("KL Loss", kl_losses, "plum", "purple", axes[0, 1], False),
        ("CE Loss", ce_losses, "lightskyblue", "blue", axes[1, 0], False),
        ("Accuracy", accuracies, "lightgreen", "green", axes[1, 1], True),
    ]
    for name, data, light_color, dark_color, axis, is_percent in metrics:
        data_90 = data[last_90_start:] if last_90_start < len(data) else data
        plot_timeseries(
            axis, batch_indices, data, light_color, dark_color, name,
            f"{name} - All ({len(data)} batches, y-axis: last 90%)",
            ma_window=ma_window,
            ylim_data=data_90,
            default_padding=5 if is_percent else 0.1,
            is_percent=is_percent,
            show_last_200_avg=True,
        )
        if name == "Accuracy" and len(data) >= 400:
            current = sum(data[-200:]) / 200
            previous = sum(data[-400:-200]) / 200
            axis.text(
                0.02, 0.06,
                f"Last-200 change: {current - previous:+.2f} pp",
                transform=axis.transAxes,
                fontsize=10,
                bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8),
            )

    fig.tight_layout()
    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path} ({len(batches)} batches)")


if __name__ == "__main__":
    main()
