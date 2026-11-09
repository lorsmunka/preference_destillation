"""Thesis figure 9.3 (chapter 9, section 9.3): task accuracy against student size, two panels (small multiples, each
with its own y axis; one series per panel, so no legend):
- left: the math tutor's students of 4 sizes on 64 554 training examples (KL 0.9 -> 0.1, 10 passes, one seed);
  data: experiments/results/math_full.csv (the "kd" runs);
- right: sentiment students of 7 sizes on ~490 000 training examples (3 passes, one seed each, full input vocabulary,
  field-level task accuracy over the valid JSON replies); data: thesis_work/grafikonok/data/03_sentiment_scaling_curve.csv.
Every point is labelled on the left (the thesis has no table for it); the right labels its first and last point.
Run: python thesis_work/grafikonok/plot_16_sentiment_size.py (redraws the PNG next to this file)."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

ROOT = Path(r"D:\preference_destillation")
MATH_CSV = ROOT / "experiments" / "results" / "math_full.csv"
SENTIMENT_CSV = ROOT / "thesis_work" / "grafikonok" / "data" / "03_sentiment_scaling_curve.csv"
OUTPUT = ROOT / "thesis_work" / "grafikonok" / "16_sentiment_size.png"

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"

with open(MATH_CSV, encoding="utf-8") as file:
    math_points = sorted((int(row["parameters"]) / 1e6, float(row["task_accuracy_final"]) * 100)
                         for row in csv.DictReader(file) if row["arm"] == "kd")
with open(SENTIMENT_CSV, encoding="utf-8") as file:
    sentiment_points = sorted((float(row["parameters_m"]), float(row["task_accuracy"])) for row in csv.DictReader(file))

plt.rcParams.update({"font.family": "Arial", "font.size": 10})
figure, (math_axis, sentiment_axis) = plt.subplots(1, 2, figsize=(6.6, 3.1), dpi=200)
figure.patch.set_facecolor(SURFACE)
panels = ((math_axis, math_points, "Matematikai feladatmegoldó ügynök", (0, 80), "all"),
          (sentiment_axis, sentiment_points, "Szöveghangulat-elemzés", (80, 90), "ends"))
for axis, points, title, limits, labels in panels:
    sizes = [size for size, _ in points]
    accuracies = [accuracy for _, accuracy in points]
    axis.plot(sizes, accuracies, color=SERIES, linewidth=2, marker="o", markersize=6,
              markeredgecolor=SURFACE, markeredgewidth=2, zorder=2)
    chosen = range(len(points)) if labels == "all" else (0, len(points) - 1)
    for index in chosen:
        if labels == "ends" and index == 0:
            offset, align = (0, -15), "center"
        elif labels == "all" and index == 0:
            offset, align = (9, -3), "left"
        else:
            offset, align = (0, 9), "center"
        axis.annotate(f"{accuracies[index]:.1f}%".replace(".", ","), (sizes[index], accuracies[index]),
                      textcoords="offset points", xytext=offset, ha=align, fontsize=8.5, color=TEXT_SECONDARY)
    axis.set_xscale("log")
    axis.set_xticks(sizes if labels == "all" else sizes[::2])   # every other size on the right, or the ticks collide
    axis.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _position: f"{value:.1f}M".replace(".", ",")))
    axis.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axis.set_ylim(*limits)
    axis.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _position: f"{value:g}%"))
    axis.set_facecolor(SURFACE)
    axis.set_title(title, color=TEXT_PRIMARY, fontsize=10, loc="left")
    axis.set_xlabel("Paraméterszám (logaritmikus skála)", color=TEXT_SECONDARY, fontsize=9)
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)
    axis.spines["bottom"].set_color(GRID)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.tick_params(colors=TEXT_SECONDARY, length=0, labelsize=8)
figure.tight_layout()
figure.savefig(OUTPUT, facecolor=SURFACE)
print(f"saved {OUTPUT}")
