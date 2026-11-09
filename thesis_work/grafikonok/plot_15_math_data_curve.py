"""Thesis figure 9.2 (chapter 9, section 9.3): the math tutor's data curve on one test set. The same 16.5M student
(KL, settings of math-agent-kl-m-s1, seed 1, 10 passes) trained on 6 000, 20 000, 65 000 and all 195 980 replies;
last pass: task accuracy (agreement with Gemma's answer) and the test CE loss. One series per panel, so no legend.
Data: experiments/results/math_data_curve.csv (pass 10).
Run: python thesis_work/grafikonok/plot_15_math_data_curve.py (redraws the PNG next to this file)."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

ROOT = Path(r"D:\preference_destillation")
CSV_PATH = ROOT / "experiments" / "results" / "math_data_curve.csv"
OUTPUT = ROOT / "thesis_work" / "grafikonok" / "15_math_data_curve.png"

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"
FULL_DATA_REPLIES = 195980   # the full run's training replies (the CSV counts 6 126 batches x 32 = 196 032)

points = {}
with open(CSV_PATH, encoding="utf-8") as file:
    for row in csv.DictReader(file):
        if row["pass"] == "10" and row["seed"] == "1":
            replies = int(row["training_replies"])
            replies = FULL_DATA_REPLIES if replies > 190000 else replies
            points[replies] = (float(row["task_accuracy"]) * 100, float(row["test_error"]))
replies = sorted(points)

plt.rcParams.update({"font.family": "Arial", "font.size": 10})
figure, (accuracy_axis, loss_axis) = plt.subplots(1, 2, figsize=(6.6, 3.0), dpi=200)
figure.patch.set_facecolor(SURFACE)
for axis, column, title in ((accuracy_axis, 0, "Feladatpontosság (%)"), (loss_axis, 1, "CE-veszteség a tesztadatokon")):
    values = [points[count][column] for count in replies]
    axis.plot(replies, values, color=SERIES, linewidth=2, marker="o", markersize=6,
              markeredgecolor=SURFACE, markeredgewidth=2, zorder=2)
    # direct labels on the first and last point only, placed clear of the axes' tick labels
    offsets = [((0, 9), "center"), ((0, 9), "center")] if column == 0 else [((10, 2), "left"), ((-8, -13), "right")]
    for (count, value), (offset, align) in zip(((replies[0], values[0]), (replies[-1], values[-1])), offsets):
        text = f"{value:.1f}%" if column == 0 else f"{value:.3f}"
        axis.annotate(text.replace(".", ","), (count, value), textcoords="offset points",
                      xytext=offset, ha=align, fontsize=8.5, color=TEXT_SECONDARY)
    axis.set_xscale("log")
    axis.set_xticks(replies)
    axis.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda value, _position: f"{round(value / 1000):d} ezer"))
    axis.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axis.set_facecolor(SURFACE)
    axis.set_title(title, color=TEXT_PRIMARY, fontsize=10, loc="left")
    axis.set_xlabel("Tanítópéldák száma (logaritmikus skála)", color=TEXT_SECONDARY, fontsize=9)
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)
    axis.spines["bottom"].set_color(GRID)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.tick_params(colors=TEXT_SECONDARY, length=0, labelsize=8.5)
accuracy_axis.set_ylim(0, 80)
accuracy_axis.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _position: f"{value:g}"))
loss_axis.set_ylim(0.2, 1.8)
loss_axis.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _position: f"{value:.1f}".replace(".", ",")))
figure.tight_layout()
figure.savefig(OUTPUT, facecolor=SURFACE)
print(f"saved {OUTPUT}")
