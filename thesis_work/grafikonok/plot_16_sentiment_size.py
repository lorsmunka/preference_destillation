"""Thesis figure 9.3 (chapter 9, section 9.3): sentiment students of 7 sizes on ~490 000 training examples (3 passes,
one seed each, full input vocabulary, field-level task accuracy over the valid JSON replies). One series, so no legend.
Data: thesis_work/grafikonok/data/03_sentiment_scaling_curve.csv.
Run: python thesis_work/grafikonok/plot_16_sentiment_size.py (redraws the PNG next to this file)."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

ROOT = Path(r"D:\preference_destillation")
CSV_PATH = ROOT / "thesis_work" / "grafikonok" / "data" / "03_sentiment_scaling_curve.csv"
OUTPUT = ROOT / "thesis_work" / "grafikonok" / "16_sentiment_size.png"

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"

with open(CSV_PATH, encoding="utf-8") as file:
    points = sorted((float(row["parameters_m"]), float(row["task_accuracy"])) for row in csv.DictReader(file))
sizes = [size for size, _ in points]
accuracies = [accuracy for _, accuracy in points]

plt.rcParams.update({"font.family": "Arial", "font.size": 10})
figure, axis = plt.subplots(figsize=(6.6, 3.0), dpi=200)
figure.patch.set_facecolor(SURFACE)
axis.plot(sizes, accuracies, color=SERIES, linewidth=2, marker="o", markersize=6,
          markeredgecolor=SURFACE, markeredgewidth=2, zorder=2)
for (size, accuracy), offset, align in (((sizes[0], accuracies[0]), (0, -15), "center"),
                                        ((sizes[-1], accuracies[-1]), (0, 9), "center")):
    axis.annotate(f"{accuracy:.1f}%".replace(".", ","), (size, accuracy), textcoords="offset points",
                  xytext=offset, ha=align, fontsize=8.5, color=TEXT_SECONDARY)
axis.set_xscale("log")
axis.set_xticks(sizes)
axis.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _position: f"{value:.1f}M".replace(".", ",")))
axis.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
axis.set_ylim(80, 90)
axis.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _position: f"{value:g}%"))
axis.set_facecolor(SURFACE)
axis.set_title("Feladatpontosság", color=TEXT_PRIMARY, fontsize=10, loc="left")
axis.set_xlabel("A tanulómodell paraméterszáma (logaritmikus skála)", color=TEXT_SECONDARY, fontsize=9)
for spine in ("top", "right", "left"):
    axis.spines[spine].set_visible(False)
axis.spines["bottom"].set_color(GRID)
axis.grid(axis="y", color=GRID, linewidth=0.8)
axis.tick_params(colors=TEXT_SECONDARY, length=0, labelsize=8.5)
figure.tight_layout()
figure.savefig(OUTPUT, facecolor=SURFACE)
print(f"saved {OUTPUT}")
