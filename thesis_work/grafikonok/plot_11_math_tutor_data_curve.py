"""Thesis figure 2.1 (section 2.5.4): test CE loss vs training replies (math tutor slice curve, one seed),
with a straight line fitted against log2(replies). Data: experiments/results/slice_curve.csv.
Run: python thesis_work/grafikonok/plot_11_math_tutor_data_curve.py (redraws the PNG next to this file)."""
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

CSV_PATH = Path(r"D:\preference_destillation\experiments\results\slice_curve.csv")
OUTPUT = Path(r"D:\preference_destillation\thesis_work\grafikonok\11_math_tutor_data_curve.png")

SURFACE = "#fcfcfb"
SERIES = "#2a78d6"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"

with open(CSV_PATH, encoding="utf-8") as file:
    rows = list(csv.DictReader(file))
replies = [int(row["train_records"]) for row in rows]
loss = [float(row["test_error_final"]) for row in rows]

doublings = [math.log2(count) for count in replies]
mean_x = sum(doublings) / len(doublings)
mean_y = sum(loss) / len(loss)
slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(doublings, loss)) / sum((x - mean_x) ** 2 for x in doublings)
intercept = mean_y - slope * mean_x
fit_x = [400, 8000]
fit_y = [intercept + slope * math.log2(x) for x in fit_x]

plt.rcParams.update({"font.family": "Arial", "font.size": 11})
figure, axis = plt.subplots(figsize=(6.4, 3.8), dpi=200)
figure.patch.set_facecolor(SURFACE)
axis.set_facecolor(SURFACE)

axis.plot(fit_x, fit_y, linestyle=(0, (4, 3)), linewidth=1.5, color=TEXT_SECONDARY, zorder=1)
axis.plot(replies, loss, linewidth=2, color=SERIES, zorder=2)
axis.scatter(replies, loss, s=64, color=SERIES, edgecolor=SURFACE, linewidth=2, zorder=3)

axis.set_xscale("log", base=2)
axis.set_xticks(replies)
axis.set_xticklabels([f"{count:,}".replace(",", " ") if count >= 10000 else str(count) for count in replies])   # thesis style: a space only from 5 digits
axis.minorticks_off()
axis.set_xlim(400, 8000)
axis.set_ylim(1.6, 3.2)
axis.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _position: f"{value:.1f}".replace(".", ",")))
axis.set_xlabel("Tanítópéldák száma (logaritmikus skála)", color=TEXT_SECONDARY)
axis.set_ylabel("CE-veszteség a tesztadatokon", color=TEXT_SECONDARY)

for spine in ("top", "right", "left"):
    axis.spines[spine].set_visible(False)
axis.spines["bottom"].set_color(GRID)
axis.grid(axis="y", color=GRID, linewidth=0.8)
axis.tick_params(colors=TEXT_SECONDARY, length=0)

for count, value in [(replies[0], loss[0]), (replies[-1], loss[-1])]:
    axis.annotate(f"{value:.2f}".replace(".", ","), (count, value), textcoords="offset points",
                  xytext=(8, 6), color=TEXT_PRIMARY, fontsize=10)
axis.annotate(f"illesztett egyenes, {slope:.2f} duplázásonként".replace(".", ",").replace("-", "−"),
              (5600, intercept + slope * math.log2(5600)), textcoords="offset points", xytext=(-150, -26),
              color=TEXT_SECONDARY, fontsize=10)

figure.tight_layout()
figure.savefig(OUTPUT, facecolor=SURFACE)
print(f"slope {slope:.3f} per doubling; saved {OUTPUT}")
