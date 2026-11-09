"""Thesis figure (chapter 9, section 9.6): the math tutor's reply time against model size on four machines. The four
students of the size sweep (2.7M / 6.1M / 16.4M / 53.7M, 2 / 4 / 6 / 8 layers) as one line per machine, Gemma 4 12B
as its own point on the two GPUs (bf16 on the A100, 4-bit NF4 on the RTX 4070: it fits there only so). Both axes
logarithmic; the time gridlines double at each step (25, 50, ..., 6400), so equal gaps mean equal ratios (user,
2026-11-09: with 20 / 50 / 100 / 200 ticks the gap 20-50 looked wider than 1000-2000). Median time per reply (input text to reply text, batch 1, greedy), 80 test replies x 3 timed rounds after
an untimed warm-up pass. Data: experiments/results/speed.csv (experiments/speed_benchmark.py, speed_merge.py).
Run: python thesis_work/grafikonok/plot_20_speed_curve.py (redraws the PNG next to this file)."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

ROOT = Path(r"D:\preference_destillation")
DATA = ROOT / "experiments" / "results" / "speed.csv"
OUTPUT = ROOT / "thesis_work" / "grafikonok" / "20_speed_curve.png"

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SIZE_RUNS = ["math-full-kd-xs-s1", "math-full-kd-s1", "math-full-kd-m-s1", "math-full-kd-l-s1"]
# (hardware as recorded, device, label, colour, line style)
MACHINES = [
    ("NVIDIA A100-SXM4-80GB", "cuda", "A100 80 GB (GPU)", "#eb6834", "-"),
    ("NVIDIA GeForce RTX 4070", "cuda", "RTX 4070 (GPU)", "#2a78d6", "-"),
    ("Intel64 Family 6 Model 183 Stepping 1, GenuineIntel", "cpu", "i7-13700K (CPU)", "#1baf7a", "--"),
    ("x86_64", "cpu", "4 vCPU szerver (CPU)", "#eda100", "--"),
]
GEMMA_LABELS = {"NVIDIA A100-SXM4-80GB": "Gemma 4 12B, bf16", "NVIDIA GeForce RTX 4070": "Gemma 4 12B, NF4"}

rows = list(csv.DictReader(open(DATA, encoding="utf-8")))
# one x for Gemma on both cards: its bf16 count (the 4-bit copy is the same model, stored packed)
GEMMA_PARAMETERS = int(next(row for row in rows if row["model"] == "gemma-4-12B-it" and row["precision"] == "bf16")["parameters"])


def find(hardware, device, model, kind="math"):
    return next(row for row in rows if row["hardware"] == hardware and row["device"] == device
                and row["model"] == model and row["kind"] == kind)


def milliseconds(value):
    return f"{value:.0f} ms" if value < 10000 else f"{value / 1000:.1f} s".replace(".", ",")


plt.rcParams.update({"font.family": "Arial", "font.size": 9})
figure, axis = plt.subplots(figsize=(6.6, 3.4), dpi=200)
figure.patch.set_facecolor(SURFACE)
axis.set_facecolor(SURFACE)
for hardware, device, label, color, style in MACHINES:
    points = [find(hardware, device, run) for run in SIZE_RUNS]
    sizes = [int(point["parameters"]) for point in points]
    times = [float(point["median_ms"]) for point in points]
    axis.plot(sizes, times, color=color, linestyle=style, linewidth=2, marker="o", markersize=5,
              markeredgecolor=SURFACE, markeredgewidth=1.2, label=label, zorder=3)
    if hardware in GEMMA_LABELS:
        gemma = find(hardware, device, "gemma-4-12B-it")
        above = hardware.startswith("NVIDIA GeForce")
        # the two cards' points sit a little apart around the 12B tick so both stay visible
        size, time_ms = GEMMA_PARAMETERS * (1.3 if above else 1 / 1.3), float(gemma["median_ms"])
        axis.plot([size], [time_ms], color=color, marker="D", markersize=6, markeredgecolor=SURFACE,
                  markeredgewidth=1.2, linestyle="none", zorder=3)
        axis.annotate(f"{GEMMA_LABELS[hardware]}\n{milliseconds(time_ms)}".replace(".", ","), (size, time_ms),
                      textcoords="offset points", xytext=(0, 11 if above else -13), ha="center",
                      va="bottom" if above else "top", fontsize=7.5, color=TEXT_SECONDARY)
axis.set_xscale("log")
axis.set_yscale("log")
axis.set_xlim(1.6e6, 4e10)
axis.set_ylim(18, 12000)
size_ticks = [2.7e6, 6.1e6, 16.4e6, 53.7e6, 12e9]
axis.set_xticks(size_ticks)
axis.set_xticklabels(["2,7M", "6,1M", "16,4M", "53,7M", "12B"])
axis.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
time_ticks = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400]
axis.set_yticks(time_ticks)
axis.set_yticklabels([str(tick) for tick in time_ticks])
axis.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
axis.set_title("Válaszidő (ms, medián, logaritmikus skála, rácsvonalanként kétszeres)", color=TEXT_PRIMARY,
               fontsize=9, loc="left")
axis.set_xlabel("Paraméterek száma (logaritmikus skála)", color=TEXT_SECONDARY, fontsize=9)
for spine in ("top", "right", "left"):
    axis.spines[spine].set_visible(False)
axis.spines["bottom"].set_color(GRID)
axis.grid(axis="y", color=GRID, linewidth=0.8)
axis.tick_params(colors=TEXT_SECONDARY, length=0, labelsize=8.5)
axis.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, ncol=2)
figure.tight_layout()
figure.savefig(OUTPUT, facecolor=SURFACE)
print(f"saved {OUTPUT}")
