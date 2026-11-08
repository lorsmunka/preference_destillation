"""Thesis figure 9.1 (chapter 9): the math tutor's 16.5M student on the full data, per pass: task accuracy
(agreement with Gemma's answer) and the test CE loss, pure KL vs pure CE (3 seeds each) and KL -> CE (3 seeds).
Thick line = mean of the seeds, thin lines = each seed.
Data: experiments/results/math_agent.csv (KL, CE) and runs/math-agent-kl-to-ce-m-s{1,2,3}/logs/training.jsonl.
Run: python thesis_work/grafikonok/plot_14_math_agent_kl_ce.py (redraws the PNG next to this file)."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

ROOT = Path(r"D:\preference_destillation")
CSV_PATH = ROOT / "experiments" / "results" / "math_agent.csv"
KL_TO_CE_RUNS = [ROOT / "runs" / f"math-agent-kl-to-ce-m-s{seed}" / "logs" / "training.jsonl" for seed in (1, 2, 3)]
OUTPUT = ROOT / "thesis_work" / "grafikonok" / "14_math_agent_kl_ce.png"

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
ARMS = [("kl", "KL", "#2a78d6", "-"), ("ce", "CE", "#eb6834", "-"), ("kl_to_ce", "KL → CE", "#1baf7a", (0, (4, 2)))]

curves = {"kl": {}, "ce": {}, "kl_to_ce": {}}   # arm -> seed -> [(pass, task accuracy, test loss)]
with open(CSV_PATH, encoding="utf-8") as file:
    for row in csv.DictReader(file):
        if row["arm"] not in curves:   # the CSV now also has "kl-to-ce" rows; KL -> CE is read from its runs below
            continue
        curves[row["arm"]].setdefault(row["seed"], []).append(
            (int(row["pass"]), float(row["task_accuracy"]), float(row["test_error"])))
for seed, path in enumerate(KL_TO_CE_RUNS, start=1):
    evaluations = [json.loads(line) for line in open(path, encoding="utf-8") if '"eval_epoch"' in line]
    curves["kl_to_ce"][str(seed)] = [(record["epoch"] + 1, record["task_accuracy"], record["ce_loss"])
                                     for record in evaluations]

plt.rcParams.update({"font.family": "Arial", "font.size": 10})
figure, (accuracy_axis, loss_axis) = plt.subplots(1, 2, figsize=(6.6, 3.3), dpi=200)
figure.patch.set_facecolor(SURFACE)

for arm, label, color, style in ARMS:
    seeds = [sorted(points) for points in curves[arm].values()]
    passes = [point[0] for point in seeds[0]]
    for column, axis, scale in ((1, accuracy_axis, 100), (2, loss_axis, 1)):
        for points in seeds:
            axis.plot(passes, [point[column] * scale for point in points], color=color, linewidth=0.8,
                      linestyle=style, alpha=0.35, zorder=1)
        mean = [sum(points[index][column] for points in seeds) / len(seeds) * scale for index in range(len(passes))]
        axis.plot(passes, mean, color=color, linewidth=2, linestyle=style, zorder=2,
                  label=f"{label} ({len(seeds)} véletlenmag)" if axis is accuracy_axis else None)

comma = matplotlib.ticker.FuncFormatter(lambda value, _position: f"{value:g}".replace(".", ","))
for axis, title in ((accuracy_axis, "Feladatpontosság (%)"), (loss_axis, "CE-veszteség a tesztadatokon")):
    axis.set_facecolor(SURFACE)
    axis.set_title(title, color=TEXT_PRIMARY, fontsize=10, loc="left")
    axis.set_xlabel("Epoch", color=TEXT_SECONDARY)
    axis.set_xticks(range(1, 11))
    axis.yaxis.set_major_formatter(comma)
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)
    axis.spines["bottom"].set_color(GRID)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.tick_params(colors=TEXT_SECONDARY, length=0)
accuracy_axis.set_ylim(0, 80)
loss_axis.set_ylim(0.4, 1.2)
loss_axis.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _position: f"{value:.1f}".replace(".", ",")))

figure.legend(loc="upper center", ncol=3, frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY,
              bbox_to_anchor=(0.5, 1.0))
figure.tight_layout(rect=(0, 0, 1, 0.92))
figure.savefig(OUTPUT, facecolor=SURFACE)
print(f"saved {OUTPUT}")
