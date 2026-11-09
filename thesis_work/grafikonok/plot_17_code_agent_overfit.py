"""Thesis figure 9.5 (chapter 9, section 9.5): the coding agent's student (code-agent-kl-m-s1, 12.0M, pure KL, one
seed) per pass: the CE loss on its training data falls while the CE loss on the test projects is lowest at pass 2
and then rises. Two series on one axis (same quantity), direct labels on the end points, legend kept.
Data: runs/code-agent-kl-m-s1/logs/training.jsonl (train_epoch ce_loss, eval_epoch ce_loss).
Run: python thesis_work/grafikonok/plot_17_code_agent_overfit.py (redraws the PNG next to this file)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

ROOT = Path(r"D:\preference_destillation")
LOG = ROOT / "runs" / "code-agent-kl-m-s1" / "logs" / "training.jsonl"
OUTPUT = ROOT / "thesis_work" / "grafikonok" / "17_code_agent_overfit.png"

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = [("train_epoch", "Tanítóadat", "#2a78d6"), ("eval_epoch", "Tesztprojektek", "#eb6834")]

records = [json.loads(line) for line in open(LOG, encoding="utf-8")]
plt.rcParams.update({"font.family": "Arial", "font.size": 10})
figure, axis = plt.subplots(figsize=(6.6, 3.0), dpi=200)
figure.patch.set_facecolor(SURFACE)
for kind, label, color in SERIES:
    values = [record["ce_loss"] for record in records if record.get("type") == kind]
    passes = list(range(1, len(values) + 1))
    axis.plot(passes, values, color=color, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE,
              markeredgewidth=1.5, label=label, zorder=2)
    if kind == "train_epoch":   # the end only (the start sits on the test line and the y-axis labels)
        marks = [(len(values) - 1, (0, 8), "center")]
    else:                       # the lowest point and the end, both above the line
        lowest = min(range(len(values)), key=values.__getitem__)
        marks = [(lowest, (0, 8), "center"), (len(values) - 1, (0, 8), "center")]
    for index, offset, align in marks:
        axis.annotate(f"{values[index]:.2f}".replace(".", ","), (passes[index], values[index]), textcoords="offset points",
                      xytext=offset, ha=align, fontsize=8.5, color=TEXT_SECONDARY)
axis.set_xticks(range(1, 11))
axis.set_ylim(0, 3.6)
axis.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _position: f"{value:.1f}".replace(".", ",")))
axis.set_facecolor(SURFACE)
axis.set_title("CE-veszteség", color=TEXT_PRIMARY, fontsize=10, loc="left")
axis.set_xlabel("Epoch", color=TEXT_SECONDARY, fontsize=9)
for spine in ("top", "right", "left"):
    axis.spines[spine].set_visible(False)
axis.spines["bottom"].set_color(GRID)
axis.grid(axis="y", color=GRID, linewidth=0.8)
axis.tick_params(colors=TEXT_SECONDARY, length=0, labelsize=8.5)
axis.legend(loc="center right", frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY)
figure.tight_layout()
figure.savefig(OUTPUT, facecolor=SURFACE)
print(f"saved {OUTPUT}")
