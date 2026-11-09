"""Thesis figures 9.6 and 9.7 (chapter 9, section 9.5): where the best attention head looks while the model writes the
part of the reply that has to be looked up in the input. Three bars each, on the same 0-100% scale:
the untrained student (random weights, the even spread), the trained student and Gemma 4 12B (bf16).

- 18a (coding agent, 12.0M, KL, after pass 10): the share of attention on the target lines while writing a line
  number, the best of the student's 24 heads / Gemma's 768 heads.
  Data: agents/work/chapter9_synthesis_2026-11-06/code_attention_passes.csv (rows code-untrained-m, code-kl-m-p10,
  gemma; column "target lines when writing a line number (best head)").
- 18b (math tutor, 16.5M, KL, mean of seeds 1-3): the share of attention on the numbers while writing a number,
  the best head. Data: runs/attention_study/report_cache.pkl (math metrics "numbers when writing a number", the
  maximum over layers and heads; keys math-untrained-m, math-kl-m-s1..3, gemma).
Probes: 24 scope replies of Gemma on the 4 test projects (23 with line numbers) and 80 replies of Gemma on test
conversations, read by every model (experiments/attention_study.py).
Run: python thesis_work/grafikonok/plot_18_attention_lookup.py (redraws both PNGs next to this file)."""
import csv
import os
import pickle
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(r"D:\preference_destillation")
sys.path.insert(0, str(ROOT / "experiments"))   # the cache was pickled with the study's modules

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE_CSV = ROOT / "agents" / "work" / "chapter9_synthesis_2026-11-06" / "code_attention_passes.csv"
CACHE = ROOT / "runs" / "attention_study" / "report_cache.pkl"
OUTPUT_DIR = ROOT / "thesis_work" / "grafikonok"

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
UNTRAINED_COLOR = "#b9b7b1"
STUDENT_COLOR = "#2a78d6"
GEMMA_COLOR = "#eb6834"

code_rows = {row["model"]: row for row in csv.DictReader(open(CODE_CSV, encoding="utf-8"))}
code_column = "target lines when writing a line number (best head)"
code_values = [float(code_rows[key][code_column]) for key in ("code-untrained-m", "code-kl-m-p10", "gemma")]

math_metrics = pickle.load(open(CACHE, "rb"))["math"]["metrics"]
math_name = "numbers when writing a number"
trained = [float(math_metrics[f"math-kl-m-s{seed}"][math_name].max()) for seed in (1, 2, 3)]
math_values = [float(math_metrics["math-untrained-m"][math_name].max()), sum(trained) / len(trained),
               float(math_metrics["gemma"][math_name].max())]

FIGURES = [
    ("18a_code_attention.png", "A célsorokra jutó figyelem sorszám írásakor", code_values,
     ["Tanítatlan tanulómodell", "Tanulómodell, 10 epoch", "Gemma 4 12B"]),
    ("18b_math_attention.png", "A számokra jutó figyelem szám írásakor", math_values,
     ["Tanítatlan tanulómodell", "Tanulómodell, 10 epoch", "Gemma 4 12B"]),
]

plt.rcParams.update({"font.family": "Arial", "font.size": 9})
for file_name, title, values, labels in FIGURES:
    figure, axis = plt.subplots(figsize=(5.4, 1.55), dpi=200)
    figure.patch.set_facecolor(SURFACE)
    axis.set_facecolor(SURFACE)
    positions = [2, 1, 0]   # top to bottom: untrained, student, Gemma
    colors = [UNTRAINED_COLOR, STUDENT_COLOR, GEMMA_COLOR]
    axis.barh(positions, values, height=0.62, color=colors, zorder=2)
    for position, value in zip(positions, values):
        axis.annotate(f"{value * 100:.1f}%".replace(".", ","), (value, position), textcoords="offset points",
                      xytext=(4, 0), va="center", ha="left", fontsize=8.5, color=TEXT_PRIMARY)
    axis.set_yticks(positions)
    axis.set_yticklabels(labels)
    axis.set_xlim(0, 1)
    axis.set_xticks([0, 0.25, 0.5, 0.75, 1])
    axis.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    axis.set_title(title + " (a legjobb fejnél)", color=TEXT_PRIMARY, fontsize=9, loc="left")
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)
    axis.spines["bottom"].set_color(GRID)
    axis.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    axis.tick_params(colors=TEXT_SECONDARY, length=0, labelsize=8.5)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / file_name, facecolor=SURFACE)
    plt.close(figure)
    print(f"saved {OUTPUT_DIR / file_name}:", [round(value, 4) for value in values])
