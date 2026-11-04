"""Thesis figure 5.1 (section 5.1): the framework's parts and the one-way data flow, plus the agent
(harness + trained student) between the user and the model. Labels match table 5.1's packages.
Run: python thesis_work/grafikonok/plot_13_framework_components.py (redraws the PNG next to this file)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUTPUT = Path(__file__).with_name("13_framework_components.png")

SURFACE = "#fcfcfb"
BOX = "#ffffff"
SERIES = "#2a78d6"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
BORDER = "#8d8c87"
GROUP = "#b9b8b3"

BOX_WIDTH = 2.8
BOX_HEIGHT = 1.45
COLUMNS = [1.6, 5.0, 8.4]
ROW_DATA = 6.35
ROW_TRAINING = 4.05
ROW_AGENT = 1.15
AGENT_BOX_HEIGHT = 1.2

plt.rcParams.update({"font.family": "Arial", "font.size": 10})
figure, axis = plt.subplots(figsize=(6.4, 4.608), dpi=200)
figure.patch.set_facecolor(SURFACE)
axis.set_facecolor(SURFACE)
axis.set_xlim(0, 10)
axis.set_ylim(0, 7.2)
axis.set_aspect("equal")
axis.axis("off")


def box(center_x, center_y, title, package=None, description=None, height=BOX_HEIGHT, width=BOX_WIDTH):
    axis.add_patch(FancyBboxPatch((center_x - width / 2, center_y - height / 2), width, height,
                                  boxstyle="round,pad=0,rounding_size=0.12", facecolor=BOX, edgecolor=BORDER, linewidth=1.0))
    lines_below = (package is not None) + (description.count("\n") + 1 if description else 0)
    top = center_y + 0.11 * lines_below + 0.1
    axis.text(center_x, top, title, ha="center", va="center", fontsize=10, fontweight="bold", color=TEXT_PRIMARY)
    next_y = top - 0.3
    if package:
        axis.text(center_x, next_y, package, ha="center", va="center", fontsize=8, family="Consolas", color=SERIES)
        next_y -= 0.18
    if description:
        axis.text(center_x, next_y, description, ha="center", va="top", fontsize=8, color=TEXT_SECONDARY, linespacing=1.25)


def arrow(start, end, color=SERIES, both=False):
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle="<|-|>" if both else "-|>", mutation_scale=11,
                                   color=color, linewidth=1.5, shrinkA=0, shrinkB=0))


half = BOX_WIDTH / 2
box(COLUMNS[0], ROW_DATA, "Domain", "library/domain", "feladatok, utasítások,\nmetrikák, szótár")
box(COLUMNS[1], ROW_DATA, "Adatgenerálás", "library/data_gen", "a tanármodell generál\nvagy pontoz")
box(COLUMNS[2], ROW_DATA, "Kötegfájlok", "batches/<domain>/", "bf16 logitok,\nujjlenyomat, metaadatok")
box(COLUMNS[2], ROW_TRAINING, "Tanítás", "library/training", "KL + CE veszteség,\nepochonkénti kiértékelés")
box(COLUMNS[1], ROW_TRAINING, "Futtatási könyvtár", "runs/<név>/", "info.json, naplók,\nellenőrzőpontok")
box(COLUMNS[0], ROW_TRAINING, "Elemzés", "library/tooling", "csoportok, összesítés,\nstatisztikai próbák")

arrow((COLUMNS[0] + half, ROW_DATA), (COLUMNS[1] - half, ROW_DATA))
arrow((COLUMNS[1] + half, ROW_DATA), (COLUMNS[2] - half, ROW_DATA))
arrow((COLUMNS[2], ROW_DATA - BOX_HEIGHT / 2), (COLUMNS[2], ROW_TRAINING + BOX_HEIGHT / 2))
arrow((COLUMNS[2] - half, ROW_TRAINING), (COLUMNS[1] + half, ROW_TRAINING))
arrow((COLUMNS[1] - half, ROW_TRAINING), (COLUMNS[0] + half, ROW_TRAINING))

group_left, group_right, group_bottom, group_top = 3.4, 9.95, 0.3, 2.4
axis.add_patch(FancyBboxPatch((group_left, group_bottom), group_right - group_left, group_top - group_bottom,
                              boxstyle="round,pad=0,rounding_size=0.15", facecolor="none", edgecolor=GROUP,
                              linewidth=1.2, linestyle=(0, (4, 3))))
axis.text(group_left + 0.15, group_top - 0.2, "Ügynök (library/agent)", ha="left", va="center", fontsize=8.5,
          fontweight="bold", color=TEXT_SECONDARY)

agent_center = ROW_AGENT - 0.08
box(COLUMNS[0], agent_center, "Felhasználó", None, "kérdés vagy feladat", height=AGENT_BOX_HEIGHT)
box(COLUMNS[1], agent_center, "Modellhám", "<domain>/harness.py", "útvonal, ellenőrzés,\nújrakérdezés", height=AGENT_BOX_HEIGHT)
box(COLUMNS[2], agent_center, "Tanulómodell", "library/model", "a válasz generálása", height=AGENT_BOX_HEIGHT)
arrow((COLUMNS[0] + half, agent_center), (COLUMNS[1] - half, agent_center), color=TEXT_SECONDARY, both=True)
arrow((COLUMNS[1] + half, agent_center), (COLUMNS[2] - half, agent_center), color=TEXT_SECONDARY, both=True)

arrow((COLUMNS[2], ROW_TRAINING - BOX_HEIGHT / 2), (COLUMNS[2], agent_center + AGENT_BOX_HEIGHT / 2))
axis.text(COLUMNS[2] - 0.15, (ROW_TRAINING - BOX_HEIGHT / 2 + group_top) / 2 + 0.02, "betanított tanulómodell",
          ha="right", va="center", fontsize=8, color=TEXT_SECONDARY)

figure.savefig(OUTPUT, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.05)
print("saved", OUTPUT)
