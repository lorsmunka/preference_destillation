"""Thesis figure (section 2.2): the 2.2 example replies token by token, each token shaded by the
teacher's entropy at that step (softmax over the stored reduced-vocabulary logits at T=1, nats).
Input: data/12_token_entropy_examples.json from extract_12_token_entropy.py (tokens + per-token entropy).
Run: python thesis_work/grafikonok/plot_12_token_entropy.py"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.patches import Rectangle
import matplotlib.ticker

OUTPUT = Path(__file__).resolve().parent / "12_token_entropy_examples.png"
PANELS = [
    ("reddit", "Szöveghangulat-elemzés (Gemma 3 4B)"),
    ("math_v1", "Matematikai szöveges feladatok, első verzió (Gemma 3 4B)"),
    ("post_generation", "Posztgenerálás (Gemma 3 4B)"),
    ("tutor_solution", "Matematikai feladatmegoldó ügynök, megoldás (Gemma 4 12B)"),
    ("tutor_explanation", "Matematikai feladatmegoldó ügynök, magyarázat (Gemma 4 12B)"),
    ("code_scope", "Kódügynök, 1. lépés: a kódrészlet kijelölése (Gemma 4 12B)"),
    ("code_explanation", "Kódügynök, 2. lépés: magyarázat (Gemma 4 12B)"),
]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
RAMP = ["#f3f7fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
MAX_ENTROPY = 2.0
MAX_CHARS = 96
FONT_SIZE = 8.5
CHAR_WIDTH = FONT_SIZE * 0.602 / 72      # DejaVu Sans Mono advance, inches
LINE_HEIGHT = FONT_SIZE * 1.55 / 72
TITLE_HEIGHT = 0.30
PANEL_GAP = 0.16
MARGIN = 0.25

colormap = LinearSegmentedColormap.from_list("entropy", RAMP)
norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=MAX_ENTROPY, clip=True)


def pieces(token):
    """A token as display pieces: text, or None for a line break (drawn as a small arrow)."""
    text = token.replace("▁", " ")
    parts = text.split("\n")
    result = []
    for index, part in enumerate(parts):
        if part:
            result.append(part)
        if index < len(parts) - 1:
            result.append(None)
    return result


def layout(tokens, entropies):
    """Lines of (column, text, entropy) cells; a newline piece shows as ↵ and ends the line."""
    lines, line, column = [], [], 0
    for token, entropy in zip(tokens, entropies):
        for piece in pieces(token):
            if piece is None:
                line.append((column, "↵", entropy))
                lines.append(line)
                line, column = [], 0
                continue
            if column + len(piece) > MAX_CHARS and line:
                lines.append(line)
                line, column = [], 0
            line.append((column, piece, entropy))
            column += len(piece)
    if line:
        lines.append(line)
    return lines


def text_color(entropy):
    red, green, blue, _ = colormap(norm(entropy))
    return "#ffffff" if 0.2126 * red + 0.7152 * green + 0.0722 * blue < 0.5 else TEXT_PRIMARY


DATA = Path(__file__).resolve().parent / "data" / "12_token_entropy_examples.json"
data = json.load(open(sys.argv[1] if len(sys.argv) > 1 else DATA, encoding="utf-8"))
laid_out = [(title, layout(data[key]["tokens"], data[key]["entropy"])) for key, title in PANELS]
width = 2 * MARGIN + MAX_CHARS * CHAR_WIDTH + 0.2
colorbar_height = 0.75
height = MARGIN + sum(TITLE_HEIGHT + len(lines) * LINE_HEIGHT + PANEL_GAP for _, lines in laid_out) + colorbar_height

plt.rcParams.update({"font.family": "DejaVu Sans"})
figure = plt.figure(figsize=(width, height), dpi=220)
figure.patch.set_facecolor(SURFACE)
axis = figure.add_axes([0, 0, 1, 1])
axis.set_xlim(0, width)
axis.set_ylim(height, 0)
axis.axis("off")

y = MARGIN
for title, lines in laid_out:
    axis.text(MARGIN, y + TITLE_HEIGHT * 0.55, title, fontsize=9.5, fontweight="bold", color=TEXT_PRIMARY, va="center")
    y += TITLE_HEIGHT
    for line in lines:
        for column, text, entropy in line:
            x = MARGIN + column * CHAR_WIDTH
            cell_width = max(len(text), 1) * CHAR_WIDTH
            axis.add_patch(Rectangle((x, y), cell_width, LINE_HEIGHT * 0.92, facecolor=colormap(norm(entropy)),
                                     edgecolor=SURFACE, linewidth=0.4))
            axis.text(x, y + LINE_HEIGHT * 0.47, text, fontsize=FONT_SIZE, family="DejaVu Sans Mono",
                      color=text_color(entropy), va="center", ha="left")
        y += LINE_HEIGHT
    y += PANEL_GAP

bar_axis = figure.add_axes([MARGIN / width, 0.32 / height, 0.45, 0.11 / height])
colorbar = figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=colormap), cax=bar_axis, orientation="horizontal")
colorbar.set_ticks([0, 0.1, 0.5, 1.0, 2.0])
colorbar.ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _position: f"{value:g}".replace(".", ",")))
colorbar.ax.tick_params(labelsize=8, colors=TEXT_SECONDARY, length=0)
colorbar.outline.set_visible(False)
figure.text(MARGIN / width, 0.50 / height,"A tanármodell entrópiája az adott tokennél (nat; 2 felett a legsötétebb)",
            fontsize=8.5, color=TEXT_SECONDARY, ha="left", va="bottom")

figure.savefig(OUTPUT, facecolor=SURFACE)
print(f"saved {OUTPUT} ({width:.1f} x {height:.1f} in)")
