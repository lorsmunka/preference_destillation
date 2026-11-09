"""Thesis figures 9.7 and 9.9 (chapter 9, section 9.5): snippets of the attention study's explorer
(runs/attention_study/attention_study.html, "Explorer: one reply, token by token"), Gemma next to a student, the text
shaded by where each model's kereső figyelemfej looks while it writes one token of Gemma's test reply. The kereső
head is the one of plot_18: the head that puts the largest share of its attention on what has to be looked up.
The page itself stores only layer means, so the one head's weights are read from the saved attention rows (Gemma's
measured in bf16 on the rented card, runs/attention_study/teacher.pt) and put into the page in the page's own format
(per reply token: the weights over the groups scaled to their maximum, 0-255, plus that maximum), as a one-layer
model. One colour scale for both panels (the page's "one scale for all panels").

- 19a_code_attention_snippet.png: shop~v15#0 ("Which lines hold the code that places an order?", Gemma's scope
  checkout.py 11-22), while writing the first line number. Gemma 4 12B layer 36 head 15 vs the 12.0M KL student after
  pass 10 (code-agent-kl-m-s1; rows: agents/work/chapter9_synthesis_2026-11-06/code_attention_passes.py rows
  code-kl-m-p10) layer 2 head 4. The red bar marks the target lines; the box shows the lines around them.
- 19b_math_attention_snippet.png: the pumpkins record, while writing the "5" of "pumpkins picked = 35". Gemma layer 18
  head 1 vs math-kl-m-s1 (16.5M, KL, after pass 10) layer 2 head 1. The boxed token is the one being written; the
  conversation is shown from the pupil's message on (the earlier task and solution are hidden).
Each panel title adds the measured value: code, the share on the target lines; math, the strongest weight.

How: a copy of the page with a small script appended (puts in the heads, picks the record, the two models and the
token, hides the rest, swaps the English labels for Hungarian ones) goes next to the page (its explorer/ files load
relative to it), headless Chrome screenshots it, the copy is deleted and the PNG trimmed.
Run: python thesis_work/grafikonok/plot_19_attention_snippets.py"""
import base64
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(r"D:\preference_destillation")
sys.path.insert(0, str(ROOT / "experiments"))

import torch
from PIL import Image, ImageChops

import attention_report as report

PAGE = ROOT / "runs" / "attention_study" / "attention_study.html"
OUTPUT_DIR = ROOT / "thesis_work" / "grafikonok"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
CODE_STUDENT_ROWS = ROOT / "agents" / "work" / "chapter9_synthesis_2026-11-06" / "code_rows" / "code-kl-m-p10.pt"

# model key on the page -> (where its rows come from, layer, head), 0-based; the kereső heads of plot_18
SNAPSHOTS = [
    {"file": "19a_code_attention_snippet.png", "kind": "code", "listIndex": 16, "probeIndex": 82, "row": None,
     "heads": {"gemma": ("gemma", 35, 14), "code-kl-m": (CODE_STUDENT_ROWS, 1, 3)},
     "titles": {"gemma": "Gemma 4 12B (36. réteg, 15. fej)",
                "code-kl-m": "Tanulómodell, 12,0M, 10 epoch (2. réteg, 4. fej)"},
     "font": 15, "width": 1000, "boxHeight": 330},
    {"file": "19b_math_attention_snippet.png", "kind": "math", "listIndex": 2, "probeIndex": 2, "row": 20,
     "heads": {"gemma": ("gemma", 17, 0), "math-kl-m-s1": ("math-kl-m-s1", 1, 0)},
     "titles": {"gemma": "Gemma 4 12B (18. réteg, 1. fej)",
                "math-kl-m-s1": "Tanulómodell, 16,5M, 10 epoch (2. réteg, 1. fej)"},
     "font": 15, "width": 1000, "boxHeight": None},
]

SNAPSHOT_SCRIPT = """
<script>
(function () {
  const SNAP = __CONFIG__;
  document.documentElement.dataset.theme = "light";
  const models = Object.keys(SNAP.override);
  DATA.model_sets[SNAP.kind]["Snippet"] = models;
  setChoice[SNAP.kind] = "Snippet";
  models.forEach(key => { layerChoice[`${key}|${SNAP.kind}`] = 0; });
  colsSelect.value = String(models.length);
  fontSelect.value = String(SNAP.font);
  scaleSelect.value = "shared";
  // wait for the page's own first record, then load ours, put the one-head data in and show it
  const start = setInterval(() => {
    if (!current.data) return;
    clearInterval(start);
    const meta = DATA.explorer[SNAP.listIndex];
    const script = document.createElement("script");
    script.src = meta.file;
    script.onload = () => {
      Object.assign(window.EXPLORER[meta.index].models, SNAP.override);
      pickProbe(SNAP.listIndex);
    };
    document.head.appendChild(script);
  }, 50);
  const wait = setInterval(() => {
    if (!current.data || current.data.index !== SNAP.probeIndex || current.panels.length !== models.length) return;
    clearInterval(wait);
    if (SNAP.row !== null) setRow(SNAP.row);
    const panels = document.getElementById("ex-panels");
    [...document.body.children].forEach(child => { if (child.tagName !== "SCRIPT") child.style.display = "none"; });
    const holder = document.createElement("div");
    holder.style.cssText = `width:${SNAP.width}px;padding:10px;background:var(--surface)`;
    document.body.appendChild(holder);
    holder.appendChild(panels);
    document.body.style.cssText = "margin:0;padding:0;background:var(--surface)";
    panels.style.gridTemplateColumns = `repeat(${models.length}, minmax(0, 1fr))`;
    const percent = value => (value * 100).toFixed(1).replace(".", ",") + "%";
    for (const entry of current.panels) {
      const header = entry.note.parentElement.querySelector("header");
      header.innerHTML = "";
      header.style.cssText = "height:auto;white-space:normal;display:block";
      const title = document.createElement("b");
      title.textContent = SNAP.titles[entry.key];
      header.appendChild(title);
      const measured = document.createElement("div");
      measured.style.cssText = "font-weight:400;color:var(--text-2)";
      if (SNAP.kind === "code") {
        const share = entry.cells.reduce((sum, cell, group) => sum + (current.data.regions[group] === "target lines" ? cell._weight : 0), 0);
        measured.textContent = `a keresett sorokon ${percent(share)}`;
      } else {
        measured.textContent = `legnagyobb súly ${percent(entry.model.maxima[0][current.row])}`;
      }
      header.appendChild(measured);
      [entry.note, entry.top, entry.tail].forEach(node => { if (node) node.style.display = "none"; });
      if (SNAP.kind !== "code") {
        const query = current.data.queries[current.row];
        const labels = current.data.labels;
        const messageStart = labels.findIndex((label, group) => label === "message" && labels[group - 1] === "<") - 1;
        entry.cells.forEach((cell, group) => { if (group > query || group < messageStart) cell.style.display = "none"; });
        const earlier = document.createElement("span");
        earlier.style.color = "var(--muted)";
        earlier.textContent = "… (a korábbi feladat és megoldása)\\n";
        entry.cells[messageStart].before(earlier);
        entry.cells[query].classList.remove("q");
        const written = document.createElement("span");
        written.className = "q";
        written.textContent = current.data.reply[current.row];
        entry.cells[query].after(written);
        entry.box.style.height = "auto";
      } else {
        entry.box.style.height = SNAP.boxHeight + "px";
        const target = entry.box.querySelector(".target");
        entry.box.scrollTop = Math.max(0, target.offsetTop - 70);
      }
    }
    document.title = "ready";
  }, 50);
})();
</script>
"""


def one_head(rows_by_probe, probe_index: int, layer: int, head: int) -> dict:
    """One head's weights for one record in the page's format: a model with a single "layer"."""
    weights = report.view_of(rows_by_probe[probe_index])[0][layer, head]          # [reply tokens, groups]
    maxima = weights.max(dim=1).values.clamp(min=1e-12)
    scaled = (weights / maxima[:, None] * 255).round().to(torch.uint8)
    return {"layers": 1, "data": base64.b64encode(scaled.numpy().tobytes()).decode("ascii"),
            "maxima": [[round(float(value), 5) for value in maxima]], "outside": None}


def trim(path: Path) -> None:
    image = Image.open(path).convert("RGB")
    background = Image.new("RGB", image.size, image.getpixel((2, 2)))
    box = ImageChops.difference(image, background).getbbox()
    if box:
        margin = 16
        image.crop((max(0, box[0] - margin), max(0, box[1] - margin), min(image.width, box[2] + margin),
                    min(image.height, box[3] + margin))).save(path)


loaded = {}


def rows_of(source):
    if source not in loaded:
        loaded[source] = (torch.load(source, weights_only=False) if isinstance(source, Path)
                          else report.load_rows(report.GEMMA if source == "gemma" else source))
    return loaded[source]


page = PAGE.read_text(encoding="utf-8")
assert page.count("</body>") == 1, "the page's end tag moved"
profile = tempfile.mkdtemp(prefix="chrome_snippet_")
for snapshot in SNAPSHOTS:
    config = {name: value for name, value in snapshot.items() if name != "heads"}
    config["override"] = {key: one_head(rows_of(source), snapshot["probeIndex"], layer, head)
                          for key, (source, layer, head) in snapshot["heads"].items()}
    copy = PAGE.parent / f"_snippet_{snapshot['kind']}.html"
    copy.write_text(page.replace("</body>", SNAPSHOT_SCRIPT.replace("__CONFIG__", json.dumps(config)) + "</body>"),
                    encoding="utf-8")
    output = OUTPUT_DIR / snapshot["file"]
    try:
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--user-data-dir={profile}",
                        "--force-device-scale-factor=2", f"--window-size={snapshot['width'] + 20},1400",
                        "--virtual-time-budget=30000", f"--screenshot={output}", copy.as_uri()],
                       check=True, capture_output=True, timeout=180)
    finally:
        os.remove(copy)
    trim(output)
    print("saved", output, Image.open(output).size)
