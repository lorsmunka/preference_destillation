"""Thesis figures (chapter 9, section 9.5): snippets of the attention study's explorer
(runs/attention_study/attention_study.html, "Explorer: one reply, token by token"), Gemma next to a student, the text
shaded by where each model looks while it writes one token of Gemma's test reply. The explorer shows a layer's
mean over its heads; the layers are the ones holding the best heads of plot_18 (Gemma: 36 for code, 18 for math;
the students: 2). One colour scale for both panels (the page's "one scale for all panels").
Not the mean over every head: there Gemma's few lookup heads drown among its 768 (code: Gemma 3.0% vs the
student 3.2% on the target lines, both an even spread), so the averaged view hides what the heads do.

- 19a_code_attention_snippet.png: shop~v15#0 ("Which lines hold ...", Gemma's scope checkout.py 11-22), while writing
  the first line number; Gemma 4 12B layer 36 vs code-kl-m (12.0M, KL, after pass 8: the student the page holds)
  layer 2. The red bar marks the target lines. The box shows the lines around the target.
- 19b_math_attention_snippet.png: the pumpkins record, while writing the "5" of "pumpkins picked = 35"; Gemma layer
  18 vs math-kl-m-s1 (16.5M, KL, after pass 10) layer 2. The boxed token is the one being written; the conversation
  is shown from the pupil's message on (the earlier task and solution are hidden).
Each panel title adds the measured value: code, the share on the target lines; math, the strongest weight.

How: a copy of the page with a small script appended (picks the record, the two models, the layers and the token,
hides the rest, swaps the English labels for Hungarian ones) goes next to the page (its explorer/ files load
relative to it), headless Chrome screenshots it, the copy is deleted and the PNG trimmed.
Run: python thesis_work/grafikonok/plot_19_attention_snippets.py"""
import json
import os
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageChops

ROOT = Path(r"D:\preference_destillation")
PAGE = ROOT / "runs" / "attention_study" / "attention_study.html"
OUTPUT_DIR = ROOT / "thesis_work" / "grafikonok"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

SNAPSHOTS = [
    {"file": "19a_code_attention_snippet.png", "kind": "code", "listIndex": 16, "probeIndex": 82, "row": None,
     "models": ["gemma", "code-kl-m"], "layers": {"gemma|code": 35, "code-kl-m|code": 1},
     "titles": {"gemma": "Gemma 4 12B, 36. réteg", "code-kl-m": "Tanulómodell (12,0M, 8. epoch után), 2. réteg"},
     "font": 15, "width": 1000, "boxHeight": 330},
    {"file": "19b_math_attention_snippet.png", "kind": "math", "listIndex": 2, "probeIndex": 2, "row": 20,
     "models": ["gemma", "math-kl-m-s1"], "layers": {"gemma|math": 17, "math-kl-m-s1|math": 1},
     "titles": {"gemma": "Gemma 4 12B, 18. réteg", "math-kl-m-s1": "Tanulómodell (16,5M, 10. epoch után), 2. réteg"},
     "font": 15, "width": 1000, "boxHeight": None},
]

SNAPSHOT_SCRIPT = """
<script>
(function () {
  const SNAP = __CONFIG__;
  document.documentElement.dataset.theme = "light";
  DATA.model_sets[SNAP.kind]["Snippet"] = SNAP.models;
  setChoice[SNAP.kind] = "Snippet";
  Object.assign(layerChoice, SNAP.layers);
  colsSelect.value = String(SNAP.models.length);
  fontSelect.value = String(SNAP.font);
  scaleSelect.value = "shared";
  pickProbe(SNAP.listIndex);
  const wait = setInterval(() => {
    if (!current.data || current.data.index !== SNAP.probeIndex || current.panels.length !== SNAP.models.length) return;
    clearInterval(wait);
    if (SNAP.row !== null) setRow(SNAP.row);
    const panels = document.getElementById("ex-panels");
    [...document.body.children].forEach(child => { if (child.tagName !== "SCRIPT") child.style.display = "none"; });
    const holder = document.createElement("div");
    holder.style.cssText = `width:${SNAP.width}px;padding:10px;background:var(--surface)`;
    document.body.appendChild(holder);
    holder.appendChild(panels);
    document.body.style.cssText = "margin:0;padding:0;background:var(--surface)";
    panels.style.gridTemplateColumns = `repeat(${SNAP.models.length}, minmax(0, 1fr))`;
    for (const entry of current.panels) {
      const header = entry.note.parentElement.querySelector("header");
      header.innerHTML = "";
      header.style.height = "auto";
      const title = document.createElement("b");
      title.textContent = SNAP.titles[entry.key];
      header.appendChild(title);
      const percent = value => (value * 100).toFixed(1).replace(".", ",") + "%";
      const measured = document.createElement("span");
      measured.style.cssText = "margin-left:10px;font-weight:400;color:var(--text-2)";
      if (SNAP.kind === "code") {
        const share = entry.cells.reduce((sum, cell, group) => sum + (current.data.regions[group] === "target lines" ? cell._weight : 0), 0);
        measured.textContent = `a keresett sorokon ${percent(share)}`;
      } else {
        measured.textContent = `legnagyobb súly ${percent(entry.model.maxima[+entry.select.value][current.row])}`;
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


def trim(path: Path) -> None:
    image = Image.open(path).convert("RGB")
    background = Image.new("RGB", image.size, image.getpixel((2, 2)))
    box = ImageChops.difference(image, background).getbbox()
    if box:
        margin = 16
        image.crop((max(0, box[0] - margin), max(0, box[1] - margin), min(image.width, box[2] + margin),
                    min(image.height, box[3] + margin))).save(path)


page = PAGE.read_text(encoding="utf-8")
assert page.count("</body>") == 1, "the page's end tag moved"
profile = tempfile.mkdtemp(prefix="chrome_snippet_")
for snapshot in SNAPSHOTS:
    copy = PAGE.parent / f"_snippet_{snapshot['kind']}.html"
    copy.write_text(page.replace("</body>", SNAPSHOT_SCRIPT.replace("__CONFIG__", json.dumps(snapshot)) + "</body>"),
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
