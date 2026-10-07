"""Label transfer to renumbered copies: number_mapping reads digits only, so number WORDS that renumber swapped
('four friends' -> 'seven friends') are not in the mapping."""
import sys, json, re
from collections import Counter
from pathlib import Path
sys.path.insert(0, r"D:\preference_destillation")
import library.domain.math_tutor.labels as L
from library.domain.math_tutor.items import text_numbers
from library.domain.math_tutor.checks import parse
from library.domain.math_tutor.annotations import turn_executions
scratch = Path(r"C:\Users\Ors\AppData\Local\Temp\claude\D--preference-destillation\ffe199e2-eaba-41f3-879f-8ba3f4ec1e4f\scratchpad\review_agent\transfer_out.jsonl")
if scratch.exists(): scratch.unlink()
L.LABELS_PATH = scratch
originals = {}
for l in open(L.ITEMS_DIR / "items.jsonl", encoding="utf-8"):
    d = json.loads(l); originals[d["id"]] = d
sample = {}
for l in open(r"D:\preference_destillation\library\domain\math_tutor\questions\sample_labels.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r["item_id"] in originals: sample[(r["item_id"], r["turn"])] = r["block"]
print("original labels in sample:", len(sample), "items", len({k[0] for k in sample}))
labels = dict(sample)
outcomes = L.transfer_to_copies(labels)
print(outcomes)
# compare transferred blocks with a word-aware mapping
copies = {}
for l in open(L.ITEMS_DIR / "renumbered.jsonl", encoding="utf-8"):
    d = json.loads(l)
    if d["renumbered_from"] in {k[0] for k in sample}: copies[d["id"]] = d
texts = lambda item: " ".join([item["problem"]] + [f["question"] for f in item["follow_ups"]])
bad = 0; checked = 0; words_lost = 0
for l in open(scratch, encoding="utf-8"):
    r = json.loads(l); copy = copies[r["item_id"]]; orig = originals[copy["renumbered_from"]]
    a, b = text_numbers(texts(orig)), text_numbers(texts(copy))
    true_map = dict(zip(a, b)) if len(a) == len(b) else None
    if true_map is None: continue
    for o, n in zip([s.value for run in turn_executions(orig) for s in run.steps if s.computes], [s.value for run in turn_executions(copy) for s in run.steps if s.computes]):
        true_map.setdefault(o, n)
    ofacts = [x.number for x in parse(sample[(orig["id"], r["turn"])]) if x.kind == "fact"]
    nfacts = [x.number for x in parse(r["block"]) if x.kind == "fact"]
    expect = [true_map.get(v, v) for v in ofacts]
    checked += 1
    if expect != nfacts:
        bad += 1
        if bad <= 3: print("MISMATCH", r["item_id"], r["turn"], "\n", orig["problem"], "\n", copy["problem"], "\n", sample[(orig["id"], r["turn"])], "\n", r["block"])
# how many copies have a word-number swap at all
for cid, copy in copies.items():
    orig = originals[copy["renumbered_from"]]
    if len(re.findall(r"\d+", texts(orig))) != len(text_numbers(texts(orig))): words_lost += 1
print("transferred blocks checked", checked, "facts differ from word-aware mapping (but passed the check)", bad)
print("copies of sampled originals whose texts contain number words:", words_lost, "of", len(copies))
