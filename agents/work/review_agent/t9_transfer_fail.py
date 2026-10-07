import sys, json, re
from collections import Counter
sys.path.insert(0, r"D:\preference_destillation")
import library.domain.math_tutor.labels as L
from library.domain.math_tutor.annotations import turn_executions
from library.domain.math_tutor.checks import check_against_trace
originals = {}
for l in open(L.ITEMS_DIR / "items.jsonl", encoding="utf-8"):
    d = json.loads(l); originals[d["id"]] = d
sample = {}
for l in open(r"D:\preference_destillation\library\domain\math_tutor\questions\sample_labels.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r["item_id"] in originals: sample[(r["item_id"], r["turn"])] = r["block"]
ids = {k[0] for k in sample}
texts = lambda item: " ".join([item["problem"]] + [f["question"] for f in item["follow_ups"]])
labels = dict(sample); why = Counter(); shown = 0
for l in open(L.ITEMS_DIR / "renumbered.jsonl", encoding="utf-8"):
    copy = json.loads(l)
    if copy["renumbered_from"] not in ids: continue
    original = originals[copy["renumbered_from"]]
    mapping = L.number_mapping(texts(original), texts(copy))
    o_runs, c_runs = turn_executions(original), turn_executions(copy)
    for a, b in zip(o_runs, c_runs):
        for old, new in zip([s.value for s in a.steps if s.computes], [s.value for s in b.steps if s.computes]):
            if mapping is not None and mapping.setdefault(old, new) != new: mapping = None
    if mapping is None: continue
    for turn in range(len(c_runs)):
        block = labels.get((original["id"], turn))
        if block is None: break
        nb = L.transfer(block, mapping, L.gold_calculations(copy, turn, c_runs), copy["answers"][turn])
        ctx = L.solve_context(copy, turn, labels)
        p = check_against_trace(nb, L.computed_values(c_runs, turn), copy["answers"][turn], ctx or "")
        if p:
            why[re.sub(r"\d+", "N", p)[:60]] += 1
            if shown < 40 and "Answer" in p:
                shown += 1; print("----", copy["id"], turn, p); print(original["problem"], "|", copy["problem"]); print(block); print(nb)
            break
        labels[(copy["id"], turn)] = nb
print(why)
