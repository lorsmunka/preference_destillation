import sys, json, re
from collections import Counter
sys.path.insert(0, r"D:\preference_destillation")
import library.domain.math_tutor.labels as L
from library.domain.math_tutor.items import text_numbers, digit_numbers
from library.domain.math_tutor.annotations import turn_executions
from library.domain.math_tutor.questions import code_block
from library.domain.math_tutor.checks import check_against_trace, parse
originals = {}
for l in open(L.ITEMS_DIR / "items.jsonl", encoding="utf-8"):
    d = json.loads(l); originals[d["id"]] = d
texts = lambda item: " ".join([item["problem"]] + [f["question"] for f in item["follow_ups"]])
c = Counter(); shown = 0
for l in open(L.ITEMS_DIR / "renumbered.jsonl", encoding="utf-8"):
    copy = json.loads(l); orig = originals.get(copy["renumbered_from"])
    if orig is None or str(orig.get("source","")).startswith("reword:"): continue
    a, b = text_numbers(texts(orig)), text_numbers(texts(copy))
    word_changed = {x for x, y in zip(a, b) if x != y} - set(digit_numbers(texts(orig)))
    if not word_changed: continue
    mapping = L.number_mapping(texts(orig), texts(copy))
    o_runs, c_runs = turn_executions(orig), turn_executions(copy)
    for x, y in zip(o_runs, c_runs):
        for old, new in zip([s.value for s in x.steps if s.computes], [s.value for s in y.steps if s.computes]):
            if mapping is not None and mapping.setdefault(old, new) != new: mapping = None
    if mapping is None: continue
    labels = {(orig["id"], t): code_block(orig, t, o_runs) for t in range(len(o_runs))}
    for t in range(len(c_runs)):
        nb = L.transfer(labels[(orig["id"], t)], mapping, L.gold_calculations(copy, t, c_runs), copy["answers"][t])
        ctx = L.solve_context(copy, t, labels)
        p = check_against_trace(nb, L.computed_values(c_runs, t), copy["answers"][t], ctx or "")
        if p: c["rejected"] += 1; break
        truth = code_block(copy, t, c_runs)
        if sorted(x.number for x in parse(nb) if x.kind == "fact") != sorted(x.number for x in parse(truth) if x.kind == "fact"):
            c["passes with a stale fact"] += 1
            if shown < 2:
                shown += 1; print("----", copy["id"], t); print(orig["problem"]); print(copy["problem"]); print("TRANSFERRED:"); print(nb); print("TRUE FACTS:"); print(truth)
        else: c["ok"] += 1
        labels[(copy["id"], t)] = nb
print(c)
