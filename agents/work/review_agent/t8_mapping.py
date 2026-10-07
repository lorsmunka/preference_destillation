import sys, json, re
from collections import Counter
sys.path.insert(0, r"D:\preference_destillation")
import library.domain.math_tutor.labels as L
from library.domain.math_tutor.annotations import turn_executions
originals = {}
for l in open(L.ITEMS_DIR / "items.jsonl", encoding="utf-8"):
    d = json.loads(l); originals[d["id"]] = d
texts = lambda item: " ".join([item["problem"]] + [f["question"] for f in item["follow_ups"]])
why = Counter(); ex = {}
for i, l in enumerate(open(L.ITEMS_DIR / "renumbered.jsonl", encoding="utf-8")):
    copy = json.loads(l); orig = originals.get(copy["renumbered_from"])
    if orig is None: why["no original"] += 1; continue
    mapping = L.number_mapping(texts(orig), texts(copy))
    if mapping is None: why["text"] += 1; continue
    reason = None
    for old_run, new_run in zip(turn_executions(orig), turn_executions(copy)):
        for old, new in zip([s.value for s in old_run.steps if s.computes], [s.value for s in new_run.steps if s.computes]):
            if mapping.setdefault(old, new) != new:
                reason = ("computed value collides with a text number" if old in [int(x) for x in re.findall(r"\d+", texts(orig))]
                          else "computed value collides with another computed value")
                if old in (0, 1): reason = "computed value 0/1"
                break
        if reason: break
    if reason:
        why[reason] += 1; ex.setdefault(reason, (orig["id"], orig["problem"], copy["problem"], [s.value for r in turn_executions(orig) for s in r.steps if s.computes], [s.value for r in turn_executions(copy) for s in r.steps if s.computes]))
    else: why["ok"] += 1
print(why)
for k, v in ex.items(): print(k, v)
