import sys, json, random, re
from collections import Counter
sys.path.insert(0, r"D:\preference_destillation")
from library.domain.math_tutor.annotations import turn_executions
from library.domain.math_tutor.questions import what_if_foci, tasks_of, sources_of, SAMPLE_LABELS
from library.domain.math_tutor.labels import transfer, gold_calculations, computed_values
from library.domain.math_tutor.checks import parse, check_against_trace
from library.domain.math_tutor.items import text_numbers
items = {}
for name in ("items.jsonl", "renumbered.jsonl"):
    for l in open(rf"D:\preference_destillation\library\domain\math_tutor\items\{name}", encoding="utf-8"):
        d = json.loads(l); items[d["id"]] = d
labels = {}
for l in open(SAMPLE_LABELS, encoding="utf-8"):
    r = json.loads(l); labels[(r["item_id"], r["turn"])] = r["block"]
rng = random.Random(0)
out = Counter(); shown = 0
for (iid, turn), block in labels.items():
    item = items[iid]
    if turn != len(item["follow_ups"]): continue
    for seed in range(5):
        rng = random.Random(seed)
        for focus in what_if_foci(item, turn, block, rng):
            old, new, gold = focus.slots["old"], focus.slots["new"], focus.data["gold"]
            if not gold:
                out["no gold"] += 1; continue
            facts_old = [l for l in parse(block) if l.kind == "fact" and l.number == old]
            out["gold"] += 1
            if len(facts_old) > 1 or (turn > 0 and old in item["answers"][:turn]):
                out["old also another quantity"] += 1
                if shown < 3:
                    shown += 1
                    print("----", iid, "turn", turn, f"what if {new} instead of {old}"); print(tasks_of(item)[turn]); print(block); print("GOLD:"); print(gold)
print(out)
