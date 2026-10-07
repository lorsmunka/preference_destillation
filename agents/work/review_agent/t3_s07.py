import sys, json, re
from collections import Counter
sys.path.insert(0, r"D:\preference_destillation")
from library.domain.math_tutor.annotations import turn_executions
from library.domain.math_tutor.labels import gold_calculations
from library.domain.math_tutor.checks import check_lines
items = [json.loads(l) for l in open(r"D:\preference_destillation\library\domain\math_tutor\items\items.jsonl", encoding="utf-8")]
kinds = Counter(); items_hit = set(); ex = {}
s07_items = 0
for item in items:
    if item["slice"] != "S07": continue
    s07_items += 1
    runs = turn_executions(item)
    for turn in range(len(runs)):
        for c in gold_calculations(item, turn, runs):
            if check_lines(f"x = 1\ns = {c}\nAnswer: 1"):
                k = "remainder of" if c.startswith("remainder of") else ("rounded up, compound" if "rounded up" in c else ("// left in" if "//" in c else ("remainder, compound" if "remainder" in c else "other")))
                kinds[k] += 1; items_hit.add(item["id"]); ex.setdefault(k, c)
print("S07 items", s07_items, "items with a failing gold line", len(items_hit))
print(kinds); print(ex)
