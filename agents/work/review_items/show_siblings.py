import json, sys, collections
sys.path.insert(0, r"D:\preference_destillation")
from library.domain.math_tutor.items import code_literals
items = [json.loads(l) for l in open(r"D:\preference_destillation\library\domain\math_tutor\items\items.jsonl", encoding="utf-8")]
by_seed = collections.defaultdict(list)
for i in items:
    if str(i["source"]).startswith("seeded:"):
        by_seed[i["source"]].append(i)
shown = 0
for seed, group in by_seed.items():
    for t in [i for i in group if i["split"] == "test"]:
        for r in [i for i in group if i["split"] == "train"]:
            if sorted(code_literals(r["solve"])) == sorted(code_literals(t["solve"])) and shown < 4:
                shown += 1
                print(seed, "\n  TEST ", t["problem"], t["answers"], "\n  TRAIN", r["problem"], r["answers"])
