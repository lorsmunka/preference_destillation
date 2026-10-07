import json, sys, collections
sys.path.insert(0, r"D:\preference_destillation")
from library.domain.math_tutor.items import code_literals
items = [json.loads(l) for l in open(r"D:\preference_destillation\library\domain\math_tutor\items\items.jsonl", encoding="utf-8")]
seeds = {json.loads(l)["id"]: json.loads(l) for l in open(r"D:\preference_destillation\library\domain\math_tutor\benchmarks\seeds.jsonl", encoding="utf-8")}
by_seed = collections.defaultdict(list)
for i in items:
    if str(i["source"]).startswith("seeded:"):
        by_seed[i["source"]].append(i)
same_answer = same_literals = same_first_answer = pairs = 0
seed_numbers_kept = 0
for seed, group in by_seed.items():
    tests = [i for i in group if i["split"] == "test"]
    trains = [i for i in group if i["split"] == "train"]
    for t in tests:
        pairs += 1
        if any(r["answers"][0] == t["answers"][0] for r in trains): same_first_answer += 1
        if any(sorted(code_literals(r["solve"])) == sorted(code_literals(t["solve"])) for r in trains): same_literals += 1
        s = seeds.get(seed.split(":",1)[1])
        if s and s["answer"] == t["answers"][0]: seed_numbers_kept += 1
print("test seeded items with train sibling:", pairs, "| sibling has same first answer:", same_first_answer, "| same turn-0 literals:", same_literals, "| test item keeps seed's answer:", seed_numbers_kept)
