import json, sys, time, random
sys.path.insert(0, r"D:\preference_destillation")
from library.domain.math_tutor.items import validate
from library.domain.math_tutor.renumber import renumber_item
items = [json.loads(l) for l in open(r"D:\preference_destillation\library\domain\math_tutor\items\items.jsonl", encoding="utf-8")]
start = time.time()
for item in items: validate(item)
print("validate all", len(items), "items: %.1fs" % (time.time() - start))
rng = random.Random(0)
sample = random.Random(1).sample(items, 1000)
start = time.time()
for item in sample: renumber_item(item, rng)
print("renumber_item on 1000 items: %.1fs" % (time.time() - start))
