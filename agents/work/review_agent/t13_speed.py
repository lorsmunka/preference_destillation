import sys, json, random, time
sys.path.insert(0, r"D:\preference_destillation")
import library.domain.math_tutor.labels as L
from library.domain.math_tutor.questions import generator, solution_foci, what_if_foci, SAMPLE_LABELS
from library.domain.math_tutor import threads
gen = generator(); print("templates", len(gen.templates))
items = {}
for l in open(L.ITEMS_DIR / "items.jsonl", encoding="utf-8"):
    d = json.loads(l); items[d["id"]] = d
for l in open(L.ITEMS_DIR / "renumbered.jsonl", encoding="utf-8"):
    d = json.loads(l); items[d["id"]] = d
labels = {}
for l in open(SAMPLE_LABELS, encoding="utf-8"):
    r = json.loads(l); labels[(r["item_id"], r["turn"])] = r["block"]
sample_items = [items[i] for i in sorted({k[0] for k in labels})]
start = time.time()
plans = threads.plan_threads(sample_items, labels, gen)
elapsed = time.time() - start
print(f"plan_threads: {len(sample_items)} items in {elapsed:.2f}s -> {elapsed/len(sample_items)*34002/60:.1f} min for 34,002 items")
