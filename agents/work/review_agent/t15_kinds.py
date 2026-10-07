import sys, json, random
from collections import Counter
sys.path.insert(0, r"D:\preference_destillation")
import library.domain.math_tutor.labels as L
from library.domain.math_tutor.questions import generator, SAMPLE_LABELS, solution_foci, what_if_foci, kind_weights
from library.agent.question_templates import related
from library.domain.math_tutor import threads
gen = generator()
items = {}
for name in ("items.jsonl", "renumbered.jsonl"):
    for l in open(L.ITEMS_DIR / name, encoding="utf-8"):
        d = json.loads(l); items[d["id"]] = d
labels = {}
for l in open(SAMPLE_LABELS, encoding="utf-8"):
    r = json.loads(l); labels[(r["item_id"], r["turn"])] = r["block"]
by_depth = {0: Counter(), 1: Counter(), 2: Counter()}
after_off = Counter()
for seed in range(20):
    threads.SEED = seed
    for plan in threads.plan_threads([items[i] for i in sorted({k[0] for k in labels})], labels, gen):
        for position, m in enumerate(plan["messages"]):
            by_depth[position][m["kind"]] += 1
            if position > 0 and plan["messages"][position-1]["kind"] == "off_topic":
                after_off[m["kind"]] += 1
for d, c in by_depth.items():
    n = sum(c.values()); print(d, n, {k: f"{100*v/n:.0f}%" for k, v in c.most_common()})
print("kind of the message right after an off-topic one:", after_off.most_common())
