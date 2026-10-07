import json, sys, collections
sys.path.insert(0, r"D:\preference_destillation")
from library.domain.math_tutor.similarity import collect_name_words, normalise, sequence_ratio, WordSequenceIndex
from library.domain.math_tutor.prune import svamp_problems

BENCH = r"D:\preference_destillation\library\domain\math_tutor\benchmarks"
seeds = [json.loads(l) for l in open(BENCH + r"\seeds.jsonl", encoding="utf-8")]
test = [json.loads(l) for l in open(BENCH + r"\real_test.jsonl", encoding="utf-8")]
svamp = svamp_problems()
items = [json.loads(l) for l in open(r"D:\preference_destillation\library\domain\math_tutor\items\items.jsonl", encoding="utf-8")]
name_words = collect_name_words([s["problem"] for s in seeds] + [p["problem"] for p in svamp])
seed_words = {s["id"]: normalise(s["problem"], name_words) for s in seeds}
index = WordSequenceIndex(0.6, list(seed_words.values()) + [normalise(p["problem"], name_words) for p in test])
for sid, w in seed_words.items():
    index.add(sid, w)
buckets = collections.Counter()
examples = []
close_seed_ids = {}
for p in test:
    neighbour, score = index.most_similar(normalise(p["problem"], name_words))
    b = "0.7-0.8" if score > 0.7 else "0.6-0.7" if score >= 0.6 else "<0.6"
    buckets[b] += 1
    if score >= 0.6:
        close_seed_ids[neighbour] = (p["id"], round(score, 2))
    if score > 0.7 and len(examples) < 4:
        examples.append((round(score, 2), p["problem"][:170], [s for s in seeds if s["id"] == neighbour][0]["problem"][:170]))
print("SVAMP test problems by nearest seed ratio:", dict(buckets), "of", len(test))
for e in examples:
    print("   ", e)
# how many training items were written from those close seeds
seeded_from_close = [i for i in items if str(i["source"]).startswith("seeded:") and i["source"].split(":", 1)[1] in close_seed_ids]
print("seeds within 0.6 of a SVAMP test problem:", len(close_seed_ids), "| items (originals+rewordings) seeded from them:",
      len(seeded_from_close), collections.Counter(i["split"] for i in seeded_from_close))
