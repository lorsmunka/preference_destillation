import json, sys, collections
sys.path.insert(0, r"D:\preference_destillation")
from library.domain.math_tutor.similarity import collect_name_words, normalise, sequence_ratio, WordSequenceIndex

BASE = r"D:\preference_destillation\library\domain\math_tutor\items"
items = [json.loads(l) for l in open(BASE + r"\items.jsonl", encoding="utf-8")]
targets = [json.loads(l) for l in open(BASE + r"\reword_targets.jsonl", encoding="utf-8")]
by_id = {i["id"]: i for i in items}
split_by_family = {i["family"]: i["split"] for i in items}

target_splits = collections.Counter(split_by_family.get(t["id"], "dropped/absent") for t in targets)
print("reword targets (frozen as train):", len(targets), "-> their family's split now:", dict(target_splits))
reword_split = collections.Counter(i["split"] for i in items if str(i["source"]).startswith("reword:"))
print("rewordings by split now:", dict(reword_split))
test_originals = [i for i in items if i["split"] == "test" and not str(i["source"]).startswith("reword:")]
test_with_rewordings = {i["family"] for i in items if i["split"] == "test" and str(i["source"]).startswith("reword:")}
print("test families:", len({i['family'] for i in items if i['split']=='test'}), "with rewordings:", len(test_with_rewordings))

# seeded siblings: similarity between test seeded items and their same-seed train siblings
name_words = collect_name_words(i["problem"] for i in items)
words = {i["id"]: normalise(i["problem"], name_words) for i in items}
by_seed = collections.defaultdict(list)
for i in items:
    if str(i["source"]).startswith("seeded:"):
        by_seed[i["source"]].append(i)
ratios, shape_same = [], 0
pairs = 0
for seed, group in by_seed.items():
    tests = [i for i in group if i["split"] == "test"]
    trains = [i for i in group if i["split"] == "train"]
    for t in tests:
        best = max((sequence_ratio(words[t["id"]], words[r["id"]]), r) for r in trains) if trains else None
        if best:
            pairs += 1
            ratios.append(best[0])
            shape_same += best[1]["problem_shape"] == t["problem_shape"]
ratios.sort()
if ratios:
    print("test seeded items with a same-seed train sibling:", pairs, "nearest-sibling ratio median %.2f p90 %.2f max %.2f" % (ratios[len(ratios)//2], ratios[int(len(ratios)*0.9)], ratios[-1]), "same problem shape:", shape_same)

# overall: nearest train item for each test original (prefix-filtered at 0.6)
train_ids = [i["id"] for i in items if i["split"] == "train"]
index = WordSequenceIndex(0.6, (words[i] for i in words))
for train_id in train_ids:
    index.add(train_id, words[train_id])
buckets = collections.Counter()
examples = []
for t in test_originals:
    neighbour, score = index.most_similar(words[t["id"]])
    bucket = "0.75-0.80" if score > 0.75 else "0.70-0.75" if score > 0.7 else "0.60-0.70" if score >= 0.6 else "<0.60"
    buckets[bucket] += 1
    if score > 0.75 and len(examples) < 4:
        examples.append((round(score, 2), t["id"], t["problem"][:160], neighbour, by_id[neighbour]["problem"][:160]))
print("test originals by nearest train-item ratio:", dict(buckets), "of", len(test_originals))
for e in examples:
    print("   ", e)
