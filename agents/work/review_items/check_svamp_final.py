import json, sys, collections
sys.path.insert(0, r"D:\preference_destillation")
from library.domain.math_tutor.similarity import collect_name_words, normalise, WordSequenceIndex, TrigramIndex, word_trigrams
from library.domain.math_tutor.prune import svamp_problems
items = [json.loads(l) for l in open(r"D:\preference_destillation\library\domain\math_tutor\items\items.jsonl", encoding="utf-8")]
svamp = svamp_problems()
names = collect_name_words([i["problem"] for i in items] + [p["problem"] for p in svamp])
sv = {p["id"]: normalise(p["problem"], names) for p in svamp}
queries = {}
for i in items:
    queries[i["id"]] = normalise(i["problem"], names)
    if i["follow_ups"]:
        queries[i["id"] + "+f1"] = normalise(i["problem"] + " " + i["follow_ups"][0]["question"], names)
index = WordSequenceIndex(0.7, list(sv.values()) + list(queries.values()))
tri = TrigramIndex()
for k, w in sv.items():
    index.add(k, w); tri.add(k, word_trigrams(w))
over = collections.Counter(); ex = []
for k, w in queries.items():
    n, s = index.most_similar(w)
    tn, ts = tri.most_similar(word_trigrams(w))
    tag = "problem+f1" if k.endswith("+f1") else "problem"
    if s > 0.8 or ts > 0.5:
        over[tag + " over threshold"] += 1
        ex.append((k, round(s, 2), round(ts, 2)))
    elif s > 0.7:
        over[tag + " 0.7-0.8"] += 1
print(dict(over)); print(ex[:5])
