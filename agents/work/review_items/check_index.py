import json, sys, random
sys.path.insert(0, r"D:\preference_destillation")
from library.domain.math_tutor.similarity import collect_name_words, normalise, sequence_ratio, WordSequenceIndex
items = [json.loads(l) for l in open(r"D:\preference_destillation\library\domain\math_tutor\items\items.jsonl", encoding="utf-8")]
rng = random.Random(3)
# pick clusters: an item plus its rewordings, so high ratios exist
sample = rng.sample(items, 900)
names = collect_name_words(i["problem"] for i in items)
words = [normalise(i["problem"], names) for i in sample]
# add perturbed copies (one word swapped) so that >0.8 pairs exist
for w in words[:300]:
    w2 = list(w); k = rng.randrange(len(w2)); w2[k] = "zzz"; words.append(w2)
for threshold in (0.8, 0.5):
    index = WordSequenceIndex(threshold, words)
    mismatches = 0
    for k, w in enumerate(words):
        index.add(str(k), w)
    for k, w in enumerate(words):
        _, got = index.most_similar(w, exclude=str(k))
        brute = max(sequence_ratio(w, o) for j, o in enumerate(words) if j != k)
        if (brute >= threshold) != (got >= threshold) or (brute >= threshold and abs(brute - got) > 1e-12):
            mismatches += 1
    print("threshold", threshold, "mismatches vs brute force:", mismatches, "of", len(words))
