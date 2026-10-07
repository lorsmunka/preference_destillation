import json, sys, re, collections
sys.path.insert(0, r"D:\preference_destillation")
from library.domain.math_tutor.renumber import slots_of
ITEMS = r"D:\preference_destillation\library\domain\math_tutor\items\items.jsonl"
COPIES = r"D:\preference_destillation\library\domain\math_tutor\items\renumbered.jsonl"
items = {json.loads(l)["id"]: json.loads(l) for l in open(ITEMS, encoding="utf-8")}
renumbered_from = collections.Counter(json.loads(l)["renumbered_from"] for l in open(COPIES, encoding="utf-8"))
MISSING = {"doubled": 2, "doubling": 2, "tripled": 3, "halved": 2, "twins": 2, "twin": 2, "triplets": 3, "weekend": 2,
           "weekends": 2, "fortnight": 14, "decade": 10, "century": 100, "tricycle": 3, "tricycles": 3, "bicycle": 2,
           "bicycles": 2, "trio": 3, "duo": 2, "hand": 5, "hands": 2, "octopus": 8, "spider": 8, "spiders": 8,
           "triangle": 3, "triangles": 3, "square": 4, "squares": 4, "hexagon": 6, "pentagon": 5, "month": 30, "months": 30,
           "seconds": 60, "second": 60, "thrice": 3, "quadruple": 4, "once": 1, "each": None}
hits = collections.Counter(); examples = []
for item in items.values():
    if item["id"] not in renumbered_from:
        continue
    texts = " ".join([item["problem"]] + [f["question"] for f in item["follow_ups"]]).lower()
    slots = set(slots_of(item) or [])
    for word, value in MISSING.items():
        if value and re.search(rf"\b{word}\b", texts) and value in slots:
            hits[word] += 1
            if len(examples) < 12:
                examples.append((word, item["id"], item["problem"][:170], [f["question"][:90] for f in item["follow_ups"]], item["solve"].replace("\n", " ; ")[:200]))
print("renumbered items where a word implying v (not in CONSTANT_WORDS) co-occurs with slot v:", dict(hits))
for e in examples:
    print("  ", e)
