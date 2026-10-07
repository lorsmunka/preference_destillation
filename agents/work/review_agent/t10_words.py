import sys, json, re
from collections import Counter
sys.path.insert(0, r"D:\preference_destillation")
import library.domain.math_tutor.labels as L
from library.domain.math_tutor.items import text_numbers, digit_numbers
from library.domain.math_tutor.annotations import turn_executions
originals = {}
for l in open(L.ITEMS_DIR / "items.jsonl", encoding="utf-8"):
    d = json.loads(l); originals[d["id"]] = d
texts = lambda item: " ".join([item["problem"]] + [f["question"] for f in item["follow_ups"]])
c = Counter()
for l in open(L.ITEMS_DIR / "renumbered.jsonl", encoding="utf-8"):
    copy = json.loads(l); orig = originals.get(copy["renumbered_from"])
    if orig is None: continue
    c["copies"] += 1
    mapping = L.number_mapping(texts(orig), texts(copy))
    a, b = text_numbers(texts(orig)), text_numbers(texts(copy))
    word_changed = {x for x, y in zip(a, b) if x != y} - set(digit_numbers(texts(orig)))
    if word_changed:
        c["a number word changed value (not in the digit mapping)"] += 1
        # is the unmapped old value still somewhere in the copy's texts or answers? then a stale fact could pass
        if any(v in set(b) | set(copy["answers"]) for v in word_changed):
            c["... and the old value still appears in the copy's text/answers (stale fact could pass)"] += 1
print(c)
