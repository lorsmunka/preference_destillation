import sys, json, random
from collections import Counter
sys.path.insert(0, r"D:\preference_destillation")
from library.domain.math_tutor.questions import solution_foci, SAMPLE_LABELS
from library.domain.math_tutor.threads import reply_entry
import library.domain.math_tutor.labels as L
items = {}
for name in ("items.jsonl", "renumbered.jsonl"):
    for l in open(L.ITEMS_DIR / name, encoding="utf-8"):
        d = json.loads(l); items[d["id"]] = d
c = Counter(); ex = None
for l in open(SAMPLE_LABELS, encoding="utf-8"):
    r = json.loads(l)
    for f in solution_foci(items[r["item_id"]], r["turn"], r["block"], random.Random(0)):
        if f.type == "step" and "a_words" in f.slots:
            c["steps with words"] += 1
            if f.slots["a_words"] == f.slots["b_words"]:
                c["a_words == b_words"] += 1; ex = ex or (r["block"], f.slots)
print(c); print(ex)
# truncated explanation (no closing tag) is accepted
entry = reply_entry({"question": "why?", "kind": "why", "route": "explanation"}, "<explanation>Because each bag holds 7 bottles, so we count how many groups of")
print("truncated:", repr(entry["explanation"]), "check:", entry["check"])
