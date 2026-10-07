import sys, json, re
from collections import Counter
sys.path.insert(0, r"D:\preference_destillation")
from library.domain.math_tutor.checks import parse
c = Counter(); ex = []
for l in open(r"D:\preference_destillation\library\domain\math_tutor\questions\sample_labels.jsonl", encoding="utf-8"):
    r = json.loads(l); lines = parse(r["block"]); known = set(); ok = True
    for line in lines:
        if line.kind == "fact": known.add(line.number)
        elif line.kind == "calculation":
            expr = line.text.split("=", 1)[1].rsplit("=", 1)[0].split("remainder")[0]
            for n in re.findall(r"\d+", expr):
                if int(n) not in known and int(n) not in (0, 1):
                    ok = False
            known.add(line.number)
    c["strict ok" if ok else "strict would reject"] += 1
    if not ok and len(ex) < 3: ex.append(r["block"])
print(c); [print(e) for e in ex]
