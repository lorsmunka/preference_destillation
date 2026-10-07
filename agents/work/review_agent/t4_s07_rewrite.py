import sys, json, re
from collections import Counter
sys.path.insert(0, r"D:\preference_destillation")
from library.domain.math_tutor.annotations import turn_executions
from library.domain.math_tutor.labels import gold_calculations, computed_values
from library.domain.math_tutor.checks import check_against_trace
from library.domain.math_tutor.conversation import Conversation
from library.domain.math_tutor.questions import tasks_of
items = [json.loads(l) for l in open(r"D:\preference_destillation\library\domain\math_tutor\items\items.jsonl", encoding="utf-8")]
outcome = Counter(); ex = {}
for item in items:
    if item["slice"] != "S07": continue
    runs = turn_executions(item)
    for turn in range(len(runs)):
        calcs = gold_calculations(item, turn, runs)
        if not any(c.startswith("remainder of") for c in calcs): continue
        # best-case rewrite: 'a / b = q remainder r'
        new = []
        for c in calcs:
            m = re.match(r"remainder of (\d+) / (\d+) = (\d+)", c)
            if m:
                a, b = int(m.group(1)), int(m.group(2)); new.append(f"{a} / {b} = {a//b} remainder {a%b}")
            else: new.append(c)
        tasks = tasks_of(item)
        ctx = Conversation().solve_context(" ".join(tasks[:turn+1]))  # rough: all texts so numbers are available
        facts = sorted({int(n) for c in new for n in re.findall(r"\d+", c.split("=")[0])})
        block = "\n".join(["<solution>"] + [f"f{i} = {n}" for i, n in enumerate(facts)] + [f"s{i} = {c}" for i, c in enumerate(new)] + [f"Answer: {item['answers'][turn]}", "</solution>"])
        ctx += " " + " ".join(str(a) for a in item["answers"][:turn]) + " " + " ".join(str(v) for t in range(turn) for v in computed_values(runs, t))
        p = check_against_trace(block, computed_values(runs, turn), item["answers"][turn], ctx)
        key = "passes" if p is None else re.sub(r"\d+", "N", p)[:50]
        outcome[key] += 1; ex.setdefault(key, (item["id"], turn, new, computed_values(runs, turn)))
print(outcome)
for k, v in ex.items(): print(k, v)
