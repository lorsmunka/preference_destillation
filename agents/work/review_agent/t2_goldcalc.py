import sys, json
from collections import Counter
sys.path.insert(0, r"D:\preference_destillation")
from library.domain.math_tutor.annotations import turn_executions
from library.domain.math_tutor.labels import gold_calculations, computed_values
from library.domain.math_tutor.checks import check_lines, parse
items = [json.loads(l) for l in open(r"D:\preference_destillation\library\domain\math_tutor\items\items.jsonl", encoding="utf-8")]
fail = Counter(); total = Counter(); examples = {}
unreachable = Counter()
for item in items:
    ex = turn_executions(item)
    for turn in range(len(ex)):
        calcs = gold_calculations(item, turn, ex)
        block = "\n".join(["<solution>", "x = 1"] + [f"step {i} = {c}" for i, c in enumerate(calcs)] + [f"Answer: {item['answers'][turn]}", "</solution>"])
        p = check_lines(block)
        total[item["slice"]] += 1
        if p:
            key = p.split(":")[0]
            fail[(item["slice"], key)] += 1
            examples.setdefault((item["slice"], key), (item["id"], turn, calcs))
        # does each gold line's parsed number belong to function values?
        values = set(computed_values(ex, turn))
        for line in parse(block):
            if line.kind == "calculation" and line.number not in values:
                unreachable[item["slice"]] += 1
                examples.setdefault(("unreach", item["slice"]), (item["id"], turn, calcs, sorted(values)))
                break
print("turns per slice", dict(total))
for k, v in sorted(fail.items()): print(k, v)
print("gold line value not in function values:", dict(unreachable))
for k, v in examples.items(): print(k, v)
