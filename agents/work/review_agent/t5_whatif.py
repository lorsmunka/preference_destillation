"""What-if gold coherence: transfer() maps every `old` in the shown block's facts to `new`, including facts that
are a different quantity with the same value (an earlier answer read from state, a second quantity)."""
import sys, json, random, re
from collections import Counter
sys.path.insert(0, r"D:\preference_destillation")
from library.domain.math_tutor.annotations import turn_executions
from library.domain.math_tutor.questions import what_if_foci, code_block, tasks_of, sources_of
from library.domain.math_tutor.renumber import replace_in_code
from library.domain.math_tutor.functions import execute
from library.domain.math_tutor.checks import parse
from library.domain.math_tutor.items import REMAINDER_SLICE, text_numbers
items = [json.loads(l) for l in open(r"D:\preference_destillation\library\domain\math_tutor\items\items.jsonl", encoding="utf-8")]
rng = random.Random(0)
outcome = Counter(); examples = []
for item in items:
    if item["split"] != "train" or str(item.get("source","")).startswith("reword:"): continue
    turn = len(item["follow_ups"])
    runs = turn_executions(item)
    block = code_block(item, turn, runs)
    for focus in what_if_foci(item, turn, block, rng):
        old, new, gold = focus.slots["old"], focus.slots["new"], focus.data["gold"]
        if not gold:
            outcome["no gold"] += 1; continue
        # the truth: the changed code's own fact steps (state copies keep the old value)
        state = runs[turn-1].result if turn > 0 else None
        changed = execute(replace_in_code(sources_of(item)[turn], {old: new}), state, item["slice"] == REMAINDER_SLICE, trace=True)
        true_facts = sorted(s.value for s in changed.steps if not s.computes)
        gold_facts = sorted(l.number for l in parse(gold) if l.kind == "fact")
        if true_facts != gold_facts:
            outcome["gold passes the check, facts wrong"] += 1
            if len(examples) < 4: examples.append((item["id"], turn, old, new, block, gold, true_facts))
        else:
            outcome["ok"] += 1
        # ambiguity: old written more than once in this turn's text
        if text_numbers(tasks_of(item)[turn]).count(old) > 1:
            outcome["old written twice in the turn's text (both changed)"] += 1
print(outcome)
for e in examples:
    print("----", e[0], "turn", e[1], f"what if {e[3]} instead of {e[2]}")
    print(e[4]); print("GOLD:"); print(e[5]); print("true fact values", e[6])
