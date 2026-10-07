import json, sys, re, collections
sys.path.insert(0, r"D:\preference_destillation")
from library.domain.math_tutor.items import text_numbers, UNITS, TENS
from library.domain.math_tutor.renumber import replace_in_text

ITEMS = r"D:\preference_destillation\library\domain\math_tutor\items\items.jsonl"
COPIES = r"D:\preference_destillation\library\domain\math_tutor\items\renumbered.jsonl"
items = {json.loads(l)["id"]: json.loads(l) for l in open(ITEMS, encoding="utf-8")}
copies = [json.loads(l) for l in open(COPIES, encoding="utf-8")]

NUMBER_WORDS = set(UNITS) | set(TENS)
token_pattern = re.compile(r"\d+|[A-Za-z]+(?:-[A-Za-z]+)?")

# 1. hyphenated compounds "two-day" whose suffix is lost
lost = collections.Counter()
lost_examples = []
for copy in copies:
    original = items[copy["renumbered_from"]]
    texts = [(original["problem"], copy["problem"])] + [(o["question"], c["question"]) for o, c in zip(original["follow_ups"], copy["follow_ups"])]
    for before, after in texts:
        for token in token_pattern.findall(before):
            parts = token.lower().split("-")
            if len(parts) == 2 and (parts[0] in NUMBER_WORDS) != (parts[1] in NUMBER_WORDS) or (len(parts) == 2 and parts[0] in NUMBER_WORDS and parts[1] in NUMBER_WORDS and not (parts[0] in TENS and parts[1] in UNITS and UNITS[parts[1]] < 10)):
                # compound with a non-number half ("two-day", "half-dozen")
                suffix = parts[1] if parts[0] in NUMBER_WORDS else parts[0]
                if token not in after and suffix not in after.lower():
                    lost[token.lower()] += 1
                    if len(lost_examples) < 8:
                        lost_examples.append((copy["id"], before, after))
print("copies where a hyphenated compound lost its non-number half:", sum(lost.values()), lost.most_common(25))
for e in lost_examples:
    print("  ", e)

# 2. digit ordinals / times / decimals in copies
patterns = {
    "ordinal digits (3rd, 2nd)": re.compile(r"\b\d+(st|nd|rd|th)\b"),
    "o'clock / am / pm": re.compile(r"\b\d+\s*(o'clock|a\.?m\.?|p\.?m\.?)\b", re.I),
    "clock time h:mm": re.compile(r"\b\d+:\d+\b"),
    "decimal": re.compile(r"\b\d+\.\d+\b"),
}
for name, pattern in patterns.items():
    originals_hit = sum(1 for item in items.values() if pattern.search(item["problem"] + " " + " ".join(f["question"] for f in item["follow_ups"])))
    copies_hit = [c for c in copies if pattern.search(c["problem"] + " " + " ".join(f["question"] for f in c["follow_ups"]))]
    print(f"{name}: originals {originals_hit}, copies {len(copies_hit)}")
    for c in copies_hit[:3]:
        o = items[c["renumbered_from"]]
        m = pattern.search(c["problem"] + " " + " ".join(f["question"] for f in c["follow_ups"]))
        print("     copy:", m.group(0), "|", c["problem"][:220])
        print("     orig:", o["problem"][:220])

# 3. bad ordinal suffix after renumbering (e.g. "5rd", "21th"?)
def bad_suffix(match):
    number, suffix = int(match.group(1)), match.group(2).lower()
    if number % 100 in (11, 12, 13):
        right = "th"
    else:
        right = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return suffix != right
bad = [c for c in copies if any(bad_suffix(m) for m in re.finditer(r"\b(\d+)(st|nd|rd|th)\b", c["problem"] + " " + " ".join(f["question"] for f in c["follow_ups"])))]
print("copies with a wrong ordinal suffix:", len(bad))
for c in bad[:4]:
    print("   ", c["id"], c["problem"][:200])

# 4. world facts renumbered
facts = re.compile(r"\b(\d+|[a-z]+) (legs|wheels|sides|wings|days in a week|months|players on a team|eyes|arms|corners|faces)\b", re.I)
changed = collections.Counter()
fact_examples = []
for c in copies:
    o = items[c["renumbered_from"]]
    for mo, mc in zip(facts.finditer(o["problem"]), facts.finditer(c["problem"])):
        if mo.group(1).lower() != mc.group(1).lower():
            changed[mo.group(2).lower()] += 1
            if len(fact_examples) < 10:
                fact_examples.append((c["id"], mo.group(0), "->", mc.group(0), c["problem"][:200]))
print("copies changing a '<n> legs/wheels/sides/...' phrase:", dict(changed))
for e in fact_examples:
    print("   ", e)
