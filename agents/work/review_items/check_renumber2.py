import json, sys, re, collections
sys.path.insert(0, r"D:\preference_destillation")
from library.domain.math_tutor.items import text_numbers

ITEMS = r"D:\preference_destillation\library\domain\math_tutor\items\items.jsonl"
COPIES = r"D:\preference_destillation\library\domain\math_tutor\items\renumbered.jsonl"
items = {json.loads(l)["id"]: json.loads(l) for l in open(ITEMS, encoding="utf-8")}
copies = [json.loads(l) for l in open(COPIES, encoding="utf-8")]


def all_text(r):
    return r["problem"] + " || " + " || ".join(f["question"] for f in r["follow_ups"])


# denominations
REAL = {1, 2, 5, 10, 20, 50, 100}
note = re.compile(r"\$(\d+) (bill|note)s?\b|\b(\d+)-dollar (bill|note)s?\b", re.I)
bad_notes = [c for c in copies if any(int(m.group(1) or m.group(3)) not in REAL for m in note.finditer(all_text(c)))]
orig_bad = [i for i in items.values() if any(int(m.group(1) or m.group(3)) not in REAL for m in note.finditer(all_text(i)))]
print("copies with a non-existent banknote ($N bill/note):", len(bad_notes), "| originals:", len(orig_bad), collections.Counter(c["split"] for c in bad_notes))
for c in bad_notes[:3]:
    print("   ", c["id"], [m.group(0) for m in note.finditer(all_text(c))], "| orig:", [m.group(0) for m in note.finditer(all_text(items[c['renumbered_from']]))])

# clock hours above 12
clock = re.compile(r"\b(\d+)\s*(o'clock|a\.m\.|p\.m\.|am|pm)\b", re.I)
bad_clock = [c for c in copies if any(int(m.group(1)) > 12 or int(m.group(1)) == 0 for m in clock.finditer(all_text(c)))]
orig_clock = [i for i in items.values() if any(int(m.group(1)) > 12 for m in clock.finditer(all_text(i)))]
print("copies with an impossible clock hour (>12 o'clock/am/pm):", len(bad_clock), "| originals:", len(orig_clock))
for c in bad_clock[:4]:
    print("   ", c["id"], [m.group(0) for m in clock.finditer(all_text(c))], c["problem"][:160])

# elapsed-time problems where the end is before the start ("from 9 am to 4 pm" fine, but "from 7 pm to 3 pm")
span = re.compile(r"\bfrom (\d+)\s*(o'clock|am|pm)? (?:to|until|till) (\d+)\s*(o'clock|am|pm)?", re.I)
inverted = []
for c in copies:
    for m in span.finditer(all_text(c)):
        start, end = int(m.group(1)), int(m.group(3))
        if (m.group(2) or "").lower() == (m.group(4) or "").lower() and end <= start:
            inverted.append((c["id"], m.group(0)))
print("copies with 'from X to Y' times where Y <= X (same suffix):", len(inverted), inverted[:4])

# facts: "<animal/shape/vehicle> ... N legs/sides/wheels"
FACTS = [
    (re.compile(r"\b(spiders?|octopus(?:es)?)\b[^.?!]*?\b(\d+) (legs|arms)\b", re.I), 8),
    (re.compile(r"\b(insects?|ants?|bees?|beetles?|ladybirds?|ladybugs?)\b[^.?!]*?\b(\d+) legs\b", re.I), 6),
    (re.compile(r"\b(dogs?|cats?|horses?|cows?|sheep|goats?|pigs?|tables?|chairs?)\b[^.?!]*?\b(\d+) legs\b", re.I), 4),
    (re.compile(r"\b(chickens?|hens?|ducks?|birds?|geese|goose|ostriche?s?|people|children)\b[^.?!]*?\b(\d+) legs\b", re.I), 2),
    (re.compile(r"\b(triangles?)\b[^.?!]*?\b(\d+) (sides|corners)\b", re.I), 3),
    (re.compile(r"\b(squares?|rectangles?)\b[^.?!]*?\b(\d+) (sides|corners)\b", re.I), 4),
    (re.compile(r"\b(hexagons?)\b[^.?!]*?\b(\d+) (sides|corners)\b", re.I), 6),
    (re.compile(r"\b(pentagons?)\b[^.?!]*?\b(\d+) (sides|corners)\b", re.I), 5),
    (re.compile(r"\b(cubes?|dice|die)\b[^.?!]*?\b(\d+) (faces|sides)\b", re.I), 6),
    (re.compile(r"\b(cars?|skateboards?|board|wagons?|vans?)\b[^.?!]*?\b(\d+) wheels\b", re.I), 4),
    (re.compile(r"\b(bikes?|bicycles?|motorbikes?|scooters?)\b[^.?!]*?\b(\d+) wheels\b", re.I), 2),
    (re.compile(r"\b(tricycles?|trikes?)\b[^.?!]*?\b(\d+) wheels\b", re.I), 3),
]
wrong_fact = collections.Counter()
fact_examples = []
for c in copies:
    o = items[c["renumbered_from"]]
    for pattern, truth in FACTS:
        mo, mc = pattern.search(o["problem"]), pattern.search(c["problem"])
        if mo and mc and int(mo.group(2)) == truth and int(mc.group(2)) != truth:
            wrong_fact[c["split"]] += 1
            if len(fact_examples) < 6:
                fact_examples.append((c["id"], mo.group(0)[-40:], "->", mc.group(0)[-40:]))
            break
print("copies turning a true world fact (spider 8 legs, square 4 sides, bike 2 wheels...) into a false one:", sum(wrong_fact.values()), dict(wrong_fact))
for e in fact_examples:
    print("   ", e)

# prose number words changed although they are not a quantity the code uses with that meaning:
# items where a value is written both as a digit and as a word -> both change
both = collections.Counter()
both_examples = []
word_token = re.compile(r"\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b", re.I)
for c in copies:
    o = items[c["renumbered_from"]]
    for before, after in [(o["problem"], c["problem"])] + [(a["question"], b["question"]) for a, b in zip(o["follow_ups"], c["follow_ups"])]:
        words_before = word_token.findall(before)
        if not words_before:
            continue
        digits_before = set(int(d) for d in re.findall(r"\d+", all_text(o)))
        for w in words_before:
            value = text_numbers(w)[0]
            if value in digits_before and w.lower() not in after.lower():
                both[w.lower()] += 1
                if len(both_examples) < 6:
                    both_examples.append((c["id"], before[:200], after[:200]))
print("prose number word changed because the same value is also a digit elsewhere:", sum(both.values()), both.most_common(8))
for e in both_examples:
    print("   ", e)
