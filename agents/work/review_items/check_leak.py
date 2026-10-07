import json, sys, re, collections, random
sys.path.insert(0, r"D:\preference_destillation")
from library.domain.math_tutor.similarity import collect_name_words, normalise, sequence_ratio, word_trigrams, jaccard, NEAR_DUPLICATE
from library.domain.math_tutor.prune import svamp_problems
from library.domain.math_tutor.items import code_literals

BASE = r"D:\preference_destillation\library\domain\math_tutor\items"
items = [json.loads(l) for l in open(BASE + r"\items.jsonl", encoding="utf-8")]

print(normalise("Mia's dog has 5 bones and Tom's cat has 3.", {"Mia", "Tom"}))

svamp = svamp_problems()
name_words = collect_name_words([i["problem"] for i in items] + [p["problem"] for p in svamp])
possessive = sum(1 for p in svamp if re.search(r"\b([A-Z][a-z]+)'s\b", p["problem"]) and re.search(r"\b([A-Z][a-z]+)'s\b", p["problem"]).group(1) in name_words)
print("SVAMP problems with a possessive name ('Jake's'):", possessive, "of", len(svamp))

# Simulated leak: copy each SVAMP problem, give every name a different name, new numbers.
replacement_names = ["Priya", "Kofi", "Lena", "Mateo", "Aisha", "Bruno"]
assert all(n in name_words for n in replacement_names)
caught_sequence = caught_trigram = 0
missed = []
rng = random.Random(0)
for p in svamp:
    text = p["problem"]
    names = sorted({w for w in re.findall(r"[A-Z][a-z]+", text) if w in name_words})
    mapping = {n: replacement_names[k % len(replacement_names)] for k, n in enumerate(names)}
    leaked = re.sub(r"[A-Z][a-z]+", lambda m: mapping.get(m.group(0), m.group(0)), text)
    leaked = re.sub(r"\d+", lambda m: str(rng.randint(2, 99)), leaked)
    a, b = normalise(text, name_words), normalise(leaked, name_words)
    seq, tri = sequence_ratio(a, b), jaccard(word_trigrams(a), word_trigrams(b))
    if seq > NEAR_DUPLICATE:
        caught_sequence += 1
    elif tri > 0.5:
        caught_trigram += 1
    else:
        missed.append((round(seq, 2), round(tri, 2), text[:150], leaked[:150]))
print(f"name+number-swapped SVAMP copies: {len(svamp)}; caught by sequence {caught_sequence}, by trigram {caught_trigram}, missed {len(missed)}")
for m in missed[:5]:
    print("   ", m)

# same, but the fix: strip possessive before name lookup
def normalise_fixed(text):
    return normalise(re.sub(r"\b([A-Z][a-z]+)'s\b", r"\1", text), name_words)
lower = 0
for p in svamp[:]:
    text = p["problem"]
    names = sorted({w for w in re.findall(r"[A-Z][a-z]+", text) if w in name_words})
    mapping = {n: replacement_names[k % len(replacement_names)] for k, n in enumerate(names)}
    leaked = re.sub(r"[A-Z][a-z]+", lambda m: mapping.get(m.group(0), m.group(0)), text)
    a, b = normalise(text, name_words), normalise(leaked, name_words)
    if a != b:
        lower += 1
print("SVAMP problems whose normalised text changes when only the names change:", lower)
