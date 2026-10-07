import json, sys, collections, random
sys.path.insert(0, r"D:\preference_destillation")
from library.domain.math_tutor.items import validate
from library.domain.math_tutor.prune import HELDOUT_SHAPES

ITEMS = r"D:\preference_destillation\library\domain\math_tutor\items\items.jsonl"
COPIES = r"D:\preference_destillation\library\domain\math_tutor\items\renumbered.jsonl"
items = [json.loads(l) for l in open(ITEMS, encoding="utf-8")]
copies = [json.loads(l) for l in open(COPIES, encoding="utf-8")]
print("items", len(items), "copies", len(copies))

# 1. family -> split consistency
splits_by_family = collections.defaultdict(set)
for record in items + copies:
    splits_by_family[record["family"]].add(record["split"])
mixed = {f: s for f, s in splits_by_family.items() if len(s) > 1}
print("families with mixed splits:", len(mixed), list(mixed.items())[:5])

# copies' family/split match their original
by_id = {item["id"]: item for item in items}
bad = [c["id"] for c in copies if by_id[c["renumbered_from"]]["split"] != c["split"] or by_id[c["renumbered_from"]]["family"] != c["family"]]
print("copies whose split/family differs from original:", len(bad))

# 2. held-out shapes as problem_shape in train/test
for split in ("train", "test"):
    n = sum(1 for r in items + copies if r["split"] == split and r["problem_shape"] in HELDOUT_SHAPES)
    print(f"{split}: problem_shape in heldout:", n)

# 3. held-out shape as a follow-up turn's own shape (single-turn composition) in train
turn_counts = collections.Counter()
examples = collections.defaultdict(list)
for r in items + copies:
    if r["split"] != "train":
        continue
    for index, turn_shape in enumerate(r["shape"].split("|")[1:], start=1):
        if turn_shape in HELDOUT_SHAPES:
            turn_counts[(turn_shape, "copy" if "renumbered_from" in r else "item")] += 1
            if len(examples[turn_shape]) < 2 and "renumbered_from" not in r:
                examples[turn_shape].append((r["id"], r["follow_ups"][index - 1]["question"], r["follow_ups"][index - 1]["solve"]))
print("train follow-up turns whose own shape is held out:", dict(turn_counts))
for shape, ex in examples.items():
    for e in ex:
        print("  ", shape, e)

# 4. contiguous containment of held-out shape inside longer train problem shapes
contain = collections.Counter()
for r in items:
    if r["split"] != "train":
        continue
    ops = r["problem_shape"].split(">")
    for h in HELDOUT_SHAPES:
        hp = h.split(">")
        if r["problem_shape"] != h and any(ops[i:i + 2] == hp for i in range(len(ops) - 1)):
            contain[h] += 1
print("train items (originals) whose problem shape contains a held-out pair:", dict(contain))

# 5. seeded items: same seed in different splits
splits_by_seed = collections.defaultdict(set)
families_by_seed = collections.defaultdict(set)
for r in items:
    src = str(r["source"])
    if src.startswith("seeded:"):
        splits_by_seed[src].add(r["split"])
        families_by_seed[src].add(r["family"])
multi = [s for s, f in families_by_seed.items() if len(f) > 1]
cross = [s for s, sp in splits_by_seed.items() if "test" in sp and "train" in sp]
print("seeds:", len(families_by_seed), "seeds with >1 family:", len(multi), "seeds with families in both train and test:", len(cross))
test_families_from_cross = set()
for r in items:
    if str(r["source"]) in set(cross) and r["split"] == "test":
        test_families_from_cross.add(r["family"])
test_families = {r["family"] for r in items if r["split"] == "test"}
print("test families", len(test_families), "of which seeded with sibling in train:", len(test_families_from_cross))
for s in cross[:3]:
    for r in items:
        if str(r["source"]) == s and "~" not in r["id"]:
            print("   ", r["split"], r["id"], r["problem"][:200])

# 6. revalidate random sample
rng = random.Random(1)
sample = rng.sample(items, 2000) + rng.sample(copies, 4000)
fails, mismatches = 0, 0
for r in sample:
    v, reason = validate(r)
    if v is None:
        fails += 1
        if fails < 5:
            print("FAIL", r["id"], reason)
    elif v["answers"] != r["answers"] or v["shape"] != r["shape"]:
        mismatches += 1
print("revalidated", len(sample), "fails", fails, "answer/shape mismatches", mismatches)
