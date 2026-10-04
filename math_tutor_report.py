"""The math tutor build, step by step, as one local HTML page: where the problems come
from, how many survive each stage, how they are stored, and how pruning works.

    python math_tutor_report.py      # writes thesis_work/math_tutor_report.html

Local file only, nothing is uploaded. Reads the logs and data the build scripts wrote
under library/domain/math_tutor/ (benchmarks/, items/, teacher_check.json); every block
shows what is there and says so when a stage has not run yet.
"""

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from library.domain.math_tutor import EXAMPLE_RESPONSES, PROMPTS
from library.domain.math_tutor.corpus import turn_text
from library.domain.math_tutor.items import SLICES, validate
from library.domain.math_tutor.similarity import collect_name_words, normalise, sequence_ratio, word_trigrams, jaccard
from library.domain.math_tutor.validator_cases import CASES

PACKAGE = Path("library") / "domain" / "math_tutor"
OUTPUT = Path("thesis_work") / "math_tutor_report.html"
TEMPLATE = Path("library") / "tooling" / "templates" / "math_tutor_report.html"


def read_json(path):
    return json.load(open(path, encoding="utf-8")) if path.exists() else None


def read_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()] if path.exists() else []


def source_kind(item):
    return str(item.get("source", "?")).split(":")[0]


def human_sets():
    log = read_json(PACKAGE / "benchmarks" / "build_log.json")
    if not log:
        return None
    test = read_jsonl(PACKAGE / "benchmarks" / "real_test.jsonl")
    seeds = read_jsonl(PACKAGE / "benchmarks" / "seeds.jsonl")
    dropped = read_jsonl(PACKAGE / "benchmarks" / "dropped.jsonl")
    examples = {}
    for drop in dropped:
        examples.setdefault(drop["reason"], drop)
    return {"log": log, "test_example": test[0] if test else None,
            "seed_examples": [seeds[0], next((s for s in seeds if s["set"] == "mawps"), None),
                              next((s for s in seeds if s["set"] == "harvest"), None)],
            "drop_examples": list(examples.values())}


def writing():
    """Raw files: who wrote what, how much of it is valid, and how much survives pruning."""
    kept = {item["id"] for item in read_jsonl(PACKAGE / "items" / "items.jsonl")}
    cells = defaultdict(lambda: {"written": 0, "valid": 0, "kept": 0})
    files = []
    for path in sorted((PACKAGE / "items" / "raw").glob("*.jsonl")):
        written = valid = 0
        for item in read_jsonl(path):
            validated, _ = validate(item)
            written += 1
            valid += validated is not None
            source = source_kind(item)
            if source == "seeded" and item.get("slice") not in SLICES:
                continue
            cell = cells[(item.get("slice", "?"), source)]
            cell["written"] += 1
            cell["valid"] += validated is not None
            cell["kept"] += item.get("id") in kept
        files.append({"file": path.name, "written": written, "valid": valid})
    matrix = [{"slice": slice_name, "source": source, **counts} for (slice_name, source), counts in sorted(cells.items())]
    return {"files": files, "matrix": matrix, "writing_log": read_json(PACKAGE / "items" / "writing_log.json")}


def pruning():
    log = read_json(PACKAGE / "items" / "build_log.json")
    if not log:
        return None
    dropped = read_jsonl(PACKAGE / "items" / "dropped.jsonl")
    examples = {}
    for drop in dropped:
        examples.setdefault(drop["reason"], drop)
    items = read_jsonl(PACKAGE / "items" / "items.jsonl")
    shapes_by_slice = defaultdict(Counter)
    for item in items:
        shapes_by_slice[item["slice"]][item["shape"]] += 1
    return {"log": log, "drop_examples": list(examples.values()),
            "shapes_by_slice": {name: counter.most_common(8) for name, counter in sorted(shapes_by_slice.items())}}


def similarity_demo():
    original = "A baker made 36 muffins and sold 19 of them in the morning. How many muffins are left?"
    pairs = [
        ("one word changed", "A baker made 48 muffins and sold 21 of them in the evening. How many muffins are left?"),
        ("two words changed", "A baker baked 48 cookies and sold 21 of them in the morning. How many cookies are left?"),
        ("same frame, new nouns", "A farmer picked 48 apples and sold 21 of them in the morning. How many apples are left?"),
        ("reordered", "In the morning a baker sold 21 muffins out of the 48 she made. How many are left?"),
        ("reworded", "There were 48 muffins at the bakery. By lunch 21 had been sold. How many muffins remain?"),
        ("another problem", "Omar had 52 marbles. After a game he had 37 marbles. How many marbles did he lose?"),
    ]
    names = set()
    base = normalise(original, names)
    rows = []
    for label, text in pairs:
        words = normalise(text, names)
        rows.append({"label": label, "text": text, "sequence": round(sequence_ratio(base, words), 2),
                     "trigram": round(jaccard(word_trigrams(base), word_trigrams(words)), 2)})
    return {"original": original, "normalised": " ".join(base), "rows": rows}


def validator_cases():
    rows = []
    for name, item, expected in CASES:
        validated, reason = validate(item)
        rows.append({"case": name, "result": reason or f"valid, answers {validated['answers']}"})
    return rows


def renumber_example():
    items = {item["id"]: item for item in read_jsonl(PACKAGE / "items" / "items.jsonl")}
    copies = read_jsonl(PACKAGE / "items" / "renumbered.jsonl")
    by_original = defaultdict(list)
    for copy in copies:
        by_original[copy["renumbered_from"]].append(copy)
    for original_id, its_copies in by_original.items():
        original = items.get(original_id)
        if original and original["follow_ups"] and len(its_copies) >= 3:
            return {"original": original, "copies": its_copies[:3],
                    "log": read_json(PACKAGE / "items" / "renumber_log.json")}
    return {"log": read_json(PACKAGE / "items" / "renumber_log.json")}


def storage():
    rows = []
    for path in sorted(PACKAGE.rglob("*")):
        if path.is_file() and path.suffix in (".jsonl", ".json", ".md") and "__pycache__" not in path.parts:
            lines = sum(1 for _ in open(path, encoding="utf-8")) if path.suffix == ".jsonl" else None
            rows.append({"path": str(path).replace("\\", "/"), "kb": round(path.stat().st_size / 1024, 1), "lines": lines})
    items = read_jsonl(PACKAGE / "items" / "items.jsonl")
    example = next((item for item in items if len(item["follow_ups"]) >= 2), items[0] if items else None)
    turns = [turn_text(example, turn) for turn in range(1 + len(example["follow_ups"]))] if example else []
    return {"files": rows, "example": example, "turns": turns}


def templateness():
    path = Path("experiments") / "results" / "math_templateness.csv"
    if not path.exists():
        return []
    import csv
    return list(csv.DictReader(open(path, encoding="utf-8")))


def build():
    data = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "slices": SLICES,
        "human": human_sets(),
        "writing": writing(),
        "pruning": pruning(),
        "similarity": similarity_demo(),
        "validator_cases": validator_cases(),
        "renumber": renumber_example(),
        "storage": storage(),
        "teacher": read_json(PACKAGE / "teacher_check.json"),
        "templateness": templateness(),
        "prompts": PROMPTS,
        "example_responses": EXAMPLE_RESPONSES,
    }
    template = open(TEMPLATE, encoding="utf-8").read()
    page = template.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    OUTPUT.parent.mkdir(exist_ok=True)
    open(OUTPUT, "w", encoding="utf-8").write(page)
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    build()
