"""Human-written problems for the math tutor. SVAMP is the held-out test and is never
trained on or shown to a writer; ASDiv and MAWPS are the seeds the training set grows from.

    python -m library.domain.math_tutor.benchmarks

Downloads the three sets into datasets/math/ (skipped when already there), keeps the
problems inside the domain's range, turns each equation into a solve() function and keeps
it only if running it reproduces the dataset's own answer. Seeds that are too close to any
SVAMP problem, or to another seed, are dropped. Writes benchmarks/real_test.jsonl,
seeds.jsonl, dropped.jsonl (every drop with its reason) and build_log.json (the counts).
"""

import csv
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

from library.shared.config import PROJECT_ROOT
from .functions import NotAllowed, equation_to_solve, execute, trace_equation
from .harvest import HARVEST_DIR, check_harvested
from .prune import reason_category
from .similarity import TrigramIndex, WordSequenceIndex, collect_name_words, normalise, word_trigrams

DOWNLOAD_DIR = PROJECT_ROOT / "datasets" / "math"
OUTPUT_DIR = Path(__file__).resolve().parent / "benchmarks"

MAX_NUMBER = 99
MAX_ANSWER = 999
MAX_STEPS = 3
MAX_GRADE = 4
NEAR_DUPLICATE = 0.8       # word-sequence ratio
TOO_CLOSE_TO_TEST = 0.5    # word-trigram Jaccard, on top of NEAR_DUPLICATE against SVAMP

SOURCES = {
    "svamp": {
        "role": "test",
        "files": {"SVAMP.json": "https://raw.githubusercontent.com/arkilpatel/SVAMP/main/SVAMP.json"},
        "licence": "MIT",
        "citation": "Patel, Bhattamishra, Goyal (2021). Are NLP Models really able to Solve Simple Math Word Problems? NAACL.",
    },
    "asdiv": {
        "role": "seeds",
        "files": {"ASDiv.xml": "https://raw.githubusercontent.com/chaochun/nlu-asdiv-dataset/master/dataset/ASDiv.xml"},
        "licence": "CC BY-NC 4.0",
        "citation": "Miao, Liang, Su (2020). A Diverse Corpus for Evaluating and Developing English Math Word Problem Solvers. ACL.",
    },
    "mawps": {
        "role": "seeds",
        "files": {
            "train.csv": "https://raw.githubusercontent.com/arkilpatel/SVAMP/main/data/cv_mawps/fold0/train.csv",
            "dev.csv": "https://raw.githubusercontent.com/arkilpatel/SVAMP/main/data/cv_mawps/fold0/dev.csv",
        },
        "licence": "MIT (the copy in the SVAMP repository; fold0 train + dev = all 1,921)",
        "citation": "Koncel-Kedziorski, Roy, Amini, Kushman, Hajishirzi (2016). MAWPS: A Math Word Problem Repository. NAACL.",
    },
}


def download(set_name: str) -> Path:
    folder = DOWNLOAD_DIR / set_name
    folder.mkdir(parents=True, exist_ok=True)
    for filename, url in SOURCES[set_name]["files"].items():
        if not (folder / filename).exists():
            print(f"  downloading {set_name}/{filename}")
            urllib.request.urlretrieve(url, folder / filename)
    return folder


# --- readers: each returns raw records {id, set, problem, equation, dataset_answer, ...} ---

def read_svamp(folder: Path) -> List[Dict]:
    records = []
    for item in json.load(open(folder / "SVAMP.json", encoding="utf-8")):
        body, question = item["Body"].strip(), item["Question"].strip()
        if body[-1] in ".?!":
            problem = f"{body} {question}"
        else:  # the body is an "If ..." clause that runs into the question
            problem = f"{body}, {question[0].lower()}{question[1:]}"
        records.append({
            "id": f"svamp-{item['ID']}", "set": "svamp", "type": item["Type"],
            "problem": re.sub(r"\$ (\d)", r"$\1", problem),
            "equation": item["Equation"], "dataset_answer": float(item["Answer"]),
        })
    return records


def read_asdiv(folder: Path) -> List[Dict]:
    records = []
    for problem in ET.parse(folder / "ASDiv.xml").getroot().iter("Problem"):
        formula = problem.findtext("Formula") or ""
        single_equation = formula.count("=") == 1 and not re.search(r"[;<>]", formula)
        answer_match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*(\(|$)", problem.findtext("Answer") or "")
        records.append({
            "id": f"asdiv-{problem.get('ID')}", "set": "asdiv",
            "grade": int(problem.get("Grade")), "type": problem.findtext("Solution-Type"),
            "problem": f"{problem.findtext('Body').strip()} {problem.findtext('Question').strip()}",
            "equation": re.sub(r"(?<=\d)\s*x\s*(?=\d)", "*", formula.split("=")[0]) if single_equation else None,
            "dataset_answer": float(answer_match.group(1)) if answer_match else None,
        })
    return records


def prefix_to_infix(tokens: List[str], numbers: List[str]) -> str:
    token = tokens.pop(0)
    if token in "+-*/":
        left = prefix_to_infix(tokens, numbers)
        right = prefix_to_infix(tokens, numbers)
        return f"({left} {token} {right})"
    if token.startswith("number"):
        return numbers[int(token[len("number"):])]
    return token


def read_mawps(folder: Path) -> List[Dict]:
    records = []
    rows = list(csv.DictReader(open(folder / "train.csv", encoding="utf-8")))
    rows += list(csv.DictReader(open(folder / "dev.csv", encoding="utf-8")))
    for index, row in enumerate(rows, start=1):
        numbers = [str(int(float(n))) if float(n).is_integer() else n for n in row["Numbers"].split()]
        text = re.sub(r"number(\d+)", lambda match: numbers[int(match.group(1))], row["Question"])
        text = re.sub(r" ([.,?!;:]|'s|n't|')", r"\1", text)
        text = re.sub(r"\$ (\d)", r"$\1", text).strip()
        records.append({
            "id": f"mawps-{index:04d}", "set": "mawps", "problem": text,
            "equation": prefix_to_infix(row["Equation"].split(), numbers),
            "dataset_answer": float(row["Answer"]),
        })
    return records


# --- the range filter: one reason per dropped problem, the first rule it breaks ---------

def check(record: Dict) -> Optional[str]:
    """None if the problem is inside the domain's range, else the reason it is not."""
    if "�" in record["problem"]:
        return "damaged characters"
    if record["equation"] is None:
        return "no single arithmetic equation"
    if record.get("grade", 0) > MAX_GRADE:
        return "grade 5-6"
    try:
        numbers, operations = trace_equation(record["equation"])
    except NotAllowed as error:
        return str(error)
    values = numbers + [value for _, value in operations]
    if not operations:
        return "no operation"
    if len(operations) > MAX_STEPS:
        return "more than 3 steps"
    if any(value.denominator != 1 or value < 0 for value in values):
        return "not whole numbers all the way"
    if max(numbers) > MAX_NUMBER:
        return "number above 99"
    if operations[-1][1] > MAX_ANSWER:
        return "answer above 999"
    if record["dataset_answer"] is None or operations[-1][1] != record["dataset_answer"]:
        return "equation disagrees with the dataset's answer"
    return None


def to_item(record: Dict) -> Dict:
    solve = equation_to_solve(record["equation"])
    execution = execute(solve)
    item = {"id": record["id"], "set": record["set"], "problem": record["problem"], "solve": solve,
            "answer": execution.result["answer"], "shape": ">".join(execution.operations),
            "steps": len(execution.operations)}
    for optional in ("grade", "type"):
        if optional in record:
            item[optional] = record[optional]
    assert item["answer"] == record["dataset_answer"], record["id"]
    return item


def write_jsonl(path: Path, records: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_benchmarks() -> None:
    readers = {"svamp": read_svamp, "asdiv": read_asdiv, "mawps": read_mawps}
    raw = {name: reader(download(name)) for name, reader in readers.items()}

    log = {"built": date.today().isoformat(), "sources": SOURCES, "sets": {},
           "rules": {"max_number": MAX_NUMBER, "max_answer": MAX_ANSWER, "max_steps": MAX_STEPS,
                     "max_grade": MAX_GRADE, "near_duplicate_sequence_ratio": NEAR_DUPLICATE,
                     "too_close_to_test_trigram_jaccard": TOO_CLOSE_TO_TEST}}
    dropped, kept = [], {}
    for name, records in raw.items():
        reasons = Counter()
        kept[name] = []
        for record in records:
            reason = check(record)
            if reason:
                reasons[reason] += 1
                dropped.append({"id": record["id"], "set": name, "stage": "range", "reason": reason,
                                "problem": record["problem"]})
            else:
                kept[name].append(to_item(record))
        log["sets"][name] = {"downloaded": len(records), "dropped_by_reason": dict(reasons.most_common()),
                             "in_range": len(kept[name])}

    # Harvested problems (copied from openly licensed sources by the harvest agents) join
    # the seed candidates after ASDiv and MAWPS, under the same dedup and SVAMP rules.
    kept["harvest"], harvest_reasons, harvest_count = [], Counter(), 0
    for path in sorted(HARVEST_DIR.glob("*.jsonl")):
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            harvest_count += 1
            record = json.loads(line)
            reason = check_harvested(record)
            if reason:
                harvest_reasons[reason_category(reason)] += 1
                dropped.append({"id": record.get("id"), "set": "harvest", "stage": "range", "reason": reason,
                                "problem": record.get("problem")})
                continue
            execution = execute(record["solve"], allow_remainders=True)
            kept["harvest"].append({
                "id": record["id"], "set": "harvest", "problem": record["problem"], "solve": record["solve"],
                "answer": execution.result["answer"], "shape": ">".join(execution.operations),
                "steps": len(execution.operations), "grade": record.get("grade"),
                "site": record["site"], "url": record["url"], "licence": record["licence"]})
    log["sets"]["harvest"] = {"downloaded": harvest_count, "dropped_by_reason": dict(harvest_reasons.most_common()),
                              "in_range": len(kept["harvest"]),
                              "by_site": dict(Counter(item["site"] for item in kept["harvest"]).most_common())}

    # Seeds: drop anything close to any SVAMP problem (all 1,000, kept or not), then
    # anything that repeats an earlier seed. ASDiv goes first, so a MAWPS copy is the one dropped.
    # Same definitions as the synthetic items (prune.py): exact = same normalised words,
    # near = word-sequence ratio > 0.8, too close to SVAMP = that or trigram Jaccard > 0.5.
    name_words = collect_name_words(record["problem"] for records in raw.values() for record in records)
    candidates = kept["asdiv"] + kept["mawps"] + kept["harvest"]
    words_by_id = {record["id"]: normalise(record["problem"], name_words) for record in raw["svamp"] + candidates}
    test_trigrams = TrigramIndex()
    test_sequences = WordSequenceIndex(NEAR_DUPLICATE, words_by_id.values())
    seed_sequences = WordSequenceIndex(NEAR_DUPLICATE, words_by_id.values())
    for record in raw["svamp"]:
        test_trigrams.add(record["id"], word_trigrams(words_by_id[record["id"]]))
        test_sequences.add(record["id"], words_by_id[record["id"]])
    seeds, seed_reasons, first_by_words = [], Counter(), {}
    for item in candidates:
        words = words_by_id[item["id"]]
        trigram_id, trigram_score = test_trigrams.most_similar(word_trigrams(words))
        test_id, test_score = test_sequences.most_similar(words)
        seed_id, seed_score = seed_sequences.most_similar(words)
        if test_score > NEAR_DUPLICATE:
            reason, detail = "too close to a SVAMP problem", f"{test_id} sequence={test_score:.2f}"
        elif trigram_score > TOO_CLOSE_TO_TEST:
            reason, detail = "too close to a SVAMP problem", f"{trigram_id} trigram J={trigram_score:.2f}"
        elif tuple(words) in first_by_words:
            reason, detail = "exact duplicate", first_by_words[tuple(words)]
        elif seed_score > NEAR_DUPLICATE:
            reason, detail = "near-duplicate", f"{seed_id} sequence={seed_score:.2f}"
        else:
            first_by_words[tuple(words)] = item["id"]
            seed_sequences.add(item["id"], words)
            seeds.append(item)
            continue
        seed_reasons[reason] += 1
        dropped.append({"id": item["id"], "set": item["set"], "stage": "dedup", "reason": reason,
                        "detail": detail, "problem": item["problem"]})

    test = kept["svamp"]
    log["test"] = {"set": "svamp", "count": len(test),
                   "by_steps": dict(sorted(Counter(item["steps"] for item in test).items())),
                   "by_shape": dict(Counter(item["shape"] for item in test).most_common())}
    log["seeds"] = {"candidates": len(candidates),
                    "dropped_by_reason": dict(seed_reasons.most_common()), "count": len(seeds),
                    "by_set": dict(Counter(item["set"] for item in seeds)),
                    "by_steps": dict(sorted(Counter(item["steps"] for item in seeds).items())),
                    "by_shape": dict(Counter(item["shape"] for item in seeds).most_common())}

    OUTPUT_DIR.mkdir(exist_ok=True)
    write_jsonl(OUTPUT_DIR / "real_test.jsonl", test)
    write_jsonl(OUTPUT_DIR / "seeds.jsonl", seeds)
    write_jsonl(OUTPUT_DIR / "dropped.jsonl", dropped)
    with open(OUTPUT_DIR / "build_log.json", "w", encoding="utf-8") as file:
        json.dump(log, file, indent=2, ensure_ascii=False)

    for name in raw:
        entry = log["sets"][name]
        print(f"{name:6} {entry['downloaded']:5} downloaded -> {entry['in_range']:5} in range   {entry['dropped_by_reason']}")
    print(f"test   {len(test)} SVAMP problems  {log['test']['by_steps']} by steps")
    print(f"seeds  {log['seeds']['candidates']} candidates -> {len(seeds)} kept  {log['seeds']['dropped_by_reason']}")
    print(f"written to {OUTPUT_DIR}")


if __name__ == "__main__":
    build_benchmarks()
