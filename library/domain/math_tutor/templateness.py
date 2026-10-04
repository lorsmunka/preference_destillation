"""How templated is a set of word problems? Measured the same way on every set: take up to
2,000 problems, normalise them (numbers -> N, names -> NAME), and find each problem's
nearest neighbour inside the sample by word-sequence ratio.

  distinct   share of problems whose normalised text is unique in the sample
  median NN  the typical nearest-neighbour ratio (1.0 = an exact template copy)
  NN > 0.8   share that the pruning would call near-duplicates

    python -m library.domain.math_tutor.templateness

Writes experiments/results/math_templateness.csv. Sets that are not built yet are skipped.
"""

import csv
import json
import random
import statistics
from pathlib import Path
from typing import Dict, List

from library.shared.config import PROJECT_ROOT
from .similarity import collect_name_words, normalise, sequence_ratio

PACKAGE = Path(__file__).resolve().parent
SAMPLE = 2000
SEED = 0
OUTPUT = PROJECT_ROOT / "experiments" / "results" / "math_templateness.csv"


def v1_problems() -> List[str]:
    """v1 corpus records hold the scaffold too; keep only the quoted problem."""
    path = PROJECT_ROOT / "library" / "domain" / "math_word_problem" / "corpus.jsonl"
    if not path.exists():
        return []
    problems = []
    for line in open(path, encoding="utf-8"):
        text = json.loads(line)["text"]
        problems.append(text.split('"')[1] if '"' in text else text)
    return problems


def read_problems(path: Path) -> List[str]:
    return [json.loads(line)["problem"] for line in open(path, encoding="utf-8")] if path.exists() else []


def measure(name: str, problems: List[str]) -> Dict:
    random_generator = random.Random(SEED)
    sample = random_generator.sample(problems, min(SAMPLE, len(problems)))
    names = collect_name_words(sample)
    words = [normalise(problem, names) for problem in sample]
    nearest = []
    for index, first in enumerate(words):
        best = 0.0
        for other_index, second in enumerate(words):
            if other_index != index:
                best = max(best, sequence_ratio(first, second))
                if best == 1.0:
                    break
        nearest.append(best)
    distinct = len({tuple(w) for w in words}) / len(words)
    return {"set": name, "problems": len(problems), "sample": len(sample), "distinct": round(distinct, 3),
            "median_nn": round(statistics.median(nearest), 3),
            "nn_over_0.8": round(sum(score > 0.8 for score in nearest) / len(nearest), 3)}


def main() -> None:
    sets = {
        "v1 templates (math_word_problem corpus)": v1_problems(),
        "SVAMP test (human)": read_problems(PACKAGE / "benchmarks" / "real_test.jsonl"),
        "seeds (human, deduped)": read_problems(PACKAGE / "benchmarks" / "seeds.jsonl"),
        "math_tutor kept items": read_problems(PACKAGE / "items" / "items.jsonl"),
    }
    rows = []
    for name, problems in sets.items():
        if not problems:
            print(f"{name}: not built, skipped")
            continue
        row = measure(name, problems)
        rows.append(row)
        print(f"{name:42} n={row['problems']:7}  distinct {row['distinct']:.3f}  median NN {row['median_nn']:.3f}  NN>0.8 {row['nn_over_0.8']:.3f}")
    with open(OUTPUT, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"written to {OUTPUT}")


if __name__ == "__main__":
    main()
