"""The best math tutor student (math-agent-kl-m-s2) solving first-turn problems it never saw, scored
against the true answer (the item's solve() / the dataset's answer):
  test       held-out families, problem shapes seen in training
  heldout    the problem shapes kept out of training entirely (add>mul, sub>div, sub>add)
  svamp      735 human-written problems (other writers; some words outside the input vocabulary)
Accuracy is split by the number of solution steps, so like is compared with like."""
import json
import random
import re
import sys
from collections import defaultdict

sys.path.insert(0, r"D:\preference_destillation")
from library.agent.runner import Request
from library.domain.math_tutor.conversation import Conversation
from library.domain.math_tutor.demo import load_student
from library.domain.math_tutor.harness import SOLUTION_PREFIX

RUN = "math-agent-kl-m-s2"
PER_SET = 300
ITEMS = r"D:\preference_destillation\library\domain\math_tutor\items\items.jsonl"
SVAMP = r"D:\preference_destillation\library\domain\math_tutor\benchmarks\real_test.jsonl"
ANSWER = re.compile(r"Answer:\s*\$?(-?[\d,]*\.?\d+)")


def steps_of(shape):
    return shape.count(">") + 1


def read(path):
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file]


items = read(ITEMS)
sets = {
    "test": [(item["problem"], item["answers"][0], steps_of(item["problem_shape"])) for item in items if item["split"] == "test"],
    "heldout": [(item["problem"], item["answers"][0], steps_of(item["problem_shape"])) for item in items if item["split"] == "heldout_shape"],
    "svamp": [(item["problem"], item["answer"], item["steps"]) for item in read(SVAMP)],
}
generator = random.Random(0)
sets = {name: generator.sample(problems, min(PER_SET, len(problems))) for name, problems in sets.items()}

run, model = load_student(RUN)
print(f"student {run.name}")
for name, problems in sets.items():
    right = defaultdict(int)
    total = defaultdict(int)
    for problem, truth, steps in problems:
        request = Request("solution", Conversation().solve_context(problem), SOLUTION_PREFIX, stop=["</solution>"])
        reply = model.generate([request])[0]
        found = ANSWER.search(reply)
        correct = False
        if found:
            try:
                correct = abs(float(found.group(1).replace(",", "")) - float(truth)) < 1e-6
            except ValueError:
                correct = False
        bucket = "1 step" if steps == 1 else "2 steps" if steps == 2 else "3+ steps"
        for key in (bucket, "all"):
            total[key] += 1
            right[key] += correct
    summary = ", ".join(f"{key} {right[key] / total[key]:.3f} (n={total[key]})" for key in sorted(total))
    print(f"{name:8s} {summary}")
