"""Math v1 (part 1) students: the same test problems as written (template order) and with the
question sentence moved to the front (same words, new order). Scored against Gemma's answer, as the
thesis's task accuracy is; the most-common-answer baseline is printed next to it."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, r"D:\preference_destillation")
from library import RunStore
from library.domain import get_domain
from library.model.student import StudentModel
from library.shared.batch_files import batch_paths, read_batch_texts

BATCHES = Path(r"D:\preference_destillation\batches\math_word_problem\google_gemma-3-4b-it")
RUNS = ["e01-math-pure-kl-s1", "e01-math-pure-ce-s1", "e01-math-ce-to-kl-s1"]
PROBLEMS = 200
ANSWER = re.compile(r"Solution:\s*([^;\n]+);")


def test_records():
    """The run's test set is the fixed tail of the batch directory (last 1,024 records)."""
    records = []
    for path in reversed(batch_paths(BATCHES)):
        records = read_batch_texts(path) + records
        if len(records) >= 1024:
            return records[-1024:]
    return records


def question_first(sentence):
    match = re.match(r'Problem: "(.*)"(\n.*)', sentence, flags=re.S)
    if not match:
        return None
    story, scaffold = match.groups()
    parts = re.split(r"(?<=[.?!])\s+", story.strip())
    if len(parts) < 2 or not parts[-1].endswith("?"):
        return None
    return f'Problem: "{" ".join([parts[-1]] + parts[:-1])}"{scaffold}'


def answer_of(text):
    found = ANSWER.search(text)
    return found.group(1).strip() if found else None


items = []
for record in test_records():
    reordered = question_first(record["sentence"])
    teacher = answer_of(record["model_response"])
    if reordered and teacher:
        items.append((record["sentence"], reordered, teacher))
    if len(items) == PROBLEMS:
        break
majority = Counter(teacher for _, _, teacher in items).most_common(1)[0]
print(f"{len(items)} test problems; always answering {majority[0]!r} gets {majority[1] / len(items):.3f}")
print("example:", items[0][0].split("\n")[0], "->", items[0][1].split("\n")[0])

store = RunStore()
for run_name in RUNS:
    run = store.get(run_name)
    domain = get_domain(run.domain)
    _, checkpoint = store.resolve_checkpoint(run.name, "latest")
    student = StudentModel.from_run(run, checkpoint, domain=domain)
    scores = {"as written": 0, "question first": 0}
    for original, reordered, teacher in items:
        for label, text in (("as written", original), ("question first", reordered)):
            reply = student.generate(text, max_new_tokens=domain.max_steps).text
            scores[label] += answer_of(reply) == teacher
    print(f"{run_name}: as written {scores['as written'] / len(items):.3f}, "
          f"question first {scores['question first'] / len(items):.3f}")
