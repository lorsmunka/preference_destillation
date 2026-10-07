import sys, random, re, collections
sys.path.insert(0, r"D:\preference_destillation")
from library.domain.code_recall.corpus import ProgramBuilder, make_question, make_record, all_possible_answers
rng = random.Random(0)
answers = set(all_possible_answers())
by_kind = collections.Counter(); wrong = collections.Counter(); outside = 0; examples = []
for _ in range(5000):
    record = make_record(rng)
    kind, answer = record["attribute"], record["answer"]
    by_kind[kind] += 1
    if answer not in answers: outside += 1
    lines = [l.split(": ", 1)[1] if ": " in l else "" for l in record["text"].split("\n\nQuestion:")[0].split("\n")]
    if kind == "last_assignment":
        name = re.search(r"`([^`]+)`", record["text"].split("Question:")[1]).group(1)
        actual = [i + 1 for i, l in enumerate(lines) if re.match(rf"\s*{re.escape(name)}\s*(=(?!=)|\+=|-=)", l) or re.match(rf"\s*for {re.escape(name)} in", l)]
        if actual and str(actual[-1]) != answer:
            wrong[kind] += 1
            if len(examples) < 1: examples.append((answer, actual, record["text"][:900]))
    if kind == "initial_value":
        name = re.search(r"`([^`]+)`", record["text"].split("Question:")[1]).group(1)
        first = [l for l in lines if re.match(rf"\s*{re.escape(name)}\s*=(?!=)", l)]
        if first and first[0].split("=", 1)[1].strip() != answer:
            wrong[kind] += 1
print("by kind", dict(by_kind)); print("wrong ground truth", dict(wrong)); print("answers outside vocab list", outside)
for e in examples: print("recorded", e[0], "actual assignment lines", e[1]); print(e[2])
