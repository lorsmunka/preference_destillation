import json, re, collections
total = collections.Counter(); wrong = 0
for line in open(r"D:\preference_destillation\library\domain\code_recall\corpus.jsonl", encoding="utf-8"):
    r = json.loads(line); total[r["attribute"]] += 1
    if r["attribute"] != "last_assignment": continue
    body, q = r["text"].split("\n\nQuestion:")
    name = re.search(r"`([^`]+)`", q).group(1)
    lines = [l.split(": ", 1)[1] if ": " in l else "" for l in body.split("\n")]
    actual = [i + 1 for i, l in enumerate(lines) if re.match(rf"\s*{re.escape(name)}\s*(=(?!=)|\+=|-=)", l) or re.match(rf"\s*for {re.escape(name)} in", l)]
    if actual and str(actual[-1]) != r["answer"]: wrong += 1
print(sum(total.values()), "records;", total["last_assignment"], "last_assignment;", wrong, "with a later +=/-=/for assignment than the recorded answer")
