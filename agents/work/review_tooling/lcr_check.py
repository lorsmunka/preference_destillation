import sys, collections
sys.path.insert(0, r"D:\preference_destillation")
from library.shared.batch_files import read_batch_texts, batch_paths
from library.shared.metrics.task import ExactAnswerMetric
m = ExactAnswerMetric()
d = r"D:\preference_destillation\batches\long_context_recall\google_gemma-4-12B-it"
paths = batch_paths(d)
c = collections.Counter(); starts = collections.Counter(); n = 0; examples = []
for p in paths:
    for r in read_batch_texts(p):
        n += 1
        a = m._extract_answer(r["model_response"])
        if a is None: c["none"] += 1
        elif a.startswith("answer"): c["prefixed"] += 1; examples.append(r["model_response"][:60])
        else: c["clean"] += 1
        starts[repr(r["model_response"][:2])] += 1
print(len(paths), "files", n, "examples", dict(c)); print(starts.most_common(8)); print(examples[:5])
