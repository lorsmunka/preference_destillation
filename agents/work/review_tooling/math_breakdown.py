import sys, collections, torch
sys.path.insert(0, r"D:\preference_destillation")
assert not torch.cuda.is_available()
from library import Analysis
from library.tooling.analysis import _batch_paths, _iter_examples
from library.model.student import StudentModel
from library.shared.metrics.task import MathMetric
name = sys.argv[1]
a = Analysis(name)
label, path = a.store.resolve_checkpoint(name, "latest")
student = StudentModel.from_run(a.run, path, device="cpu")
m = MathMetric()
stats = collections.Counter(); predicted_answers = collections.Counter()
for bp in _batch_paths(a.run, "test", None):
    for ex in _iter_examples(bp):
        truth = m._extract_solution(ex["model_response"])
        kind = "bool" if truth in ("True", "False") else "number"
        out = student.generate(ex["sentence"], max_new_tokens=2 * len(ex["steps"]))
        pred = m._extract_solution("".join(out.token_strings).replace("▁", " "))
        stats[kind] += 1; stats[kind + "_correct"] += pred == truth
        predicted_answers[pred if pred in ("True", "False") else "number/none"] += 1
print(label, dict(stats), "overall", (stats["bool_correct"] + stats["number_correct"]) / (stats["bool"] + stats["number"]), dict(predicted_answers))
