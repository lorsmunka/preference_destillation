import json, os, collections
MAJ = {"tone": "neutral", "sentiment": "negative", "safety": "safe", "toxicity": "respectful"}
def share(name, epoch_index=-1):
    evals = [json.loads(l) for l in open(f"runs/{name}/logs/training.jsonl", encoding="utf-8") if '"eval_epoch"' in l]
    e = evals[epoch_index]
    out = {}
    for cat, matrix in e["confusion_matrices"].items():
        pred = collections.Counter()
        for true, row in matrix.items():
            for p, c in row.items(): pred[p] += c
        tot = sum(pred.values())
        out[cat] = round(pred[MAJ[cat]] / tot, 3) if tot else None
    return round(e["task_accuracy"], 4), out
for n in ["ideaD-data00256-s5", "ideaD-data00512-s2", "ideaD-data01024-s2", "ideaD-data02048-s3", "e01-reddit-pure-ce-s1", "e01-reddit-kl-to-ce-s1", "e01-reddit-pure-kl-s10"]:
    print(n, "epoch0", share(n, 0), "final", share(n, -1))
# per-arm mean of best/final for e01 reddit vs majority 0.6262
import csv
rows = list(csv.DictReader(open("experiments/results/e01_anneal_direction.csv", encoding="utf-8")))
for dom in ["reddit", "math"]:
    for arm in ["ce-to-kl", "kl-to-ce", "flat-50", "pure-ce", "pure-kl"]:
        g = [r for r in rows if r["domain"] == dom and r["arm"] == arm]
        b = sum(float(r["task_accuracy_best"]) for r in g) / len(g); f = sum(float(r["task_accuracy_final"]) for r in g) / len(g)
        print(dom, arm, len(g), "best", round(b, 4), "final", round(f, 4))
