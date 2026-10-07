import sys, json, collections
sys.path.insert(0, r"D:\preference_destillation")
from library.shared.batch_files import read_batch_texts, batch_path
from library.shared.metrics.task import ClassificationMetric, CATEGORIES
d = r"D:\preference_destillation\batches\reddit_comment_sentiment\google_gemma-3-4b-it"
m = ClassificationMetric()
def split(mte, bs=32, ratio=0.8):
    eff = mte // bs
    return int(eff*ratio), eff
for mte in [256, 512, 1024, 2048, 4096]:
    s, e = split(mte)
    counts = {c: collections.Counter() for c in CATEGORIES}
    n = 0; unparse = 0
    for i in range(s, e):
        for rec in read_batch_texts(batch_path(d, i+1)):
            gt = m._parse(rec["model_response"])
            if gt is None: unparse += 1; continue
            n += 1
            for c in CATEGORIES:
                if c in gt: counts[c][gt[c]] += 1
    tot = sum(sum(v.values()) for v in counts.values())
    maj = sum(v.most_common(1)[0][1] for v in counts.values())
    print(mte, f"test files {s+1}..{e}", "examples", n, "unparseable", unparse, "majority-baseline task acc", round(maj/tot,4),
          {c: (counts[c].most_common(1)[0][0], round(counts[c].most_common(1)[0][1]/sum(counts[c].values()),3)) for c in CATEGORIES})
