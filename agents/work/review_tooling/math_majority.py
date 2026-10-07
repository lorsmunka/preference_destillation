import sys, collections
sys.path.insert(0, r"D:\preference_destillation")
from library.shared.batch_files import read_batch_texts, batch_path
from library.shared.metrics.task import MathMetric
d = r"D:\preference_destillation\batches\math_word_problem\google_gemma-3-4b-it"
m = MathMetric()
for mte in [2048, 4096]:
    eff = mte // 32; s, e = int(eff*0.8), eff
    c = collections.Counter(); n = 0; none = 0
    for i in range(s, e):
        for rec in read_batch_texts(batch_path(d, i+1)):
            sol = m._extract_solution(rec["model_response"])
            if sol is None: none += 1; continue
            n += 1; c[sol] += 1
    print(mte, "examples", n, "no-solution", none, "top answers", [(k, v, round(v/n,4)) for k, v in c.most_common(5)])
    print("  sample responses:", [r["model_response"][:80] for r in read_batch_texts(batch_path(d, s+1))[:3]])
