import sys, json, math
sys.path.insert(0, r"D:/preference_destillation")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from transformers import AutoTokenizer
from library.shared.batch_files import read_batch
from library.shared.vocabulary import build_vocabulary
from library.domain import POST_GENERATION, MATH_WORD_PROBLEM, REDDIT_SENTIMENT

tok = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
out = {}
for key, domain, folder in [("postgen", POST_GENERATION, "post_generation"), ("math", MATH_WORD_PROBLEM, "math_word_problem"), ("reddit", REDDIT_SENTIMENT, "reddit_comment_sentiment")]:
    vocab = build_vocabulary(tok, domain)
    tl = vocab["token_list"]
    records = read_batch(f"D:/preference_destillation/batches/{folder}/google_gemma-3-4b-it/batch_1.npz")
    cands = []
    for ri, r in enumerate(records[:32]):
        for si, s in enumerate(r["steps"]):
            l = np.asarray(s["logits"], dtype=np.float64)
            p = np.exp(l - l.max()); p /= p.sum()
            H = float(-(p * np.log(np.clip(p, 1e-30, 1))).sum())
            cands.append((H, ri, si))
    cands.sort()
    Hs = [c[0] for c in cands]
    print(key, "V", len(tl), "steps", len(cands), "mean H", np.mean(Hs), "exp", math.exp(np.mean(Hs)))
    # pick a step near ~1.6 nats for postgen; for math pick median
    target = 1.6 if key == "postgen" else (Hs[-1] if key == "reddit" else float(np.median(Hs)))
    H, ri, si = min(cands, key=lambda c: abs(c[0] - target))
    r = records[ri]
    s = r["steps"][si]
    l = np.asarray(s["logits"], dtype=np.float64)
    p = np.exp(l - l.max()); p /= p.sum()
    top = np.argsort(-p)[:8]
    prefix = "".join(st["token"] for st in r["steps"][:si]).replace("▁", " ")
    out[key] = {"H": H, "expH": math.exp(H), "V": len(tl), "sentence": r["sentence"][:300], "prefix": prefix[-200:],
                "chosen": s["token"], "top": [(tl[i], float(p[i])) for i in top], "batch": "batch_1", "record": ri, "step": si}
print(json.dumps(out, ensure_ascii=False, indent=1))
