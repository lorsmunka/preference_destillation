import sys, json, math, os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, r"D:/preference_destillation")
import numpy as np
from transformers import AutoTokenizer
from library.shared.batch_files import read_batch
from library.shared.vocabulary import build_vocabulary
from library.domain import POST_GENERATION
tok = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
tl = build_vocabulary(tok, POST_GENERATION)["token_list"]
records = read_batch(r"D:/preference_destillation/batches/post_generation/google_gemma-3-4b-it/batch_1.npz")
# find the step chosen before: H closest to 1.6 among first 32 records
best = None
for ri, r in enumerate(records[:32]):
    for si, s in enumerate(r["steps"]):
        l = np.asarray(s["logits"], dtype=np.float64); p = np.exp(l - l.max()); p /= p.sum()
        H = float(-(p * np.log(np.clip(p, 1e-30, 1))).sum())
        if best is None or abs(H - 1.6) < abs(best[0] - 1.6): best = (H, ri, si)
H, ri, si = best
l = np.asarray(records[ri]["steps"][si]["logits"], dtype=np.float64)
p1 = np.exp(l - l.max()); p1 /= p1.sum()
top = list(np.argsort(-p1)[:8])
res = {"record": ri, "step": si, "sentence": records[ri]["sentence"], "response": records[ri]["model_response"],
       "prefix": "".join(st["token"] for st in records[ri]["steps"][:si]).replace("▁", " "), "tokens": [tl[i] for i in top], "by_T": {}}
for T in [0.5, 1, 2, 3, 5]:
    p = np.exp((l - l.max()) / T); p /= p.sum()
    HT = float(-(p * np.log(np.clip(p, 1e-30, 1))).sum())
    res["by_T"][str(T)] = {"top": [float(p[i]) for i in top], "rest": float(1 - sum(p[i] for i in top)), "H": HT, "expH": math.exp(HT)}
json.dump(res, open(r"C:/Users/Ors/AppData/Local/Temp/claude/D--preference-destillation/ffe199e2-eaba-41f3-879f-8ba3f4ec1e4f/scratchpad/slideshow/temps.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False)[:3000])
