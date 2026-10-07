import sys, time, json, random, torch, numpy as np
sys.path.insert(0, r"D:\preference_destillation")
from transformers import AutoTokenizer
from library.shared.vocabulary import build_vocabulary, get_response_tokens
from library.domain.math_tutor.agentic import MathTutorAgenticDomain
from library.shared.batch_files import write_batch, read_batch
tok = AutoTokenizer.from_pretrained(r"D:\preference_destillation\models\gemma-4-12B-it")
domain = MathTutorAgenticDomain("named", "forced")
v = build_vocabulary(tok, domain); V = v["vocab_size"]; resp = get_response_tokens(tok, domain)
records = [json.loads(l) for l in open(r"D:\preference_destillation\library\domain\math_tutor\corpus_named.jsonl", encoding="utf-8")]
print("records", len(records), "keys", sorted(records[0].keys()), "teacher modes", {r.get("teacher") for r in records})
sample = random.Random(0).sample(records, 2000)
lens = [len(tok(r["target"], add_special_tokens=False)["input_ids"]) for r in sample]
plens = [len(tok(r["text"])["input_ids"]) for r in sample]
mean_steps = sum(lens) / len(lens)
print("V", V, "mean target tokens", round(mean_steps, 1), "max", max(lens), "mean context tokens", round(sum(plens)/len(plens)), "max", max(plens))
print("est. npz bytes per record", int(mean_steps * V * 2), "-> corpus total GB", round(len(records) * mean_steps * V * 2 / 1e9, 1))
# per-step python overhead as written in model_handler
row = torch.randn(16, V)
t = time.time()
for _ in range(200):
    lv = row[3].tolist(); p = int(max(range(len(lv)), key=lv.__getitem__))
per_step = (time.time() - t) / 200
print("per-step tolist+python argmax ms", round(per_step * 1000, 3), "-> corpus hours", round(per_step * len(records) * mean_steps / 3600, 2))
t = time.time()
for r in sample[:300]:
    toks = tok.tokenize(r["target"]); ok = all(x in resp for x in toks)
per_check = (time.time() - t) / 300
print("finish_example list-membership check ms/example", round(per_check * 1000, 2), "-> corpus minutes", round(per_check * len(records) / 60, 1))
respset = set(resp)
t = time.time()
for r in sample[:300]:
    toks = tok.tokenize(r["target"]); ok = all(x in respset for x in toks)
print("same with a set ms/example", round((time.time() - t) / 300 * 1000, 3))
# write_batch for 32 examples
recs = [{"sentence": "s", "model_response": "r", "steps": [{"token": "a", "logits": torch.randn(V).tolist(), "predicted_token_index": 0} for _ in range(int(mean_steps))]} for _ in range(32)]
t = time.time(); path = write_batch(r"C:\Users\Ors\AppData\Local\Temp\claude\D--preference-destillation\ffe199e2-eaba-41f3-879f-8ba3f4ec1e4f\scratchpad\review_datagen\b.npz", recs)
import os
print("write_batch 32 ex seconds", round(time.time() - t, 2), "file MB", round(os.path.getsize(path) / 1e6, 1))
