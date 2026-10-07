import time
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(r"D:\preference_destillation\models\gemma-4-12B-it")
t = time.time(); v = tok.get_vocab(); print("get_vocab s", round(time.time() - t, 3))
t = time.time(); tl, ti = tuple(zip(*v.items())); print("zip s", round(time.time() - t, 3))
t = time.time(); hash((tl, ti, (";",))); print("hash s", round(time.time() - t, 3))
