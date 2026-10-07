"""Explain targets vs the output vocabulary: finish_example drops any target with a token outside
example+whitespace tokens. Tokenizer only (tokenizer.json), no model."""
import sys, json, glob
from collections import Counter
sys.path.insert(0, r"D:\preference_destillation")
from tokenizers import Tokenizer
from library.shared.vocabulary import WHITESPACE_TOKENS
tok = Tokenizer.from_file(r"D:\preference_destillation\models\gemma-4-12B-it\tokenizer.json")
allowed = set(json.load(open(r"D:\preference_destillation\library\domain\math_tutor\output_tokens.json", encoding="utf-8"))["tokens"]) | set(WHITESPACE_TOKENS)
c = Counter(); bad = Counter()
for p in sorted(glob.glob(r"D:\preference_destillation\library\domain\math_tutor\harness_runs\*.json")):
    d = json.load(open(p, encoding="utf-8"))
    for conv in d["conversations"]:
        for t in conv["turns"]:
            out = t.get("output", "")
            if not out.startswith("<explanation>"): continue
            target = "<explanation>" + out[len("<explanation>"):].replace("</explanation>", "").strip() + "</explanation>"
            tokens = tok.encode(target, add_special_tokens=False).tokens
            outside = [x for x in tokens if x not in allowed]
            c["explanations"] += 1
            if outside: c["with a token outside the vocabulary (dropped)"] += 1; bad.update(outside)
print(c); print(bad.most_common(20))
