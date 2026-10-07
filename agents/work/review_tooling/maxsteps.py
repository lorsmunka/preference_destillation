import sys
sys.path.insert(0, r"D:\preference_destillation")
from transformers import AutoTokenizer
from library.domain import CODE_RECALL, LONG_CONTEXT_RECALL
for teacher in ["google/gemma-3-4b-it", "google/gemma-4-12B-it"]:
    try:
        tok = AutoTokenizer.from_pretrained(teacher)
    except Exception as e:
        print(teacher, "no tokenizer:", type(e).__name__); continue
    for domain in (CODE_RECALL, LONG_CONTEXT_RECALL):
        lengths = sorted((len(tok.encode(r, add_special_tokens=False)), r) for r in domain.example_responses)
        over = [r for n, r in lengths if n > domain.max_steps]
        print(teacher, domain.name, "max_steps", domain.max_steps, "longest answer line", lengths[-1], "over cap:", len(over), over[:3])
