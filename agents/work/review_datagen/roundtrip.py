import sys
sys.path.insert(0, r"D:\preference_destillation")
from transformers import AutoTokenizer
from library.shared.vocabulary import build_vocabulary
from library.domain import MATH_TUTOR, CODE_RECALL
from library.domain.math_tutor.agentic import MathTutorAgenticDomain
tok = AutoTokenizer.from_pretrained(r"D:\preference_destillation\models\gemma-4-12B-it")
for domain in [MATH_TUTOR, CODE_RECALL, MathTutorAgenticDomain("facts", "forced")]:
    v = build_vocabulary(tok, domain)
    bad = []
    for token in v["token_list"]:
        ids = tok.encode(token, add_special_tokens=False)
        if ids != [v["token_to_id"][token]]:
            bad.append((token, tok.convert_ids_to_tokens(ids)))
    print(domain.name, "vocab", v["vocab_size"], "token-string -> encode mismatches:", len(bad), bad[:8])
