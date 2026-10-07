import sys
sys.path.insert(0, r"D:\preference_destillation")
from collections import Counter
from transformers import AutoTokenizer
from library.shared.vocabulary import build_vocabulary, WHITESPACE_TOKENS, AUXILIARY_TOKENS, get_response_tokens
from library.domain import CODE_RECALL, MATH_TUTOR
from library.domain.math_tutor.agentic import MathTutorAgenticDomain
tok = AutoTokenizer.from_pretrained(r"D:\preference_destillation\models\gemma-4-12B-it")
print("convert unknown ->", tok.convert_tokens_to_ids(["thisisnotatoken_xyz"]), "unk", tok.unk_token_id)
print("whitespace ids:", [(repr(t), tok.convert_tokens_to_ids([t])[0]) for t in WHITESPACE_TOKENS])
for domain in [CODE_RECALL, MATH_TUTOR, MathTutorAgenticDomain("facts", "forced")]:
    v = build_vocabulary(tok, domain)
    ids = [v["token_to_id"][t] for t in v["token_list"]]
    dup = [i for i, c in Counter(ids).items() if c > 1]
    none = [t for t in v["token_list"] if v["token_to_id"][t] is None]
    print(domain.name, "size", v["vocab_size"], "dup ids", len(dup), [tok.convert_ids_to_tokens([i]) for i in dup][:5], "None ids", none[:5])
    resp = get_response_tokens(tok, domain)
    print("  response tokens", len(resp))
    if domain is MATH_TUTOR:
        print("  token_list", v["token_list"][:60])
    dropped_aux = [t for t in AUXILIARY_TOKENS if tok.convert_tokens_to_ids([t])[0] == tok.unk_token_id]
    print("  aux dropped (unk)", dropped_aux[:20], len(dropped_aux))
