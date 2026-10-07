import sys
sys.path.insert(0, r"D:\preference_destillation")
from transformers import AutoTokenizer
from library.shared.batch_files import batch_paths, read_batch
from library.shared.vocabulary import build_vocabulary
from library.domain import get_domain
for domain_name in ["post_generation", "reddit_comment_sentiment"]:
    tok = AutoTokenizer.from_pretrained("google/gemma-3-4b-it", local_files_only=True)
    v = build_vocabulary(tok, get_domain(domain_name)); index = {t: i for i, t in enumerate(v["token_list"])}
    ties = other = 0; examples = []
    for p in batch_paths(rf"D:\preference_destillation\batches\{domain_name}\google_gemma-3-4b-it")[:20]:
        for r in read_batch(p):
            for s in r["steps"]:
                if s["token"] in index and index[s["token"]] != s["predicted_token_index"]:
                    a, b = s["logits"][index[s["token"]]], s["logits"][s["predicted_token_index"]]
                    if a == b: ties += 1
                    else:
                        other += 1
                        if len(examples) < 4: examples.append((s["token"], v["token_list"][s["predicted_token_index"]], float(a), float(b)))
    print(domain_name, "ties", ties, "non-ties", other, examples)
