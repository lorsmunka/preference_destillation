import sys
sys.path.insert(0, r"D:\preference_destillation")
from transformers import AutoTokenizer
from library.shared.batch_files import batch_paths, read_batch
from library.shared.vocabulary import build_vocabulary
from library.domain import as_domain
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
for domain, count in [("reddit_comment_sentiment", 64), ("post_generation", 120)]:
    vocabulary = build_vocabulary(tokenizer, as_domain(domain))
    index = {t: i for i, t in enumerate(vocabulary["token_list"])}
    ties = gaps = 0
    biggest = 0.0
    for p in batch_paths(rf"D:\preference_destillation\batches\{domain}\google_gemma-3-4b-it")[:count]:
        for r in read_batch(p):
            for s in r["steps"]:
                if s["token"] in index and index[s["token"]] != s["predicted_token_index"]:
                    gap = float(s["logits"][s["predicted_token_index"]] - s["logits"][index[s["token"]]])
                    if gap == 0: ties += 1
                    else: gaps += 1; biggest = max(biggest, gap)
    print(domain, "exact ties:", ties, "non-tie:", gaps, "largest gap:", biggest)
