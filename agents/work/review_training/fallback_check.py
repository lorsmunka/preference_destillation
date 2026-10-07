import sys
sys.path.insert(0, r"D:\preference_destillation")
from transformers import AutoTokenizer
from library.shared.batch_files import batch_paths, read_batch
from library.shared.vocabulary import build_vocabulary
from library.domain import as_domain
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
for domain, count in [("reddit_comment_sentiment", 64), ("math_word_problem", 64), ("post_generation", 120)]:
    vocabulary = build_vocabulary(tokenizer, as_domain(domain))
    index = {t: i for i, t in enumerate(vocabulary["token_list"])}
    total = missing = disagree = 0
    for p in batch_paths(rf"D:\preference_destillation\batches\{domain}\google_gemma-3-4b-it")[:count]:
        for r in read_batch(p):
            for s in r["steps"]:
                total += 1
                if s["token"] not in index:
                    missing += 1
                elif index[s["token"]] != s["predicted_token_index"]:
                    disagree += 1
                assert len(s["logits"]) == vocabulary["vocab_size"], (len(s["logits"]), vocabulary["vocab_size"])
    print(f"{domain}: steps={total} token not in output vocab={missing} chosen-token != reduced argmax={disagree}")
