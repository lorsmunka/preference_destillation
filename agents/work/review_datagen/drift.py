"""For greedy-generated batches, the stored argmax index should point at the stored token
whenever the token is in the reduced vocabulary. Rebuild today's vocabulary and count how
often that holds -> detects vocabulary drift since generation."""
import sys, numpy as np
sys.path.insert(0, r"D:\preference_destillation")
from transformers import AutoTokenizer
from library.shared.batch_files import batch_paths, read_batch
from library.shared.vocabulary import build_vocabulary
from library.shared.config import teacher_weights_path
from library.domain import get_domain
combos = [("long_context_recall", "google/gemma-4-12B-it"), ("math_word_problem", "google/gemma-3-4b-it"),
          ("reddit_comment_sentiment", "google/gemma-3-4b-it"), ("post_generation", "google/gemma-3-4b-it")]
for domain_name, teacher in combos:
    tok = AutoTokenizer.from_pretrained(teacher_weights_path(teacher), local_files_only=True)
    domain = get_domain(domain_name)
    v = build_vocabulary(tok, domain)
    index = {t: i for i, t in enumerate(v["token_list"])}
    paths = batch_paths(rf"D:\preference_destillation\batches\{domain_name}\{teacher.replace('/', '_')}")
    width = set(); agree = total = 0; examples = 0
    for p in paths[:20]:
        for r in read_batch(p):
            examples += 1
            for s in r["steps"]:
                width.add(len(s["logits"]))
                if s["token"] in index:
                    total += 1; agree += index[s["token"]] == s["predicted_token_index"]
                    agree_argmax = None
    print(f"{domain_name:28s} files {len(paths):5d} vocab now {v['vocab_size']} stored width {sorted(width)} "
          f"argmax==token-index {agree}/{total}")
