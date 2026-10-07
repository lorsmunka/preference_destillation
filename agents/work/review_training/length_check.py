import sys
sys.path.insert(0, r"D:\preference_destillation")
import numpy as np
from transformers import AutoTokenizer
from library.shared.batch_files import batch_paths, read_batch_texts
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
for domain, count in [("reddit_comment_sentiment", 64), ("math_word_problem", 64), ("post_generation", 120)]:
    paths = batch_paths(rf"D:\preference_destillation\batches\{domain}\google_gemma-3-4b-it")[:count]
    lengths, prompts, steps = [], [], []
    for p in paths:
        for r in read_batch_texts(p):
            L = len(tokenizer.encode(r["sentence"] + "\n\n", add_special_tokens=False))
            n = len(r["tokens"])
            lengths.append(L + n - 1); prompts.append(L); steps.append(n)
    lengths = np.array(lengths)
    print(f"{domain}: train seq len max={lengths.max()} p99={np.percentile(lengths,99):.0f} prompt max={max(prompts)} steps max={max(steps)}; "
          f"rollout max possible={max(L+2*n for L,n in zip(prompts,steps))}")
