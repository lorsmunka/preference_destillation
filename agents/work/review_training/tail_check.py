import sys
sys.path.insert(0, r"D:\preference_destillation")
import numpy as np, torch
from library.shared.batch_files import batch_paths, read_batch
for domain in ["reddit_comment_sentiment", "math_word_problem", "post_generation"]:
    paths = batch_paths(rf"D:\preference_destillation\batches\{domain}\google_gemma-3-4b-it")[:5]
    rows = np.concatenate([np.stack([s["logits"] for r in read_batch(p) for s in r["steps"]]) for p in paths])
    logits = torch.from_numpy(rows)
    probs = torch.softmax(logits, -1)
    log_probs = torch.log_softmax(logits, -1)
    clamped = probs < 1e-12
    spread = (logits.max(-1).values - logits.min(-1).values)
    print(f"{domain}: rows={len(rows)} V={rows.shape[1]} share of probs<1e-12={clamped.float().mean():.3f} "
          f"median max-min logit spread={spread.median():.1f} min log p median={log_probs.min(-1).values.median():.1f}")
