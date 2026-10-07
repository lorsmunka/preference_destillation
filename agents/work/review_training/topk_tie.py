import sys
sys.path.insert(0, r"D:\preference_destillation")
import numpy as np, torch
from library.shared.batch_files import batch_paths, read_batch
for domain in ["reddit_comment_sentiment", "math_word_problem", "post_generation"]:
    paths = batch_paths(rf"D:\preference_destillation\batches\{domain}\google_gemma-3-4b-it")[:5]
    rows = torch.from_numpy(np.concatenate([np.stack([s["logits"] for r in read_batch(p) for s in r["steps"]]) for p in paths]))
    top = rows.topk(21, dim=-1).values
    boundary_tie = (top[:, 19] == top[:, 20]).float().mean().item()
    probs = torch.softmax(rows, -1)
    mass_top1 = probs.max(-1).values
    mass_2_to_20 = probs.topk(20, -1).values[:, 1:].sum(-1)
    print(f"{domain}: rows={len(rows)} P(20th==21st logit, top-20 set ambiguous)={boundary_tie:.3f}  "
          f"median teacher mass on ranks 2-20={mass_2_to_20.median():.2e}  mean={mass_2_to_20.mean():.2e}")
