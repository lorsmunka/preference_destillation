import sys, numpy as np
sys.path.insert(0, r"D:\preference_destillation")
from library.shared.batch_files import read_batch, batch_path
for dom, V, files in [("reddit_comment_sentiment", 64, range(52, 65)), ("reddit_comment_sentiment", 525, range(52, 65)), ("post_generation", 26659, range(97, 121))]:
    d = rf"D:\preference_destillation\batches\{dom}\google_gemma-3-4b-it"
    steps = ties = 0
    for i in list(files)[:4]:
        for rec in read_batch(batch_path(d, i)):
            for s in rec["steps"]:
                row = np.sort(np.asarray(s["logits"][:V], dtype=np.float32))[::-1]
                steps += 1
                ties += row[19] == row[20]
    print(dom, "V", V, "steps", steps, "share of steps where teacher's 20th and 21st logits tie:", round(ties / steps, 3))
