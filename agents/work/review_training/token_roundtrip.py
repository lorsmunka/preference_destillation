"""Check: does ExampleEncoder.step_data's re-encoding of the stored token string give back
the teacher's actual token id? (read-only over batches/)"""
import sys
from collections import Counter
sys.path.insert(0, r"D:\preference_destillation")

from transformers import AutoTokenizer
from library.shared.batch_files import batch_paths, read_batch_texts

domains = {
    "reddit_comment_sentiment": ("google/gemma-3-4b-it", 200),
    "math_word_problem": ("google/gemma-3-4b-it", 200),
    "post_generation": ("google/gemma-3-4b-it", 243),
    "long_context_recall": ("google/gemma-4-12B-it", 159),
}
for domain, (teacher, max_files) in domains.items():
    tokenizer = AutoTokenizer.from_pretrained(teacher)
    directory = rf"D:\preference_destillation\batches\{domain}\{teacher.replace('/', '_')}"
    paths = batch_paths(directory)[:max_files]
    total = 0
    mismatched = Counter()
    multi = Counter()
    for path in paths:
        for record in read_batch_texts(path):
            for token in record["tokens"]:
                total += 1
                true_id = tokenizer.convert_tokens_to_ids(token)
                encoded = tokenizer.encode(token, add_special_tokens=False)
                first = encoded[0] if encoded else tokenizer.unk_token_id
                if first != true_id:
                    mismatched[(token, true_id, tuple(encoded))] += 1
                if len(encoded) != 1:
                    multi[(token, tuple(encoded))] += 1
    mismatch_total = sum(mismatched.values())
    print(f"{domain}: {len(paths)} files, {total} step tokens, mismatched={mismatch_total} "
          f"({100*mismatch_total/max(total,1):.3f}%), multi-id encodings={sum(multi.values())}")
    for (token, true_id, encoded), count in mismatched.most_common(12):
        print(f"   {token!r} true={true_id} -> encode={list(encoded)} "
              f"{[tokenizer.convert_ids_to_tokens(i) for i in encoded]} x{count}")
