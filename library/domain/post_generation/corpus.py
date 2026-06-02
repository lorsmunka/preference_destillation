"""Post-generation example producer — runs the teacher over reddit comments to write
sample_posts.txt, the example posts that seed this domain's `example_responses` vocab
section. Heavy (torch + transformers + the teacher), imported lazily. The *input* corpus
for post-generation is the reddit comments (reused), so there is no separate `build_corpus`
here — see PostGenerationDomain. Run standalone
(`python -m library.domain.post_generation.corpus`) or via
`get_domain("post_generation").build_example_responses(count=...)`.
"""

import os
import json
from typing import Optional

DEFAULT_COUNT = 10000
MAX_NEW_TOKENS = 200


def build_example_responses(output_path: Optional[str] = None, count: Optional[int] = None) -> None:
    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList,
    )
    from library.shared import MODEL_NAME
    from ..registry import get_domain

    count = count or DEFAULT_COUNT
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "sample_posts.txt")
    input_path = get_domain("reddit_comment_sentiment").corpus_path

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map="auto", dtype=torch.bfloat16)

    end_token_ids = tokenizer.encode("<end>", add_special_tokens=False)

    class StopOnEnd(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            if len(input_ids[0]) >= len(end_token_ids):
                if input_ids[0][-len(end_token_ids):].tolist() == end_token_ids:
                    return True
            return False

    with open(input_path, "r", encoding="utf-8") as file:
        comments = [json.loads(line)["text"] for line in file][:count]

    start_index = 0
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as existing:
            first_line = existing.readline().strip()
            if first_line.isdigit():
                start_index = int(first_line)
                print(f"Resuming from {start_index}")

    if start_index == 0:
        file = open(output_path, "w", encoding="utf-8")
        file.write("0         \n")
    else:
        file = open(output_path, "r+", encoding="utf-8")
        file.seek(0, 2)
    file.flush()

    all_tokens = set()
    if start_index > 0:
        with open(output_path, "r", encoding="utf-8") as existing:
            for line in existing:
                if line.strip() and not line.strip().isdigit():
                    all_tokens.update(tokenizer.tokenize(line.strip()))
        print(f"Loaded {len(all_tokens)} unique tokens from existing posts")

    for index, comment in enumerate(comments):
        if index < start_index:
            continue
        prompt = (
            f'Generate a plausible reddit post based on this comment: "{comment}"\n\n'
            "No title or text formatting needed. Only reply with the body of the post. "
            "3 sentences or about 50 words. End post with <end>.\n\nPost: "
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                stopping_criteria=StoppingCriteriaList([StopOnEnd()]))

        generated = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        file.write(generated + "\n")
        file.flush()
        file.seek(0)
        file.write(f"{index + 1:<10}")
        file.seek(0, 2)

        all_tokens.update(tokenizer.tokenize(generated))
        print(f"[{index + 1}/{count}] ({len(all_tokens)} unique tokens) Comment: {comment[:60]}")
        print(f"  -> {generated[:80]}")

    file.close()
    print(f"\nDone. Unique tokens across {count} posts: {len(all_tokens)}")


if __name__ == "__main__":
    build_example_responses()
