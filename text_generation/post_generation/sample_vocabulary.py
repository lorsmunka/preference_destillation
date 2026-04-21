import json
import sys
import os
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from shared import MODEL_NAME
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

SAMPLE_COUNT = 10000
MAX_NEW_TOKENS = 200
INPUT_PATH = "./text_generation/reddit_comment_sentiment/reddit_comments.jsonl"
OUTPUT_PATH = "./text_generation/post_generation/sample_posts.txt"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", dtype=torch.bfloat16)

end_token_ids = tokenizer.encode("<end>", add_special_tokens=False)


class StopOnEnd(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        if len(input_ids[0]) >= len(end_token_ids):
            recent = input_ids[0][-len(end_token_ids):].tolist()
            if recent == end_token_ids:
                return True
        return False


all_comments = []
with open(INPUT_PATH, "r", encoding="utf-8") as file:
    for line in file:
        all_comments.append(json.loads(line)["text"])
comments = all_comments[:SAMPLE_COUNT]

start_index = 0
if os.path.exists(OUTPUT_PATH):
    with open(OUTPUT_PATH, "r", encoding="utf-8") as existing:
        first_line = existing.readline().strip()
        if first_line.isdigit():
            start_index = int(first_line)
            print(f"Resuming from {start_index}")

if start_index == 0:
    file = open(OUTPUT_PATH, "w", encoding="utf-8")
    file.write("0         \n")
else:
    file = open(OUTPUT_PATH, "r+", encoding="utf-8")
    file.seek(0, 2)
file.flush()

all_tokens = set()
if start_index > 0:
    with open(OUTPUT_PATH, "r", encoding="utf-8") as existing:
        for line in existing:
            if line.strip() and not line.strip().isdigit():
                all_tokens.update(tokenizer.tokenize(line.strip()))
    print(f"Loaded {len(all_tokens)} unique tokens from existing posts")

for index, comment in enumerate(comments):
    if index < start_index:
        continue
    prompt = f'Generate a plausible reddit post based on this comment: "{comment}"\n\nNo title or text formatting needed. Only reply with the body of the post. 3 sentences or about 50 words. End post with <end>.\n\nPost: '
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

    tokens = tokenizer.tokenize(generated)
    all_tokens.update(tokens)
    print(
        f"[{index + 1}/{SAMPLE_COUNT}] ({len(all_tokens)} unique tokens) Comment: {comment[:60]}")
    print(f"  -> {generated[:80]}")

file.close()
print(f"\nDone. Unique tokens across {SAMPLE_COUNT} posts: {len(all_tokens)}")
