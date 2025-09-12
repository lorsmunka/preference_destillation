import requests
import json
import re
import os
import pandas as pd
from tqdm import tqdm
import time
import random
import kagglehub

model_name = "gemma3:4b"
base_url = "http://localhost:11434"

print("Downloading Reddit dataset...")
path = kagglehub.dataset_download(
    "smagnan/1-million-reddit-comments-from-40-subreddits")

print("Loading sentences...")
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
df = pd.read_csv(os.path.join(path, csv_files[0]))
comment_column = 'body' if 'body' in df.columns else 'comment'

sentences = []
for comment in df[comment_column].dropna():
    if isinstance(comment, str):
        comment = comment.strip()
        word_count = len(comment.split())
        if 3 <= word_count <= 10:
            sentences.append(comment)

sentences = list(set(sentences))
random.shuffle(sentences)
print(f"Found {len(sentences)} sentences")

os.makedirs("training_data", exist_ok=True)

generated = 0
failed_attempts = 0
file_num = 1
examples_in_file = 0

pbar = tqdm(total=50000, desc="Generating")

for sentence in sentences:
    if generated >= 50000:
        break

    try:
        prompt = f'Classify: "{sentence}"\n\nReturn only: {{"harmful": "harmful" or "not_harmful", "formality": "formal" or "informal", "sentiment": "positive" or "negative"}}'

        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3}
            },
            timeout=30
        )

        text = response.json()['response']
        json_match = re.search(r'\{.*?\}', text, re.DOTALL)

        if json_match:
            classification = json.loads(json_match.group())

            if all(key in classification for key in ["harmful", "formality", "sentiment"]):
                data = {
                    "sentence": sentence,
                    **classification
                }

                with open(f"training_data/train_{file_num:03d}.jsonl", 'a') as f:
                    f.write(json.dumps(data) + '\n')

                generated += 1
                examples_in_file += 1
                failed_attempts = 0

                pbar.update(1)
                pbar.set_description(f"Generated: {generated}")

                if examples_in_file == 5000:
                    file_num += 1
                    examples_in_file = 0

    except Exception:
        failed_attempts += 1
        continue

    time.sleep(0.05)

pbar.close()
print(f"Done! Generated {generated} examples")
