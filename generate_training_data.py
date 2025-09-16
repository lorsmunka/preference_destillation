import requests
import json
import re
import os
import pandas as pd
from tqdm import tqdm
import time
import random
import kagglehub
import glob

model_name = "gemma3:4b"
base_url = "http://localhost:11434"

# Download and load data
if not os.path.exists('reddit_sentences.json'):
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
            if 3 <= word_count <= 10 and not comment.startswith('[') and 'http' not in comment.lower():
                sentences.append(comment)

    sentences = list(set(sentences))

    # Save sentences to file for resume capability
    with open('reddit_sentences.json', 'w') as f:
        json.dump(sentences, f)
    print(f"Saved {len(sentences)} unique sentences")
else:
    print("Loading existing sentences...")
    with open('reddit_sentences.json', 'r') as f:
        sentences = json.load(f)
    print(f"Loaded {len(sentences)} sentences")

os.makedirs("training_data", exist_ok=True)

# Count existing examples and determine resume point
existing_files = glob.glob("training_data/train_*.jsonl")
generated = 0
processed_sentences = set()

if existing_files:
    print("Counting existing examples...")
    for file_path in existing_files:
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    processed_sentences.add(data['sentence'])
                    generated += 1

print(
    f"Already processed {generated} examples, {len(processed_sentences)} unique sentences")

# Filter out already processed sentences
remaining_sentences = [s for s in sentences if s not in processed_sentences]
print(f"Remaining to process: {len(remaining_sentences)}")

# Determine current file number and examples in current file
file_num = 1
examples_in_file = 0

if existing_files:
    file_nums = [int(re.search(r'train_(\d+)\.jsonl', f).group(1))
                 for f in existing_files]
    file_num = max(file_nums)

    current_file = f"training_data/train_{file_num:03d}.jsonl"
    with open(current_file, 'r') as f:
        examples_in_file = sum(1 for line in f if line.strip())

    if examples_in_file >= 5000:
        file_num += 1
        examples_in_file = 0

print(
    f"Starting with file {file_num}, {examples_in_file} examples in current file")

pbar = tqdm(total=min(50000 - generated,
            len(remaining_sentences)), desc="Processing")

for sentence in remaining_sentences:
    if generated >= 50000:
        break

    try:
        prompt = f'Classify: "{sentence}"\n\nHarmful = offensive, threatening, or inappropriate\nSentiment = positive or negative emotion\nTone = serious or joking\n\nReturn only: {{"harmful": "harmful" or "not_harmful", "sentiment": "positive" or "negative", "tone": "serious" or "joking"}}'

        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1}
            },
            timeout=30
        )

        text = response.json()['response']
        json_match = re.search(r'\{.*?\}', text, re.DOTALL)

        if json_match:
            classification = json.loads(json_match.group())

            if all(key in classification for key in ["harmful", "sentiment", "tone"]):
                data = {
                    "sentence": sentence,
                    **classification
                }

                with open(f"training_data/train_{file_num:03d}.jsonl", 'a') as f:
                    f.write(json.dumps(data) + '\n')

                generated += 1
                examples_in_file += 1

                pbar.update(1)
                pbar.set_description(f"Generated: {generated}")

                if examples_in_file >= 5000:
                    print(f"Completed file {file_num} with 5000 examples")
                    file_num += 1
                    examples_in_file = 0

    except Exception as e:
        print(f"Error processing sentence: {e}")
        continue

    time.sleep(0.05)

pbar.close()
print(f"Done! Generated {generated} total examples")
