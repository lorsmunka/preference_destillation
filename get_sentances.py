import re
import os
import json
import pandas as pd
import kagglehub
from tqdm import tqdm
from transformers import AutoTokenizer


def download_reddit_data():
    print("Downloading Reddit dataset...")
    path = kagglehub.dataset_download(
        "smagnan/1-million-reddit-comments-from-40-subreddits")

    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    df = pd.read_csv(os.path.join(path, csv_files[0]))
    comment_column = 'body'

    return df[comment_column].dropna()


def load_gemma_tokenizer():
    print("Loading Gemma-3 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-3-4b-it",
    )
    return tokenizer


def filter_sentences(comments, tokenizer):
    sentences = []
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    noise_pattern = re.compile(r'[^a-zA-Z0-9\s.,!?\'"-]')
    reddit_artifacts = {'[deleted]', '[removed]'}

    for comment in tqdm(comments, desc="Processing comments"):
        if not isinstance(comment, str):
            continue

        if comment.lower() in reddit_artifacts:
            continue

        comment = url_pattern.sub('', comment)
        comment = noise_pattern.sub('', comment)
        comment = re.sub(r'\s+', ' ', comment).strip()

        if comment:
            tokens = tokenizer.encode(comment, add_special_tokens=False)
            token_count = len(tokens)

            if 3 <= token_count <= 25:
                sentences.append(comment)

    return list(set(sentences))


def save_sentences_jsonl(sentences):
    os.makedirs("sentences", exist_ok=True)

    with open("sentences/sentences.jsonl", 'w', encoding='utf-8') as f:
        for sentence in sentences:
            json_line = {"text": sentence}
            f.write(json.dumps(json_line, ensure_ascii=False) + '\n')

    print(f"Saved {len(sentences)} sentences to sentences/sentences.jsonl")


tokenizer = load_gemma_tokenizer()
comments = download_reddit_data()
sentences = filter_sentences(comments, tokenizer)
save_sentences_jsonl(sentences)
