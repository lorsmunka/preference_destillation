"""Reddit-comment corpus producer — downloads a Kaggle comment dataset, cleans and
length-filters it into {"text": ...} records. Heavy deps (kagglehub, pandas, tqdm,
transformers) are imported lazily, so `import library` never needs them. Run standalone
(`python -m library.domain.reddit_comment_sentiment.corpus`) or via
`get_domain("reddit_comment_sentiment").build_corpus(count=...)`.
"""

import re
import os
import json
from typing import Optional

from library.shared import MODEL_NAME, MIN_SENTENCE_LENGTH, MAX_SENTENCE_LENGTH

KAGGLE_DATASET = "smagnan/1-million-reddit-comments-from-40-subreddits"


def download_reddit_data():
    import kagglehub
    import pandas as pd
    print("Downloading Reddit dataset...")
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
    df = pd.read_csv(os.path.join(path, csv_files[0]))
    return df["body"].dropna()


def load_gemma_tokenizer():
    from transformers import AutoTokenizer
    print("Loading Gemma-3 tokenizer...")
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def filter_sentences(comments, tokenizer):
    from tqdm import tqdm
    sentences = []
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    noise_pattern = re.compile(r"[^a-zA-Z0-9\s.,!?\'\"-]")
    reddit_artifacts = {"[deleted]", "[removed]"}

    for comment in tqdm(comments, desc="Processing comments"):
        if not isinstance(comment, str) or comment.lower() in reddit_artifacts:
            continue
        comment = url_pattern.sub("", comment)
        comment = noise_pattern.sub("", comment)
        comment = re.sub(r"\s+", " ", comment).strip()
        if not comment:
            continue
        token_count = len(tokenizer.encode(comment, add_special_tokens=False))
        if MIN_SENTENCE_LENGTH <= token_count <= MAX_SENTENCE_LENGTH:
            sentences.append(comment)

    return list(set(sentences))


def build_corpus(output_path: Optional[str] = None, count: Optional[int] = None) -> None:
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "corpus.jsonl")

    tokenizer = load_gemma_tokenizer()
    comments = download_reddit_data()
    sentences = filter_sentences(comments, tokenizer)
    if count is not None:
        sentences = sentences[:count]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        for sentence in sentences:
            file.write(json.dumps({"text": sentence}, ensure_ascii=False) + "\n")

    print(f"Saved {len(sentences)} sentences to {output_path}")


if __name__ == "__main__":
    build_corpus()
