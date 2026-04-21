# Centralized configuration for preference distillation project

import json
import os
import torch


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Domain paths
INPUT_PATHS = {
    "reddit_comment_sentiment": "./text_generation/reddit_comment_sentiment/reddit_comments.jsonl",
    "math_word_problem": "./text_generation/math_word_problem/math_word_problems.jsonl",
    "post_generation": "./text_generation/reddit_comment_sentiment/reddit_comments.jsonl",
}


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def get_input_path(domain: str) -> str:
    if domain not in INPUT_PATHS:
        raise ValueError(f"Unknown domain: {domain}")
    return INPUT_PATHS[domain]


def get_output_dir(domain: str, model_name: str) -> str:
    return os.path.join("./batches", domain, sanitize_model_name(model_name))


def get_batches_dir(domain: str, teacher_model: str) -> str:
    return os.path.join("./batches", domain, sanitize_model_name(teacher_model))


def get_training_run_dir(run_name: str) -> str:
    return os.path.join("./runs", run_name)


def load_input_vocabulary(domain: str, teacher_model: str):
    path = os.path.join(get_batches_dir(domain, teacher_model), "input_vocabulary.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# Queue paths
DATA_GENERATION_QUEUE_PATH = "./distillation_data_generation/data_generation_queue.json"
TRAINING_QUEUE_PATH = "./training/training_queue.json"

# Model identification (default teacher model)
MODEL_NAME = "google/gemma-3-4b-it"

# Sentence filtering (for data preparation)
MIN_SENTENCE_LENGTH = 3
MAX_SENTENCE_LENGTH = 25

# Distillation — per-domain generation limits
DOMAIN_MAX_GENERATION_STEPS = {
    "reddit_comment_sentiment": 50,
    "math_word_problem": 350,
    "post_generation": 150,
}
PROMPT_DELIMITER = "\n\n"

# Transformer architecture defaults (overridden by training queue)
HIDDEN_DIM = 384
NUM_LAYERS = 18
NUM_HEADS = 8
DOMAIN_MAX_SEQ_LENGTH = {
    "reddit_comment_sentiment": MAX_SENTENCE_LENGTH + DOMAIN_MAX_GENERATION_STEPS["reddit_comment_sentiment"],
    "math_word_problem": 210,
    "post_generation": MAX_SENTENCE_LENGTH + DOMAIN_MAX_GENERATION_STEPS["post_generation"],
}
DROPOUT = 0.15

# Mini-eval: mid-epoch evaluation on test batches
MINI_EVAL_FREQUENCY = 1000  # Run mini-eval every N training batches
MINI_EVAL_BATCH_COUNT = 10  # Number of test batches to evaluate (capped to available)

# Used by logger for batch counting
BATCH_SIZE = 32

# Used by analysis scripts
INFERENCE_TEMPERATURE = 0
LOGS_DIR = "./logs12b"
