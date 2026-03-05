# Centralized configuration for preference distillation project

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
}


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def get_input_path(domain: str) -> str:
    if domain not in INPUT_PATHS:
        raise ValueError(f"Unknown domain: {domain}")
    return INPUT_PATHS[domain]


def get_output_dir(domain: str, model_name: str) -> str:
    return os.path.join("./batches", domain, sanitize_model_name(model_name))


# Queue
DATA_GENERATION_QUEUE_PATH = "./data_generation/data-generation-queue.json"


# Sentence filtering (for data preparation)
MIN_SENTENCE_LENGTH = 3
MAX_SENTENCE_LENGTH = 25

# Distillation
MAX_GENERATION_STEPS = 50
PROMPT_DELIMITER = "\n\n"

# Transformer architecture
HIDDEN_DIM = 384
NUM_LAYERS = 18
NUM_HEADS = 8
MAX_SEQ_LENGTH = MAX_SENTENCE_LENGTH + MAX_GENERATION_STEPS
DROPOUT = 0.15

# Training hyperparameters
BATCH_SIZE = 32
EPOCH_COUNT = 4
MAX_TRAINING_EXAMPLES = 500_000
LEARNING_RATE = 6e-4
TRAINING_TEST_RATIO = 0.98

KL_RATIO_START = 1
KL_RATIO_END = 1

LR_WARMUP_RATIO = 0.03

DISTILLATION_TEMPERATURE = 3.0
INFERENCE_TEMPERATURE = 0

# Directories
BATCHES_DIR = "./batches12b"
LOGS_DIR = "./logs12b"
CHECKPOINTS_DIR = "./checkpoints"

# File paths
TEMP_CHECKPOINT_PATH = f"{CHECKPOINTS_DIR}/temp_checkpoint.pt"
