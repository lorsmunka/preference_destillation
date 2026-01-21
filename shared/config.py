# Centralized configuration for preference distillation project

import torch


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Model identification
MODEL_NAME = "google/gemma-3-4b-it"

# Sentence filtering (for data preparation)
MIN_SENTENCE_LENGTH = 3
MAX_SENTENCE_LENGTH = 25

# Distillation
MAX_GENERATION_STEPS = 50

# Transformer architecture
HIDDEN_DIM = 1024
NUM_LAYERS = 16
NUM_HEADS = 16
MAX_SEQ_LENGTH = MAX_SENTENCE_LENGTH + MAX_GENERATION_STEPS
DROPOUT = 0.1

# Training hyperparameters
BATCH_SIZE = 32
EPOCH_COUNT = 24
LEARNING_RATE = 6e-4
TRAINING_TEST_RATIO = 0.9

KL_RATIO_START = 0.9
KL_RATIO_END = 0.1

DISTILLATION_TEMPERATURE = 3.0
INFERENCE_TEMPERATURE = 0

# Directories
BATCHES_DIR = "./batches"
LOGS_DIR = "./logs"
CHECKPOINTS_DIR = "./checkpoints"

# File paths
TEMP_CHECKPOINT_PATH = f"{CHECKPOINTS_DIR}/temp_checkpoint.pt"
