"""Paths and constants. Torch-free on purpose, so importing domain/tooling never drags in
torch — `get_device` (the only torch dependency) lives in `library.shared.device`.

Paths are anchored at the repo root (PROJECT_ROOT), not the cwd, so entry scripts work from
anywhere. Both Path-returning helpers (`run_dir`, `batches_dir`) and the legacy str-returning
ones (`get_training_run_dir`, `get_batches_dir`, `get_output_dir`) are provided.
"""

import json
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "runs"
BATCHES_DIR = PROJECT_ROOT / "batches"

MODEL_NAME = "google/gemma-3-4b-it"
DEFAULT_TEACHER_MODEL = MODEL_NAME

PROMPT_DELIMITER = "\n\n"

MIN_SENTENCE_LENGTH = 3
MAX_SENTENCE_LENGTH = 25

# Per-domain generation/sequence limits now live on the Domain itself
# (`max_steps`, `max_input_tokens`, `max_seq_length`) — no name-keyed dicts here.

# Transformer architecture defaults (overridden per run)
HIDDEN_DIM = 384
NUM_LAYERS = 18
NUM_HEADS = 8
DROPOUT = 0.15

# Mid-epoch evaluation
MINI_EVAL_FREQUENCY = 1000
MINI_EVAL_BATCH_COUNT = 10
BATCH_SIZE = 32

def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")


# Input corpora are owned by each domain (Domain.corpus_path), produced by its corpus.py.


def run_dir(run_name: str) -> Path:
    return RUNS_DIR / run_name


def get_training_run_dir(run_name: str) -> str:
    return str(run_dir(run_name))


def batches_dir(domain: str, teacher_model: str) -> Path:
    return BATCHES_DIR / domain / sanitize_model_name(teacher_model)


def get_batches_dir(domain: str, teacher_model: str) -> str:
    return str(batches_dir(domain, teacher_model))


def get_output_dir(domain: str, model_name: str) -> str:
    return str(batches_dir(domain, model_name))


def input_vocabulary_path(domain: str, teacher_model: str) -> Path:
    return batches_dir(domain, teacher_model) / "input_vocabulary.json"


def load_input_vocabulary(domain: str, teacher_model: str) -> Optional[dict]:
    path = input_vocabulary_path(domain, teacher_model)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)
