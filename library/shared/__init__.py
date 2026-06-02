"""Shared infrastructure: paths/constants (config), the reduced output vocabulary
(vocabulary), the exit listener, and the metrics/ + logging/ subpackages. Torch-free at
this level — import `library.shared.device` for `get_device`.
"""

from .vocabulary import build_vocabulary, get_response_tokens, extract_logits_as_vector
from .exit_listener import ExitListener
from library.shared.config import (
    sanitize_model_name,
    get_output_dir,
    get_batches_dir,
    get_training_run_dir,
    batches_dir,
    run_dir,
    load_input_vocabulary,
    MODEL_NAME,
    DEFAULT_TEACHER_MODEL,
    MIN_SENTENCE_LENGTH,
    MAX_SENTENCE_LENGTH,
    DOMAIN_MAX_GENERATION_STEPS,
    DOMAIN_MAX_SEQ_LENGTH,
    PROMPT_DELIMITER,
    HIDDEN_DIM,
    NUM_LAYERS,
    NUM_HEADS,
    DROPOUT,
    BATCH_SIZE,
    MINI_EVAL_FREQUENCY,
    MINI_EVAL_BATCH_COUNT,
)

__all__ = [
    "build_vocabulary",
    "get_response_tokens",
    "extract_logits_as_vector",
    "ExitListener",
    "sanitize_model_name",
    "get_output_dir",
    "get_batches_dir",
    "get_training_run_dir",
    "batches_dir",
    "run_dir",
    "load_input_vocabulary",
    "MODEL_NAME",
    "DEFAULT_TEACHER_MODEL",
    "MIN_SENTENCE_LENGTH",
    "MAX_SENTENCE_LENGTH",
    "DOMAIN_MAX_GENERATION_STEPS",
    "DOMAIN_MAX_SEQ_LENGTH",
    "PROMPT_DELIMITER",
    "HIDDEN_DIM",
    "NUM_LAYERS",
    "NUM_HEADS",
    "DROPOUT",
    "BATCH_SIZE",
    "MINI_EVAL_FREQUENCY",
    "MINI_EVAL_BATCH_COUNT",
]
