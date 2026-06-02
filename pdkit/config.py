"""Filesystem paths for the project.

Paths only — no global tunables (inference temperature, a single logs dir) live here.
Those were globals in the old `shared/config.py`; in pdkit they are call arguments or
owned by a `Run`.
"""

import json
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"
BATCHES_DIR = PROJECT_ROOT / "batches"

DEFAULT_TEACHER_MODEL = "google/gemma-3-4b-it"
PROMPT_DELIMITER = "\n\n"


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def run_dir(run_name: str) -> Path:
    return RUNS_DIR / run_name


def batches_dir(domain: str, teacher_model: str) -> Path:
    return BATCHES_DIR / domain / sanitize_model_name(teacher_model)


def input_vocabulary_path(domain: str, teacher_model: str) -> Path:
    return batches_dir(domain, teacher_model) / "input_vocabulary.json"


def load_input_vocabulary(domain: str, teacher_model: str) -> Optional[dict]:
    path = input_vocabulary_path(domain, teacher_model)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)
