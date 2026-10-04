"""Reading and writing the distillation batches — the teacher's answers with its full
logit distribution at every generated token.

Two on-disk formats, one in-memory shape. Every reader returns a list of records

    {"sentence": str, "model_response": str,
     "steps": [{"token": str, "logits": row, "predicted_token_index": int}, ...]}

where `row` is a float32 numpy row for `.npz` files and a list of floats for legacy
`.jsonl` files — both index and convert the same way downstream.

`.npz` is the compact format: logits stored as bfloat16 bit patterns (the teacher runs in
bfloat16, so every stored value is exactly representable — the round-trip is bit-exact,
verified on every value of every domain), plus the strings as one JSON blob. Roughly 5x
smaller than the JSON text and far faster to load. When both formats exist for the same
batch index the `.npz` is used.
"""

import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

BATCH_FILE = re.compile(r"^batch_(\d+)\.(npz|jsonl)$")


def float32_to_bfloat16_bits(values: np.ndarray) -> np.ndarray:
    """float32 array -> int16 array holding the bfloat16 bit patterns (numpy has no bf16)."""
    return torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32)).to(torch.bfloat16).view(torch.int16).numpy()


def bfloat16_bits_to_float32(bits: np.ndarray) -> np.ndarray:
    return torch.from_numpy(np.ascontiguousarray(bits, dtype=np.int16)).view(torch.bfloat16).float().numpy()


def batch_index(path: Path) -> int:
    return int(BATCH_FILE.match(path.name).group(1))


def batch_paths(directory) -> List[Path]:
    """Every batch file in `directory`, sorted by index, `.npz` preferred over `.jsonl`."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    by_index: Dict[int, Path] = {}
    for path in directory.iterdir():
        match = BATCH_FILE.match(path.name)
        if not match:
            continue
        index = int(match.group(1))
        if index not in by_index or path.suffix == ".npz":
            by_index[index] = path
    return [by_index[index] for index in sorted(by_index)]


def batch_path(directory, index: int) -> Path:
    """The file for 1-based batch `index` (the number in the filename)."""
    directory = Path(directory)
    compact = directory / f"batch_{index}.npz"
    return compact if compact.exists() else directory / f"batch_{index}.jsonl"


def batch_count(directory) -> int:
    return len(batch_paths(directory))


def read_batch(path) -> List[dict]:
    path = Path(path)
    if path.suffix == ".npz":
        return _read_npz(path)
    with open(path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def read_batch_texts(path) -> List[dict]:
    """Sentences, responses and step tokens only — no logits. Cheap for `.npz`."""
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(data["meta"].item())
        return [{"sentence": sentence, "model_response": response, "tokens": tokens}
                for sentence, response, tokens in zip(meta["sentences"], meta["model_responses"], meta["tokens"])]
    return [{"sentence": record["sentence"], "model_response": record.get("model_response", ""),
             "tokens": [step["token"] for step in record.get("steps", [])]}
            for record in read_batch(path)]


def write_batch(path, records: List[dict]) -> Path:
    """Write `records` as a compact `.npz` at `path`."""
    path = Path(path)
    step_counts = [len(record["steps"]) for record in records]
    step_offsets = np.zeros(len(records) + 1, dtype=np.int64)
    step_offsets[1:] = np.cumsum(step_counts)

    all_logits = [step["logits"] for record in records for step in record["steps"]]
    logits = np.asarray(all_logits, dtype=np.float32) if all_logits else np.zeros((0, 0), dtype=np.float32)
    if logits.size and not np.isfinite(logits).all():
        raise ValueError(f"{path}: non-finite logits")

    meta = {
        "sentences": [record["sentence"] for record in records],
        "model_responses": [record.get("model_response", "") for record in records],
        "tokens": [[step["token"] for step in record["steps"]] for record in records],
        "thinking": [record.get("thinking", "") for record in records],   # teacher's reasoning trace, text only
    }
    if any(record.get("ground_truth") is not None for record in records):
        meta["ground_truths"] = [record.get("ground_truth") for record in records]   # e.g. the executed answer
        meta["answer_correct"] = [record.get("answer_correct") for record in records]   # teacher right? None = not checkable
    predicted = np.asarray([step["predicted_token_index"] for record in records for step in record["steps"]],
                           dtype=np.int32)

    np.savez(path, logits_bf16=float32_to_bfloat16_bits(logits), step_offsets=step_offsets,
             predicted_token_index=predicted, meta=np.array(json.dumps(meta)))
    return path


def _read_npz(path: Path) -> List[dict]:
    with np.load(path, allow_pickle=False) as data:
        logits = bfloat16_bits_to_float32(data["logits_bf16"])
        step_offsets = data["step_offsets"]
        predicted = data["predicted_token_index"]
        meta = json.loads(data["meta"].item())

    thinking = meta.get("thinking") or [""] * len(meta["sentences"])
    records = []
    for example_index, (sentence, response, tokens) in enumerate(
            zip(meta["sentences"], meta["model_responses"], meta["tokens"])):
        start, end = int(step_offsets[example_index]), int(step_offsets[example_index + 1])
        steps = [
            {"token": tokens[offset], "logits": logits[start + offset],
             "predicted_token_index": int(predicted[start + offset])}
            for offset in range(end - start)
        ]
        record = {"sentence": sentence, "model_response": response, "steps": steps}
        if thinking[example_index]:
            record["thinking"] = thinking[example_index]
        if meta.get("ground_truths") and meta["ground_truths"][example_index] is not None:
            record["ground_truth"] = meta["ground_truths"][example_index]
            record["answer_correct"] = meta["answer_correct"][example_index]
        records.append(record)
    return records
