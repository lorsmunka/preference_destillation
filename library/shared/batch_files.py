"""Reading and writing the distillation batches — the teacher's answers with its full
logit distribution at every generated token.

Two on-disk formats, one in-memory shape. Every reader returns a list of records

    {"sentence": str, "model_response": str,
     "steps": [{"token": str, "logits": row, "predicted_token_index": int}, ...],
     "split": ..., "item_id": ..., "turn_type": ..., "teacher_mode": ...}

where `row` is a float32 numpy row for `.npz` files and a list of floats for legacy
`.jsonl` files — both index and convert the same way downstream. The four metadata keys
come from the corpus record the example was made from (`teacher_mode` is "generate" or
"forced"); they are always present and None when the batch predates them or the corpus
record had no such field.

`.npz` is the compact format: logits stored as bfloat16 bit patterns (the teacher runs in
bfloat16, so every stored value is exactly representable — the round-trip is bit-exact,
verified on every value of every domain), plus the strings as one JSON blob. Roughly 5x
smaller than the JSON text and far faster to load. When both formats exist for the same
batch index the `.npz` is used. The JSON blob also carries `vocabulary_sha1`, the
fingerprint of the reduced vocabulary whose tokens the logit columns are (see
`vocabulary_fingerprint`), so a reader can check the columns still mean the same tokens.
"""

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

BATCH_FILE = re.compile(r"^batch_(\d+)\.(npz|jsonl)$")

# Per-example fields carried from the corpus record into the batch (None when absent). "group" is a
# corpus's own flag for where a record came from (e.g. "original" / "new" / "copy"), for training on a part.
RECORD_METADATA_KEYS = ("split", "item_id", "turn_type", "teacher_mode", "group")


def vocabulary_fingerprint(token_list: List[str]) -> str:
    """sha1 of the reduced vocabulary's ordered token list: the identity of the logit columns.
    Any reordering, addition or removal of a token changes it."""
    return hashlib.sha1("\n".join(token_list).encode("utf-8")).hexdigest()


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
        records = [json.loads(line) for line in file if line.strip()]
    for record in records:
        for key in RECORD_METADATA_KEYS:
            record.setdefault(key, None)
    return records


def read_batch_meta(path) -> dict:
    """The JSON blob of an `.npz` batch (e.g. its "vocabulary_sha1"), without decoding the
    logits. A legacy `.jsonl` batch has none: {}."""
    path = Path(path)
    if path.suffix != ".npz":
        return {}
    with np.load(path, allow_pickle=False) as data:
        return json.loads(data["meta"].item())


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


def write_batch(path, records: List[dict], vocabulary_sha1: Optional[str] = None) -> Path:
    """Write `records` as a compact `.npz` at `path`. `vocabulary_sha1` names the reduced
    vocabulary the logit columns belong to. The file appears complete or not at all: it is
    written under a temporary name and then renamed, so a crash never leaves half a batch."""
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
    if any(record.get("teacher_agreement") is not None for record in records):
        meta["teacher_agreement"] = [record.get("teacher_agreement") for record in records]   # teacher-forced: argmax == gold share
    if any(record.get("ground_truth") is not None for record in records):
        meta["ground_truths"] = [record.get("ground_truth") for record in records]   # e.g. the executed answer
        meta["answer_correct"] = [record.get("answer_correct") for record in records]   # teacher right? None = not checkable
    for key in RECORD_METADATA_KEYS:
        if any(record.get(key) is not None for record in records):
            meta[key] = [record.get(key) for record in records]
    if vocabulary_sha1 is not None:
        meta["vocabulary_sha1"] = vocabulary_sha1
    predicted = np.asarray([step["predicted_token_index"] for record in records for step in record["steps"]],
                           dtype=np.int32)

    temporary_path = path.with_name(path.name + ".partial")   # does not match BATCH_FILE, so never read as a batch
    with open(temporary_path, "wb") as file:
        np.savez(file, logits_bf16=float32_to_bfloat16_bits(logits), step_offsets=step_offsets,
                 predicted_token_index=predicted, meta=np.array(json.dumps(meta)))
    os.replace(temporary_path, path)
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
        if meta.get("teacher_agreement") and meta["teacher_agreement"][example_index] is not None:
            record["teacher_agreement"] = meta["teacher_agreement"][example_index]
        if meta.get("ground_truths") and meta["ground_truths"][example_index] is not None:
            record["ground_truth"] = meta["ground_truths"][example_index]
            record["answer_correct"] = meta["answer_correct"][example_index]
        for key in RECORD_METADATA_KEYS:
            record[key] = meta[key][example_index] if key in meta else None
        records.append(record)
    return records
