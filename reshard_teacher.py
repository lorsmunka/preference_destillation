"""Re-shard a single-file safetensors checkpoint into 2 GB pieces, once.

    python reshard_teacher.py                          # google/gemma-4-12B-it -> models/gemma-4-12B-it
    python reshard_teacher.py google/other-model       # another HF id

Why: a single 24 GB shard cannot be memory-mapped on this machine (Windows commit limit,
error 1455). Reading tensor by tensor with safetensors' safe_open needs no such commit, and
transformers loads small shards one at a time. Config, tokenizer and chat template are
copied alongside, so the output directory is a complete local model; ModelHandler resolves
the HF id to it through TEACHER_LOCAL_PATHS in library/shared/config.py.
"""

import json
import shutil
import sys
from pathlib import Path

import struct

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file

MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "google/gemma-4-12B-it"
SHARD_BYTES = 2 * 1024 ** 3
OUTPUT_DIR = Path("models") / MODEL_ID.split("/")[-1]

SAFETENSORS_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64,
                      "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8,
                      "U8": torch.uint8, "BOOL": torch.bool}


def iter_tensors(path: Path):
    """Yield (name, tensor) from a safetensors file with plain reads — no memory mapping, so a
    single 24 GB file works on a machine whose commit limit cannot map it (Windows error 1455 /
    segfault in safe_open). Layout: 8-byte little-endian header length, JSON header, raw data."""
    with open(path, "rb") as file:
        header_length = struct.unpack("<Q", file.read(8))[0]
        header = json.loads(file.read(header_length))
        data_start = 8 + header_length
        entries = [(name, info) for name, info in header.items() if name != "__metadata__"]
        entries.sort(key=lambda item: item[1]["data_offsets"][0])
        for name, info in entries:
            start, end = info["data_offsets"]
            file.seek(data_start + start)
            raw = file.read(end - start)
            dtype = SAFETENSORS_DTYPES[info["dtype"]]
            tensor = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(info["shape"]) if raw else torch.empty(info["shape"], dtype=dtype)
            yield name, tensor


def main():
    source = Path(snapshot_download(MODEL_ID))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file() and not path.name.endswith(".safetensors"):
            shutil.copy(path, OUTPUT_DIR / path.name)

    weight_files = sorted(source.glob("*.safetensors"))
    index = {"metadata": {"total_size": 0}, "weight_map": {}}
    shard, shard_bytes, shard_number = {}, 0, 0

    def flush():
        nonlocal shard, shard_bytes, shard_number
        if not shard:
            return
        shard_number += 1
        name = f"model-{shard_number:05d}.safetensors"
        save_file(shard, OUTPUT_DIR / name, metadata={"format": "pt"})
        for key in shard:
            index["weight_map"][key] = name
        print(f"  wrote {name}  ({shard_bytes / 1e9:.2f} GB, {len(shard)} tensors)")
        shard, shard_bytes = {}, 0

    for weight_file in weight_files:
        for key, tensor in iter_tensors(weight_file):
            size = tensor.numel() * tensor.element_size()
            if shard_bytes + size > SHARD_BYTES and shard:
                flush()
            shard[key] = tensor
            shard_bytes += size
            index["metadata"]["total_size"] += size
    flush()

    # rename to the conventional "model-00001-of-0000N" and write the index
    total = shard_number
    for number in range(1, total + 1):
        old = OUTPUT_DIR / f"model-{number:05d}.safetensors"
        new = OUTPUT_DIR / f"model-{number:05d}-of-{total:05d}.safetensors"
        old.rename(new)
    index["weight_map"] = {k: v.replace(".safetensors", f"-of-{total:05d}.safetensors") for k, v in index["weight_map"].items()}
    with open(OUTPUT_DIR / "model.safetensors.index.json", "w", encoding="utf-8") as file:
        json.dump(index, file, indent=1)
    print(f"\n{MODEL_ID} -> {OUTPUT_DIR}: {total} shards, {index['metadata']['total_size'] / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
