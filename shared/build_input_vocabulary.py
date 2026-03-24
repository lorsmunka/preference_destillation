"""
Scans distillation batches to build a reduced input vocabulary.

Collects every unique Gemma token ID that appears during training:
  - Sentence tokens (tokenized as whole strings, same as trainer)
  - Step tokens (tokenized individually, same as trainer)
  - Prompt delimiter tokens

Outputs a JSON mapping file: { gemma_token_id -> compact_id }
"""

import json
import os
import glob
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.config import get_batches_dir, PROMPT_DELIMITER, MODEL_NAME


WORKER_COUNT = 8


def load_chunk(worker_index, file_chunk):
    sentences = set()
    step_tokens = set()
    total = len(file_chunk)

    for file_index, filepath in enumerate(file_chunk):
        with open(filepath, "r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                example = json.loads(line)
                sentences.add(example['sentence'] + PROMPT_DELIMITER)
                for step in example.get('steps', []):
                    step_tokens.add(step['token'])

        if (file_index + 1) % 100 == 0:
            percentage = (file_index + 1) / total * 100
            print(f"  Worker {worker_index}: {percentage:5.1f}% ({file_index + 1}/{total})")

    print(f"  Worker {worker_index}: done — {len(sentences):,} sentences, {len(step_tokens):,} step tokens")
    return sentences, step_tokens


def scan_batches(batches_directory, tokenizer):
    batch_files = glob.glob(os.path.join(batches_directory, "batch_*.jsonl"))

    if not batch_files:
        print(f"No batch files found in {batches_directory}")
        return set()

    # Phase 1: Load all files concurrently, each worker gets an equal chunk
    chunk_size = (len(batch_files) + WORKER_COUNT - 1) // WORKER_COUNT
    chunks = [batch_files[i:i + chunk_size] for i in range(0, len(batch_files), chunk_size)]

    print(f"Phase 1: Loading {len(batch_files)} batch files ({len(chunks)} workers, ~{chunk_size} files each)...")
    unique_sentences = set()
    unique_step_tokens = set()

    with ProcessPoolExecutor(max_workers=WORKER_COUNT) as executor:
        futures = [
            executor.submit(load_chunk, worker_index, chunk)
            for worker_index, chunk in enumerate(chunks)
        ]

        for future in as_completed(futures):
            sentences, step_tokens = future.result()
            unique_sentences.update(sentences)
            unique_step_tokens.update(step_tokens)

    print(f"  All workers done — {len(unique_sentences):,} unique sentences, {len(unique_step_tokens):,} unique step tokens")

    # Phase 2: Tokenize unique strings only
    print(f"Phase 2: Tokenizing {len(unique_sentences) + len(unique_step_tokens):,} unique strings...")
    unique_token_ids = set()

    for index, sentence in enumerate(unique_sentences):
        token_ids = tokenizer.encode(sentence, add_special_tokens=False)
        unique_token_ids.update(token_ids)

        if (index + 1) % 10000 == 0:
            print(f"  Sentences: {index + 1}/{len(unique_sentences)}")

    for step_token in unique_step_tokens:
        token_ids = tokenizer.encode(step_token, add_special_tokens=False)
        unique_token_ids.update(token_ids)

    print(f"  Done — {len(unique_token_ids):,} unique token IDs")

    return unique_token_ids


def build_input_vocabulary(domain, teacher_model=MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(teacher_model)
    full_vocab_size = tokenizer.vocab_size

    batches_directory = get_batches_dir(domain, teacher_model)
    unique_token_ids = scan_batches(batches_directory, tokenizer)

    if not unique_token_ids:
        print("No tokens found. Check that batch files exist.")
        return

    sorted_token_ids = sorted(unique_token_ids)
    gemma_id_to_compact_id = {gemma_id: compact_id for compact_id, gemma_id in enumerate(sorted_token_ids)}

    output_directory = batches_directory
    output_path = os.path.join(output_directory, "input_vocabulary.json")

    vocabulary_data = {
        "domain": domain,
        "teacher_model": teacher_model,
        "full_vocab_size": full_vocab_size,
        "compact_vocab_size": len(sorted_token_ids),
        "gemma_id_to_compact_id": gemma_id_to_compact_id,
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(vocabulary_data, file, indent=2)

    reduction_percentage = (1 - len(sorted_token_ids) / full_vocab_size) * 100

    print(f"\nInput vocabulary built:")
    print(f"  Domain: {domain}")
    print(f"  Teacher: {teacher_model}")
    print(f"  Full vocab: {full_vocab_size:,} tokens")
    print(f"  Input vocab: {len(sorted_token_ids):,} tokens")
    print(f"  Reduction: {reduction_percentage:.1f}%")
    print(f"  Saved to: {output_path}")

    return vocabulary_data


def discover_domains(batches_root="./batches"):
    domains = []
    if not os.path.isdir(batches_root):
        return domains
    for domain in os.listdir(batches_root):
        domain_path = os.path.join(batches_root, domain)
        if not os.path.isdir(domain_path):
            continue
        for teacher_dir in os.listdir(domain_path):
            teacher_path = os.path.join(domain_path, teacher_dir)
            if not os.path.isdir(teacher_path):
                continue
            if glob.glob(os.path.join(teacher_path, "batch_*.jsonl")):
                teacher_model = teacher_dir.replace("_", "/", 1)
                domains.append((domain, teacher_model))
    return domains


if __name__ == "__main__":
    found = discover_domains()

    if not found:
        print("No batch directories found.")
        sys.exit(1)

    print(f"Found {len(found)} domain/teacher combinations:\n")
    for domain, teacher_model in found:
        print(f"  {domain} / {teacher_model}")
    print()

    for domain, teacher_model in found:
        print(f"{'=' * 60}")
        print(f"Building input vocabulary for domain='{domain}', teacher='{teacher_model}'\n")
        build_input_vocabulary(domain, teacher_model)
        print()
