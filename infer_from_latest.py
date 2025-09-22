# infer_from_latest.py
import sys
import os
import json
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

# CONSTANTS
CHECKPOINT_DIR = "./checkpoints"
TOKENIZER_NAME = "google/gemma-3-4b-it"
AUX_WORDS = ["the", "a", "is", "of", "and", "to", "in", "that", "it",
             "you", "very", "quite", "somewhat", "extremely", "slightly", "moderately"]
MAX_GEN_CHARS = 200
MAX_GEN_TOKENS = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def find_latest_checkpoint(dirpath):
    ckpts = sorted(Path(dirpath).glob("ckpt_*.pt"))
    return str(ckpts[-1]) if ckpts else None


def load_tokenizer_and_model(ckpt_path):
    tk = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
    if tk.pad_token is None:
        tk.add_special_tokens({"pad_token": "<|pad|>"})
    st = torch.load(ckpt_path, map_location="cpu")
    cfg_dict = st.get("config")
    if cfg_dict is None:
        raise RuntimeError("checkpoint missing config")
    cfg = GPT2Config(**cfg_dict)
    model = GPT2LMHeadModel(cfg)
    model.load_state_dict(st["model_state"])
    model.resize_token_embeddings(len(tk))
    model.to(DEVICE)
    model.eval()
    return tk, model


def build_forbidden_token_ids(tokenizer, aux_words):
    forbidden = set()
    for w in aux_words:
        for variant in (w, " " + w):
            try:
                ids = tokenizer.encode(variant, add_special_tokens=False)
            except Exception:
                ids = []
            for i in ids:
                forbidden.add(i)
    # also exclude tokenizer.pad_token_id if present
    if tokenizer.pad_token_id is not None:
        forbidden.add(tokenizer.pad_token_id)
    return forbidden


def make_prompt(sentence):
    return (
        f'Analyze the following sentence and return a JSON object with these evaluations:\n\n'
        f'    Sentence: "{sentence}"\n\n'
        f'    Return JSON format:\n'
        f'    {{\n'
        f'        "tone": "aggressive | rude | neutral | polite | friendly",\n'
        f'        "sentiment": "negative | neutral | positive",\n'
        f'        "safety": "harmful | safe",\n'
        f'        "toxicity": "toxic | respectful"\n'
        f'    }}\n\nJSON:\n'
    )


def generate_json_like(prompt, tokenizer, model, forbidden_ids):
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(DEVICE)
    generated_ids = input_ids.clone()
    generated_chars = ""
    # we will only consider produced tokens after the prompt length
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        for step in range(MAX_GEN_TOKENS):
            outputs = model(input_ids=generated_ids)
            logits = outputs.logits[0, -1, :]  # (vocab,)
            sorted_ids = torch.argsort(logits, descending=True).cpu().tolist()

            chosen_id = None
            for tid in sorted_ids:
                if tid in forbidden_ids:
                    continue
                chosen_id = tid
                break
            if chosen_id is None:
                chosen_id = sorted_ids[0]

            new_id_tensor = torch.tensor([[chosen_id]], device=DEVICE)
            generated_ids = torch.cat([generated_ids, new_id_tensor], dim=1)

            token_str = tokenizer.decode(
                [chosen_id], clean_up_tokenization_spaces=False)
            generated_chars += token_str

            if "}" in token_str:
                # cut everything after first '}' for neatness
                idx = generated_chars.find("}") + 1
                generated_chars = generated_chars[:idx]
                break
            if len(generated_chars) >= MAX_GEN_CHARS:
                generated_chars = generated_chars[:MAX_GEN_CHARS]
                break

    # trim leading whitespace/newlines and return
    return generated_chars.lstrip()


def main():
    if len(sys.argv) > 1:
        sentence = " ".join(sys.argv[1:]).strip()
    else:
        sentence = input("Enter sentence: ").strip()
    if not sentence:
        print("No sentence provided.")
        return

    ckpt = find_latest_checkpoint(CHECKPOINT_DIR)
    if not ckpt:
        print("No checkpoint found in", CHECKPOINT_DIR)
        return

    tokenizer, model = load_tokenizer_and_model(ckpt)
    forbidden_ids = build_forbidden_token_ids(tokenizer, AUX_WORDS)

    prompt = make_prompt(sentence)
    start_ts = time.time()
    out = generate_json_like(prompt, tokenizer, model, forbidden_ids)
    dur = time.time() - start_ts

    print("\n=== GENERATED (raw) ===\n")
    print(out)
    print("\n=== METADATA ===")
    print(f"checkpoint: {ckpt}")
    print(f"device: {DEVICE} gen_time_s: {dur:.3f}")


if __name__ == "__main__":
    main()
