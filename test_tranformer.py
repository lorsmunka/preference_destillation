# train_transformer_final.py
import os
import json
import time
import random
import math
import inspect
from pathlib import Path
from datetime import datetime
from contextlib import nullcontext

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, get_linear_schedule_with_warmup

# CONSTANTS
RUN_NAME = "small_gpt_distill"
DATA_DIR = "./training_data"
TOKENIZER_NAME = "google/gemma-3-4b-it"
CHECKPOINT_DIR = "./checkpoints"
TELEMETRY_DIR = "./telemetry"

# model target (change to adjust model size)
TARGET_PARAM_COUNT = 80_000_000

# default model grid (used if auto-selection fails)
DEFAULT_N_LAYERS = 10
DEFAULT_N_HEADS = 10
DEFAULT_EMBED = 640

MAX_SEQ_LEN = 512

NUM_EPOCHS = 30
PER_DEVICE_BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 200
MAX_GRAD_NORM = 1.0
SAVE_EVERY_STEPS = 500
LOG_EVERY_STEPS = 20
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0 if os.name == "nt" else 2

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TELEMETRY_DIR, exist_ok=True)


def now_tag(): return datetime.now().strftime("%Y%m%d_%H%M%S")


def set_seed(s):
    random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def estimate_params(n_layers, embed_dim, vocab_size, n_positions):
    embed_matrix = vocab_size * embed_dim
    per_layer = 12 * (embed_dim ** 2)
    return embed_matrix + n_layers * per_layer


def humanize(n):
    for u in ["", "K", "M", "B"]:
        if abs(n) < 1000.0:
            return f"{n:3.1f}{u}"
        n /= 1000.0
    return f"{n:.1f}T"


def recommend_configs(target, vocab_size, heads, max_layers=24, embed_candidates=None):
    if embed_candidates is None:
        embed_candidates = [256, 320, 384, 512, 576, 640, 704, 768, 832, 1024]
    good = []
    for embed in embed_candidates:
        if embed % heads != 0:
            continue
        for layers in range(2, max_layers+1):
            p = estimate_params(layers, embed, vocab_size, MAX_SEQ_LEN)
            if p <= target:
                good.append((layers, embed, p))
    good.sort(key=lambda x: (abs(x[2]-target), -x[2]))
    return good[:6]


class DistillJsonlDataset(Dataset):
    def __init__(self, files, tokenizer, max_len=MAX_SEQ_LEN):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []
        for p in files:
            with open(p, "r", encoding="utf-8") as fh:
                for ln in fh:
                    try:
                        obj = json.loads(ln.strip())
                        s = obj.get("sentence", "")
                        g = obj.get("generated_response", "")
                        prompt = (
                            f'Analyze the following sentence and return a JSON object with these evaluations:\n\n'
                            f'    Sentence: "{s}"\n\n'
                            f'    Return JSON format:\n'
                            f'    {{\n'
                            f'        "tone": "aggressive | rude | neutral | polite | friendly",\n'
                            f'        "sentiment": "negative | neutral | positive",\n'
                            f'        "safety": "harmful | safe",\n'
                            f'        "toxicity": "toxic | respectful"\n'
                            f'    }}\n\nJSON:\n'
                        )
                        self.examples.append(prompt + g)
                    except Exception:
                        continue

    def __len__(self): return len(self.examples)

    def __getitem__(self, idx):
        t = self.examples[idx]
        toks = self.tokenizer(t, truncation=True,
                              max_length=self.max_len, return_tensors="pt")
        return toks["input_ids"].squeeze(0), toks["attention_mask"].squeeze(0)


def collate_batch(batch):
    input_ids = [b[0] for b in batch]
    attn = [b[1] for b in batch]
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad)
    attn = pad_sequence(attn, batch_first=True, padding_value=0)
    labels = input_ids.clone()
    labels[input_ids == pad] = -100
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


def build_tokenizer(name=TOKENIZER_NAME):
    tk = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tk.pad_token is None:
        tk.add_special_tokens({"pad_token": "<|pad|>"})
    return tk


def build_model(tokenizer, n_layers, n_heads, embed_dim):
    cfg = GPT2Config(
        vocab_size=len(tokenizer), n_positions=MAX_SEQ_LEN, n_ctx=MAX_SEQ_LEN,
        n_embd=embed_dim, n_layer=n_layers, n_head=n_heads,
        bos_token_id=tokenizer.bos_token_id or 0, eos_token_id=tokenizer.eos_token_id or 1
    )
    m = GPT2LMHeadModel(cfg)
    m.resize_token_embeddings(len(tokenizer))
    m.to(DEVICE)
    return m, cfg


def latest_checkpoint(path):
    ckpts = sorted(Path(path).glob("ckpt_*.pt"))
    return str(ckpts[-1]) if ckpts else None


def save_checkpoint(path, st): torch.save(st, path)


def load_checkpoint(path, model, opt=None, sched=None):
    st = torch.load(path, map_location=DEVICE)
    model.load_state_dict(st["model_state"])
    if opt is not None and "optim_state" in st:
        opt.load_state_dict(st["optim_state"])
    if sched is not None and "sched_state" in st:
        sched.load_state_dict(st["sched_state"])
    return st


def choose_autocast(device):
    if device != "cuda":
        return nullcontext
    # prefer torch.cuda.amp.autocast
    if hasattr(torch, "cuda") and hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
        return torch.cuda.amp.autocast
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        sig = inspect.signature(torch.amp.autocast)
        if "device_type" in sig.parameters:
            return lambda enabled: torch.amp.autocast(device_type="cuda", enabled=enabled)
        return lambda enabled: torch.amp.autocast(enabled=enabled)
    return torch.cuda.amp.autocast


def create_grad_scaler(device):
    if device != "cuda":
        return None
    for ctor in (getattr(torch, "amp", None) and getattr(torch.amp, "GradScaler", None),
                 getattr(torch, "cuda", None) and getattr(
                     torch.cuda, "amp", None) and getattr(torch.cuda.amp, "GradScaler", None),
                 getattr(torch, "cuda", None) and getattr(torch.cuda, "amp", None) and getattr(torch.cuda.amp, "GradScaler", None)):
        try:
            if ctor is not None:
                return ctor()
        except Exception:
            continue
    return None


def train():
    set_seed(SEED)
    files = sorted(Path(DATA_DIR).glob("training_data_batch_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No training files in {DATA_DIR}")

    global tokenizer
    tokenizer = build_tokenizer()
    vocab = len(tokenizer)

    # choose model config to approach TARGET_PARAM_COUNT
    recs = recommend_configs(TARGET_PARAM_COUNT, vocab,
                             DEFAULT_N_HEADS, max_layers=32)
    if recs:
        n_layers, embed_dim, est = recs[0]
    else:
        n_layers, embed_dim = DEFAULT_N_LAYERS, DEFAULT_EMBED
        est = estimate_params(n_layers, embed_dim, vocab, MAX_SEQ_LEN)

    model, cfg = build_model(tokenizer, n_layers, DEFAULT_N_HEADS, embed_dim)

    dataset = DistillJsonlDataset(files, tokenizer)
    dataloader = DataLoader(dataset, batch_size=PER_DEVICE_BATCH_SIZE, shuffle=True,
                            collate_fn=collate_batch, num_workers=NUM_WORKERS, drop_last=False)

    steps_per_epoch = max(1, math.ceil(len(dataloader) / GRAD_ACCUM_STEPS))
    total_steps = steps_per_epoch * NUM_EPOCHS

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE,
                      weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)

    ckpt = latest_checkpoint(CHECKPOINT_DIR)
    start_epoch = 0
    global_step = 0
    if ckpt:
        st = load_checkpoint(ckpt, model, opt=optimizer, sched=scheduler)
        start_epoch = st.get("epoch", 0)
        global_step = st.get("global_step", 0)

    run_tag = f"{RUN_NAME}_{now_tag()}"
    tele_json = Path(TELEMETRY_DIR) / f"{run_tag}.jsonl"
    tele_txt = Path(TELEMETRY_DIR) / f"{run_tag}.txt"

    scaler = create_grad_scaler(DEVICE)
    autocast_ctx = choose_autocast(DEVICE)
    use_amp = (scaler is not None) and (DEVICE == "cuda")

    print(f"RUN {run_tag} DEVICE={DEVICE} WORKERS={NUM_WORKERS} VOCAB={vocab}")
    print(
        f"SELECTED CONFIG layers={n_layers} heads={DEFAULT_N_HEADS} embed={embed_dim} EST_PARAMS≈{humanize(est)}")
    print("OTHER RECOMMENDATIONS:")
    for r in recommend_configs(TARGET_PARAM_COUNT, vocab, DEFAULT_N_HEADS, max_layers=32):
        print(f"  layers={r[0]} embed={r[1]} params≈{humanize(r[2])}")

    model.train()
    start_time = time.time()

    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_loss = 0.0
        steps_in_epoch = 0
        for step_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            with autocast_ctx(use_amp):
                out = model(input_ids=input_ids,
                            attention_mask=attention_mask, labels=labels)
                loss = out.loss / GRAD_ACCUM_STEPS

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step_idx + 1) % GRAD_ACCUM_STEPS == 0 or (step_idx + 1) == len(dataloader):
                try:
                    if scaler is not None and hasattr(scaler, "unscale_"):
                        scaler.unscale_(optimizer)
                except Exception:
                    pass

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), MAX_GRAD_NORM)

                if scaler is not None:
                    try:
                        scaler.step(optimizer)
                        scaler.update()
                    except Exception:
                        optimizer.step()
                else:
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                raw_loss = (
                    loss.item() * GRAD_ACCUM_STEPS) if hasattr(loss, "item") else float(loss)
                epoch_loss += raw_loss
                steps_in_epoch += 1

                with open(tele_json, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"ts": time.time(), "epoch": epoch, "global_step": global_step,
                             "loss": raw_loss, "lr": optimizer.param_groups[0]["lr"]}, ensure_ascii=False) + "\n")

                if global_step % LOG_EVERY_STEPS == 0:
                    elapsed_m = (time.time()-start_time)/60.0
                    ppl = math.exp(raw_loss) if raw_loss < 20 else float("inf")
                    print(
                        f"{run_tag} epoch {epoch+1}/{NUM_EPOCHS} step {global_step} loss={raw_loss:.4f} ppl={ppl:.2f} elapsed={elapsed_m:.1f}m")

                if global_step % SAVE_EVERY_STEPS == 0:
                    path = Path(CHECKPOINT_DIR) / \
                        f"ckpt_epoch{epoch}_step{global_step}.pt"
                    save_checkpoint(str(path), {"model_state": model.state_dict(), "optim_state": optimizer.state_dict(), "sched_state": scheduler.state_dict(
                    ), "epoch": epoch, "global_step": global_step, "tokenizer": TOKENIZER_NAME, "config": cfg.to_dict()})
                    with open(tele_txt, "a", encoding="utf-8") as fh:
                        fh.write(
                            f"Saved checkpoint {path} at step {global_step}\n")

        avg_loss = epoch_loss / max(1, steps_in_epoch)
        with open(tele_txt, "a", encoding="utf-8") as fh:
            fh.write(
                f"Epoch {epoch+1} done. avg_loss={avg_loss:.6f} steps={steps_in_epoch} time={now_tag()}\n")
        epoch_ckpt = Path(CHECKPOINT_DIR) / \
            f"ckpt_epoch{epoch+1}_step{global_step}.pt"
        save_checkpoint(str(epoch_ckpt), {"model_state": model.state_dict(), "optim_state": optimizer.state_dict(
        ), "sched_state": scheduler.state_dict(), "epoch": epoch+1, "global_step": global_step, "tokenizer": TOKENIZER_NAME, "config": cfg.to_dict()})
        with open(tele_txt, "a", encoding="utf-8") as fh:
            fh.write(f"Saved epoch checkpoint {epoch_ckpt}\n")

    final_path = Path(CHECKPOINT_DIR) / f"ckpt_final_{run_tag}.pt"
    save_checkpoint(str(final_path), {"model_state": model.state_dict(), "optim_state": optimizer.state_dict(), "sched_state": scheduler.state_dict(
    ), "epoch": NUM_EPOCHS, "global_step": global_step, "tokenizer": TOKENIZER_NAME, "config": cfg.to_dict()})
    with open(tele_txt, "a", encoding="utf-8") as fh:
        fh.write(
            f"Training finished. final_checkpoint={final_path} total_steps={global_step} total_time_s={time.time()-start_time:.1f}\n")
    print("training complete")


if __name__ == "__main__":
    train()
