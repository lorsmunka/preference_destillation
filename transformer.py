"""
transformer.py

Complete, self-contained transformer training script tailored for distilling a small
decoder-only model (~100M params excluding Gemma embeddings). Key features:
 - Config object (no CLI required) with sensible defaults (20 epochs)
 - Streaming sharded dataset (never load all files into memory)
 - Deterministic 10% test split + deterministic 10% train-check split via stable hashing
 - Reduced output vocabulary mapping (watched tokens + __OTHER__ bucket)
 - Tiny decoder-only Transformer (combined QKV linear, pre-LN blocks)
 - Resumable checkpoints and mixed-precision (AMP) training
 - After each epoch: evaluate on (a) reserved test set and (b) 10% held-out training subset

Usage: python transformer.py

Edit Config below for paths, tokenizer name, device, or training hyperparams.
"""

import os
import math
import json
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer

# ----------------------------- CONFIG -----------------------------


class Config:
    # Tokenizer (Gemma tokenizer recommended)
    TOKENIZER_NAME = "google/gemma-3-4b-it"

    # Device: 'cuda' or 'cpu'
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Training schedule
    EPOCHS = 20
    GRADIENT_ACCUMULATION_STEPS = 16
    BATCH_SIZE_GPU = 1  # per-GPU micro-batch
    MAX_SEQ_LEN = 512

    # Model size (these follow the ~100M param plan)
    N_LAYERS = 12
    D_MODEL = 832
    N_HEADS = 13
    D_FF = 3328  # 4 * D_MODEL
    VOCAB_OUT = 256  # reduced output head size (includes __OTHER__)
    DROPOUT = 0.1

    # Data
    TRAINING_GLOB = "./training_data/training_data_batch_*.jsonl"
    TEST_PERCENT = 10  # deterministic test partition (10%)
    # deterministic held-out from train for overfit checks (10%)
    TRAIN_CHECK_PERCENT = 10

    # Checkpoint
    CKPT_DIR = Path("./checkpoints")
    SAVE_EVERY_EPOCHS = 1

    # Eval
    MAX_EVAL_BATCHES = 200  # limit per-eval to keep runtime reasonable

# ------------------------- Utilities -------------------------


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def stable_hash_to_percent(s: str) -> int:
    # stable hashing using md5; returns 0..99
    h = hashlib.md5(s.encode('utf-8')).hexdigest()
    return int(h, 16) % 100

# ------------------------- Tokenizer & reduced vocab -------------------------


class TokenizerWrapper:
    def __init__(self, hf_tokenizer_name: str, reduced_tokens: Optional[List[str]] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_name)
        # default reduced tokens drawn from your 'watched token' list and JSON syntax
        if reduced_tokens is None:
            reduced_tokens = self._default_reduced_tokens()
        self.reduced_tokens = reduced_tokens
        self.reduced_map = self._build_reduced_map(reduced_tokens)
        self.other_index = len(self.reduced_tokens)

    def _default_reduced_tokens(self) -> List[str]:
        structural = ['```', 'json', '{', '}', '"', ':', ',', '\n']
        fields = ['tone', 'sentiment', 'safety', 'toxicity']
        values = ['neutral', 'aggressive', 'rude', 'polite',
                  'friendly', 'toxic', 'safe', 'positive', 'negative']
        auxiliaries = ['the', 'a', 'is', 'of', 'and',
                       'to', 'very', 'quite', 'extremely']
        return list(dict.fromkeys(structural + fields + values + auxiliaries))

    def _build_reduced_map(self, reduced_tokens: List[str]) -> Dict[int, int]:
        mapping: Dict[int, int] = {}
        for i, tok in enumerate(reduced_tokens):
            # tokenizer.convert_tokens_to_ids may fail for strings not in the tokenizer vocabulary
            tids = self.tokenizer.encode(tok, add_special_tokens=False)
            if len(tids) == 1:
                mapping[tids[0]] = i
            else:
                # If multiple token ids map, prefer the first (best-effort)
                for t in tids:
                    mapping[t] = i
        return mapping

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def reduced_label_for_token_id(self, tok_id: int) -> int:
        return self.reduced_map.get(tok_id, self.other_index)

    def reduced_vocab_size(self) -> int:
        return len(self.reduced_tokens) + 1

# ------------------------- Streaming Dataset -------------------------


class ShardedJSONLines(IterableDataset):
    """
    Streams JSONL files matching a glob. Each line must contain 'sentence' and 'generated_response'.

    Deterministically assigns each example to one of: 'test', 'train_check', 'train' based on stable hash of the sentence.
    This avoids the need for external metadata and keeps the 10% splits stable across runs while accepting new files.
    """

    def __init__(self, glob_pattern: str, tokenizer: TokenizerWrapper, mode: str = 'train', max_seq_len: int = 512):
        assert mode in ('train', 'test', 'train_check')
        self.glob_pattern = glob_pattern
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_seq_len = max_seq_len

    def _file_list(self):
        import glob
        return sorted(glob.glob(self.glob_pattern))

    def __iter__(self):
        for fname in self._file_list():
            with open(fname, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        # prefer 'sentence' as the text to hash; fallback to entire line
                        key = obj.get('sentence') or line
                        p = stable_hash_to_percent(key)
                        if p < Config.TEST_PERCENT:
                            assigned = 'test'
                        elif p < (Config.TEST_PERCENT + Config.TRAIN_CHECK_PERCENT):
                            assigned = 'train_check'
                        else:
                            assigned = 'train'
                        if assigned != self.mode:
                            continue
                        # tokenize on the fly
                        input_ids = self.tokenizer.encode(obj['sentence'])
                        target_ids = self.tokenizer.encode(
                            obj['generated_response'])
                        # simple truncation strategy — drop tokens from the beginning of the input if needed
                        available = self.max_seq_len
                        if len(input_ids) + len(target_ids) > available:
                            # keep tail of input and head of target
                            keep_input = max(0, available - len(target_ids))
                            input_ids = input_ids[-keep_input:]
                            input_ids = input_ids[:available - len(target_ids)]
                            # target trimmed to fit
                            target_ids = target_ids[:available -
                                                    len(input_ids)]
                        yield {
                            'input_ids': input_ids,
                            'target_ids': target_ids
                        }
                    except Exception:
                        continue

# ------------------------- Collator -------------------------


class Collator:
    def __init__(self, tokenizer: TokenizerWrapper, max_seq_len: int = 512):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        # Each example: concat input + target; labels only for target positions mapped to reduced vocab indices
        B = len(batch)
        seqs = []
        labels = []
        for ex in batch:
            inp = ex['input_ids']
            tgt = ex['target_ids']
            seq = inp + tgt
            if len(seq) > self.max_seq_len:
                seq = seq[-self.max_seq_len:]
                # If truncated, we must also adjust labels accordingly (assume target still at end)
            # build labels: -100 for input positions, reduced indices for target positions
            n_input = max(0, len(seq) - len(tgt))
            lab = [-100] * n_input
            # map each token id in the target slice to reduced index
            for tok in seq[n_input:]:
                lab.append(self.tokenizer.reduced_label_for_token_id(tok))
            # pad
            pad_len = self.max_seq_len - len(seq)
            seq = seq + [0] * pad_len
            lab = lab + [-100] * pad_len
            seqs.append(seq)
            labels.append(lab)
        input_ids = torch.tensor(seqs, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        attention_mask = (input_ids != 0).long()
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }

# ------------------------- Model -------------------------


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # Combined linear for QKV
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        # x: (B, T, D)
        B, T, D = x.size()
        qkv = self.qkv(x)  # (B, T, 3D)
        qkv = qkv.view(B, T, 3, self.n_heads,
                       self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, heads, T, head_dim)
        # scaled dot product
        scores = torch.matmul(q, k.transpose(-2, -1)) / \
            math.sqrt(self.head_dim)  # (B, heads, T, T)
        if attn_mask is not None:
            # attn_mask expected boolean with shape (B, 1, T, T) where True means allowed
            scores = scores.masked_fill(~attn_mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # (B, heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_ff)
        self.lin2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.lin2(self.dropout(self.act(self.lin1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, d_ff, dropout=dropout)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        # Pre-LayerNorm style
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, n_layers: int, d_model: int, n_heads: int, d_ff: int, vocab_out: int, dropout: float = 0.1, max_pos: int = 2048):
        super().__init__()
        self.d_model = d_model
        # token embedding sized from tokenizer vocab to avoid out of range indices
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_pos, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(
            d_model, n_heads, d_ff, dropout=dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_out, bias=False)

        self._init_weights()

    def _init_weights(self):
        for n, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, input_embeddings: Optional[torch.Tensor] = None):
        # input_ids: (B, T)
        B, T = input_ids.size()
        device = input_ids.device
        if input_embeddings is None:
            x = self.token_embedding(input_ids.to(device))
        else:
            # input_embeddings can be either (V, D) or (B, T, D)
            if input_embeddings.dim() == 2:
                # assume embedding table (V, D)
                x = F.embedding(input_ids.to(device),
                                input_embeddings.to(device))
            else:
                x = input_embeddings.to(device)
        pos = torch.arange(0, T, device=device).unsqueeze(0).expand(B, T)
        x = x + self.pos_emb(pos)
        # attention mask: expecting (B, T) -> convert to (B, 1, T, T) causal+padding
        if attention_mask is not None:
            causal = torch.tril(torch.ones(
                (T, T), dtype=torch.bool, device=device))
            key_mask = attention_mask.to(device).unsqueeze(
                1).unsqueeze(1).expand(B, 1, 1, T).bool()
            attn_mask = causal.unsqueeze(0) & key_mask
        else:
            attn_mask = None
        for b in self.blocks:
            x = b(x, attn_mask=attn_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

# ------------------------- Checkpoint Manager -------------------------


class CheckpointManager:
    def __init__(self, ckpt_dir: Path):
        self.ckpt_dir = ckpt_dir
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, model: nn.Module, optimizer: torch.optim.Optimizer, scaler: Optional[torch.cuda.amp.GradScaler], epoch: int, step: int):
        path = self.ckpt_dir / f"ckpt_{name}.pt"
        data = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch,
            'step': step
        }
        if scaler is not None:
            data['scaler_state'] = scaler.state_dict()
        torch.save(data, path)
        # also update latest symlink-style file
        latest = self.ckpt_dir / 'latest.pt'
        try:
            if latest.exists():
                latest.unlink()
            latest.symlink_to(path.name)
        except Exception:
            # symlink may fail on Windows; instead copy
            torch.save(data, self.ckpt_dir / 'latest.pt')
        return path

    def load_latest(self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, scaler: Optional[torch.cuda.amp.GradScaler] = None):
        latest = self.ckpt_dir / 'latest.pt'
        if not latest.exists():
            return 0, 0
        data = torch.load(latest, map_location='cpu')
        model.load_state_dict(data['model_state'])
        if optimizer is not None and 'optimizer_state' in data:
            optimizer.load_state_dict(data['optimizer_state'])
        if scaler is not None and 'scaler_state' in data:
            try:
                scaler.load_state_dict(data['scaler_state'])
            except Exception:
                pass
        return data.get('epoch', 0), data.get('step', 0)

# ------------------------- Trainer -------------------------


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.DEVICE)
        self.tokenizer = TokenizerWrapper(cfg.TOKENIZER_NAME)
        self.collator = Collator(self.tokenizer, max_seq_len=cfg.MAX_SEQ_LEN)
        vocab_size = getattr(self.tokenizer.tokenizer, 'vocab_size', None) or getattr(
            self.tokenizer.tokenizer, 'vocab', None) and len(self.tokenizer.tokenizer.vocab) or 50257
        self.model = TinyTransformer(vocab_size=vocab_size, n_layers=cfg.N_LAYERS, d_model=cfg.D_MODEL,
                                     n_heads=cfg.N_HEADS, d_ff=cfg.D_FF, vocab_out=cfg.VOCAB_OUT, dropout=cfg.DROPOUT).to(self.device)
        self.input_embeddings: Optional[torch.Tensor] = None
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=3e-4, weight_decay=0.01)
        # use recommended amp API
        if self.device.type == 'cuda':
            self.scaler = torch.amp.GradScaler(device_type='cuda')
        else:
            self.scaler = torch.amp.GradScaler(device_type='cpu')
        self.ckpt = CheckpointManager(cfg.CKPT_DIR)
        self.start_epoch, self.global_step = 0, 0
        # try resume
        try:
            e, s = self.ckpt.load_latest(
                self.model, self.optimizer, self.scaler)
            self.start_epoch = e
            self.global_step = s
            print(f"[{now()}] Resumed from epoch {e} step {s}")
        except Exception:
            pass

    def attach_input_embeddings(self, embeddings_path: Optional[str] = None, freeze: bool = True):
        # Load gemma embeddings (torch saved tensor) if provided. Shape expected (V, D_model)
        if embeddings_path is None:
            return
        emb = torch.load(embeddings_path, map_location='cpu')
        if emb.dim() != 2 or emb.size(1) != self.cfg.D_MODEL:
            raise ValueError("Provided embeddings must be (V, D_MODEL)")
        self.input_embeddings = emb.to(self.device)
        if freeze:
            self.input_embeddings.requires_grad = False
        print(f"[{now()}] Attached external input embeddings: {embeddings_path}")

    def _dataloader(self, mode: str) -> DataLoader:
        ds = ShardedJSONLines(self.cfg.TRAINING_GLOB, self.tokenizer,
                              mode=mode, max_seq_len=self.cfg.MAX_SEQ_LEN)
        return DataLoader(ds, batch_size=self.cfg.BATCH_SIZE_GPU, collate_fn=self.collator, num_workers=2)

    def evaluate(self, mode: str, max_batches: int = None) -> float:
        dl = self._dataloader(mode)
        self.model.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for i, batch in enumerate(dl):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                logits = self.model(
                    input_ids, attention_mask=attention_mask, input_embeddings=self.input_embeddings)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
                total_loss += loss.item()
                n += 1
                if max_batches is not None and n >= max_batches:
                    break
        return total_loss / max(1, n)

    def train(self):
        gacc = self.cfg.GRADIENT_ACCUMULATION_STEPS
        for epoch in range(self.start_epoch, self.cfg.EPOCHS):
            print(
                f"[{now()}] Starting epoch {epoch} (global_step={self.global_step})")
            dl = self._dataloader('train')
            self.model.train()
            running_loss = 0.0
            step_in_epoch = 0
            self.optimizer.zero_grad()
            for step, batch in enumerate(dl):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                with torch.amp.autocast(device_type=self.device.type):
                    logits = self.model(
                        input_ids, attention_mask=attention_mask, input_embeddings=self.input_embeddings)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
                    loss = loss / gacc
                self.scaler.scale(loss).backward()
                running_loss += loss.item() * gacc
                if (step + 1) % gacc == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                step_in_epoch += 1
                if step_in_epoch % (gacc * 50) == 0:
                    avg = running_loss / (gacc * 50)
                    print(
                        f"[{now()}] Epoch {epoch} step {step} avg_loss={avg:.4f}")
                    running_loss = 0.0
            # epoch end
            test_loss = self.evaluate(
                'test', max_batches=Config.MAX_EVAL_BATCHES)
            train_check_loss = self.evaluate(
                'train_check', max_batches=Config.MAX_EVAL_BATCHES)
            print(
                f"[{now()}] Epoch {epoch} complete. TEST_LOSS={test_loss:.4f}  TRAIN_CHECK_LOSS={train_check_loss:.4f}")
            # save checkpoint
            if (epoch + 1) % self.cfg.SAVE_EVERY_EPOCHS == 0:
                name = f"ep{epoch}_step{self.global_step}"
                self.ckpt.save(name, self.model, self.optimizer,
                               self.scaler, epoch, self.global_step)
                print(f"[{now()}] Checkpoint saved: {name}")
        print(f"[{now()}] Training complete")


# ------------------------- MAIN -------------------------
if __name__ == '__main__':
    cfg = Config()
    trainer = Trainer(cfg)
    # Optionally attach external embeddings (uncomment and set path)
    # trainer.attach_input_embeddings('gemma_embeddings.pt', freeze=True)
    trainer.train()
