"""
Gemma-3-4B Distillation Model (~100M parameters)
A clean, resumable training pipeline for knowledge distillation.

Install dependencies:
pip install torch transformers tqdm

For CUDA 11.8:
pip install torch --index-url https://download.pytorch.org/whl/cu118

For CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121
"""

import os
import json
import glob
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import time


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ModelConfig:
    """All hyperparameters in one place"""
    # Architecture
    d_model: int = 768              # Hidden dimension
    num_layers: int = 12            # Transformer layers
    num_heads: int = 12             # Attention heads
    d_ff: int = 3072                # FFN inner dimension (4x d_model)
    dropout: float = 0.1            # Dropout rate
    max_seq_len: int = 512          # Maximum sequence length
    teacher_embedding_dim: int = 2304  # Gemma-3-4b embedding dimension

    # Vocabulary
    vocab_size: int = 100           # Output vocabulary size (will be computed)

    # Training
    batch_size: int = 16            # Batch size
    learning_rate: float = 3e-4     # Learning rate
    weight_decay: float = 0.01      # Weight decay
    num_epochs: int = 10            # Number of epochs
    gradient_clip: float = 1.0      # Gradient clipping

    # Distillation
    temperature: float = 2.0        # Temperature for soft targets

    # Paths
    teacher_model: str = "google/gemma-3-4b-it"
    training_data_dir: str = "./training_data"
    checkpoint_dir: str = "./checkpoints"

    # Data split
    test_split: float = 0.1         # Reserve 10% for testing

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# VOCABULARY BUILDER
# ============================================================================

class VocabularyBuilder:
    """Extract unique tokens from training data to build restricted vocabulary"""

    def __init__(self, tokenizer, training_data_dir: str):
        self.tokenizer = tokenizer
        self.training_data_dir = training_data_dir

    def build(self) -> Tuple[List[int], Dict[int, int]]:
        """
        Returns:
            vocab_token_ids: List of token IDs in our vocabulary
            token_id_to_vocab_idx: Mapping from full vocab ID to our restricted vocab index
        """
        print("Building vocabulary from training data...")

        # Collect all unique tokens from training data
        unique_tokens = set()

        files = glob.glob(
            f"{self.training_data_dir}/training_data_batch_*.jsonl")
        for filepath in tqdm(files, desc="Scanning files"):
            with open(filepath, 'r') as f:
                for line in f:
                    example = json.loads(line)
                    # Extract tokens from step probabilities
                    for step in example['steps']:
                        for token_name in step['probabilities'].keys():
                            if token_name != "__OTHER__":
                                unique_tokens.add(token_name)

        print(f"Found {len(unique_tokens)} unique tokens")

        # Convert token strings to token IDs
        vocab_token_ids = []
        for token_str in unique_tokens:
            token_ids = self.tokenizer.convert_tokens_to_ids([token_str])
            if token_ids[0] != self.tokenizer.unk_token_id:
                vocab_token_ids.append(token_ids[0])

        # Add special tokens (EOS, BOS, PAD if they exist)
        special_token_ids = [
            self.tokenizer.eos_token_id,
            self.tokenizer.bos_token_id,
            self.tokenizer.pad_token_id
        ]
        for tid in special_token_ids:
            if tid is not None and tid not in vocab_token_ids:
                vocab_token_ids.append(tid)

        vocab_token_ids = sorted(vocab_token_ids)

        # Create mapping: full_vocab_id -> restricted_vocab_index
        token_id_to_vocab_idx = {tid: idx for idx,
                                 tid in enumerate(vocab_token_ids)}

        print(f"Final vocabulary size: {len(vocab_token_ids)}")
        return vocab_token_ids, token_id_to_vocab_idx


# ============================================================================
# EMBEDDING CACHE - Load Gemma embeddings once and cache them
# ============================================================================

class EmbeddingCache:
    """Cache Gemma embeddings to avoid repeated loading"""

    def __init__(self, model_name: str, device: str):
        print("Loading Gemma embedding layer (one-time operation)...")
        from transformers import AutoModel

        # Load only the embedding layer
        full_model = AutoModel.from_pretrained(
            model_name,
            device_map="cpu",  # Load to CPU first
            dtype=torch.float32  # Use float32 for embeddings
        )

        # Extract embedding layer
        self.embeddings = full_model.get_input_embeddings()
        self.embeddings.to(device)
        self.embeddings.eval()

        # Freeze embeddings
        for param in self.embeddings.parameters():
            param.requires_grad = False

        # Clean up full model
        del full_model
        torch.cuda.empty_cache()

        print(f"Embedding layer loaded on {device}")

    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get embeddings for input_ids"""
        with torch.no_grad():
            return self.embeddings(input_ids)


# ============================================================================
# DATASET
# ============================================================================

class DistillationDataset(Dataset):
    """Streaming dataset that loads training examples on-the-fly"""

    def __init__(self, file_paths: List[str], tokenizer, token_id_to_vocab_idx: Dict[int, int],
                 vocab_size: int):
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.token_id_to_vocab_idx = token_id_to_vocab_idx
        self.vocab_size = vocab_size

        # Build index: (file_idx, line_idx) for each example
        self.index = []
        for file_idx, filepath in enumerate(file_paths):
            with open(filepath, 'r') as f:
                num_lines = sum(1 for _ in f)
            self.index.extend([(file_idx, line_idx)
                              for line_idx in range(num_lines)])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_idx, line_idx = self.index[idx]
        filepath = self.file_paths[file_idx]

        # Read specific line
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                if i == line_idx:
                    example = json.loads(line)
                    break

        # Tokenize the sentence (input)
        sentence = example['sentence']
        # Recreate the prompt that was used during generation
        prompt = f"""Analyze this sentence and return your evaluation as JSON:

    Sentence: "{sentence}"

    Provide exactly one value for each field based on the sentence content:
    - tone: aggressive, rude, neutral, polite, friendly
    - sentiment: negative, neutral, positive
    - safety: harmful, safe
    - toxicity: toxic, respectful

    JSON:
    """

        input_ids = self.tokenizer.encode(
            prompt, add_special_tokens=True, max_length=512, truncation=True)

        # Extract teacher probabilities and target tokens for each step
        teacher_probs_list = []
        target_token_ids = []

        for step in example['steps']:
            # Get teacher's probability distribution (logits)
            token_logits = step['probabilities']

            # Create logit vector over restricted vocabulary
            logits = torch.full((self.vocab_size,), float('-inf'))
            for token_str, logit in token_logits.items():
                if token_str == "__OTHER__":
                    continue  # Skip "other" tokens

                # Convert token string to ID
                token_ids_list = self.tokenizer.convert_tokens_to_ids([
                                                                      token_str])
                token_id = token_ids_list[0]

                # Map to our restricted vocabulary
                if token_id in self.token_id_to_vocab_idx:
                    vocab_idx = self.token_id_to_vocab_idx[token_id]
                    logits[vocab_idx] = logit

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=0)
            teacher_probs_list.append(probs)

            # Get the actual generated token as target
            token_str = step['token']
            token_ids_list = self.tokenizer.convert_tokens_to_ids([token_str])
            token_id = token_ids_list[0]
            if token_id in self.token_id_to_vocab_idx:
                target_token_ids.append(self.token_id_to_vocab_idx[token_id])
            else:
                # If token not in vocab, use 0 (will be masked)
                target_token_ids.append(0)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            # (num_steps, vocab_size)
            'teacher_probs': torch.stack(teacher_probs_list),
            'target_ids': torch.tensor(target_token_ids, dtype=torch.long)
        }


def collate_fn(batch):
    """Custom collate function to pad sequences"""
    # Find max lengths
    max_input_len = max(len(item['input_ids']) for item in batch)
    max_output_len = max(len(item['target_ids']) for item in batch)

    # Pad inputs
    input_ids = torch.zeros(len(batch), max_input_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_input_len, dtype=torch.long)

    for i, item in enumerate(batch):
        length = len(item['input_ids'])
        input_ids[i, :length] = item['input_ids']
        attention_mask[i, :length] = 1

    # Pad outputs
    teacher_probs = torch.zeros(
        len(batch), max_output_len, batch[0]['teacher_probs'].shape[-1])
    target_ids = torch.zeros(len(batch), max_output_len, dtype=torch.long)
    output_mask = torch.zeros(len(batch), max_output_len, dtype=torch.long)

    for i, item in enumerate(batch):
        length = len(item['target_ids'])
        teacher_probs[i, :length] = item['teacher_probs']
        target_ids[i, :length] = item['target_ids']
        output_mask[i, :length] = 1

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'teacher_probs': teacher_probs,
        'target_ids': target_ids,
        'output_mask': output_mask
    }


# ============================================================================
# MODEL COMPONENTS
# ============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal masking"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Q, K, V projections (combined for efficiency)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len) attention mask (1 = attend, 0 = ignore)
        Returns:
            (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Project and split into Q, K, V
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3 * d_model)
        qkv = qkv.reshape(batch_size, seq_len, 3,
                          self.num_heads, self.head_dim)
        # (3, batch, num_heads, seq_len, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        # (batch, num_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Apply causal mask (prevent attending to future tokens)
        causal_mask = torch.tril(torch.ones(
            seq_len, seq_len, device=x.device)).bool()
        scores = scores.masked_fill(~causal_mask, float('-inf'))

        # Apply padding mask if provided
        if mask is not None:
            # Expand mask to (batch, 1, 1, seq_len) for broadcasting
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)  # (batch, num_heads, seq_len, head_dim)

        # Concatenate heads
        # (batch, seq_len, num_heads, head_dim)
        out = out.transpose(1, 2).contiguous()
        out = out.reshape(batch_size, seq_len, self.d_model)

        # Final projection
        out = self.out_proj(out)

        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network (2-layer MLP)"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        x = self.linear1(x)
        x = F.gelu(x)  # GELU activation (smooth alternative to ReLU)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer layer with attention + FFN + residual connections"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # Sub-layers
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        # Layer normalization (applied before sub-layers - pre-norm architecture)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len) attention mask
        Returns:
            (batch, seq_len, d_model)
        """
        # Attention with residual
        residual = x
        x = self.norm1(x)
        x = self.attention(x, mask)
        x = self.dropout(x)
        x = x + residual

        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + residual

        return x


class DistillationModel(nn.Module):
    """Complete distillation model: Gemma embeddings -> Transformer -> Output head"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Project Gemma embeddings to our model dimension
        self.input_projection = nn.Linear(
            config.teacher_embedding_dim, config.d_model)

        # Stack of transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config.d_model, config.num_heads,
                             config.d_ff, config.dropout)
            for _ in range(config.num_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(config.d_model)

        # Output head (project to restricted vocabulary)
        self.output_head = nn.Linear(config.d_model, config.vocab_size)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with small random values"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            embeddings: (batch, seq_len, teacher_embedding_dim) - from Gemma
            mask: (batch, seq_len) attention mask
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        # Project embeddings to model dimension
        x = self.input_projection(embeddings)  # (batch, seq_len, d_model)

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        # Final normalization
        x = self.norm(x)

        # Project to vocabulary
        logits = self.output_head(x)  # (batch, seq_len, vocab_size)

        return logits

    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class CheckpointManager:
    """Manage model checkpoints with resume capability"""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(self, model: nn.Module, optimizer: torch.optim.Optimizer,
             epoch: int, step: int, loss: float):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }

        path = os.path.join(self.checkpoint_dir,
                            f"checkpoint_epoch{epoch}_step{step}.pt")
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

    def load_latest(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> Tuple[int, int]:
        """Load latest checkpoint, returns (epoch, step)"""
        checkpoints = glob.glob(os.path.join(
            self.checkpoint_dir, "checkpoint_*.pt"))

        if not checkpoints:
            print("No checkpoints found, starting from scratch")
            return 0, 0

        # Find latest checkpoint
        latest = max(checkpoints, key=os.path.getctime)

        print(f"Loading checkpoint: {latest}")
        checkpoint = torch.load(latest, map_location='cpu')

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint['epoch'], checkpoint['step']


def compute_distillation_loss(student_logits: torch.Tensor, teacher_probs: torch.Tensor,
                              temperature: float, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence loss for knowledge distillation

    Args:
        student_logits: (batch, seq_len, vocab_size) - raw logits from student
        teacher_probs: (batch, seq_len, vocab_size) - probability distribution from teacher
        temperature: Temperature for softening distributions
        mask: (batch, seq_len) - valid positions mask
    """
    # Soften distributions with temperature
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # KL divergence: sum over vocab, mean over sequence (only valid positions)
    kl_div = F.kl_div(student_log_probs, teacher_probs,
                      reduction='none').sum(dim=-1)

    # Apply mask and average
    masked_kl = (kl_div * mask).sum() / mask.sum()

    # Scale by T^2 as per distillation literature
    return masked_kl * (temperature ** 2)


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(model: DistillationModel, embedding_cache: EmbeddingCache,
                dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                config: ModelConfig, epoch: int) -> float:
    """Train for one epoch"""
    model.train()

    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in progress_bar:
        # Move to device
        input_ids = batch['input_ids'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device)
        teacher_probs = batch['teacher_probs'].to(config.device)
        output_mask = batch['output_mask'].to(config.device)

        # Get embeddings from cache (frozen)
        embeddings = embedding_cache.get_embeddings(input_ids)

        # Forward pass through student
        logits = model(embeddings, mask=attention_mask)

        # We only compute loss on the generated tokens (not the input prompt)
        # Align logits with teacher_probs by taking the last N positions
        seq_len = logits.shape[1]
        output_len = teacher_probs.shape[1]

        if seq_len >= output_len:
            # Take last output_len positions from logits
            student_logits = logits[:, -output_len:, :]
        else:
            # Pad teacher_probs if needed (shouldn't happen in practice)
            student_logits = logits
            teacher_probs = teacher_probs[:, :seq_len, :]
            output_mask = output_mask[:, :seq_len]

        # Compute loss
        loss = compute_distillation_loss(
            student_logits, teacher_probs, config.temperature, output_mask
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip)

        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        num_batches += 1

        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

    return total_loss / num_batches


def evaluate(model: DistillationModel, embedding_cache: EmbeddingCache,
             dataloader: DataLoader, config: ModelConfig) -> float:
    """Evaluate on test set"""
    model.eval()

    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            teacher_probs = batch['teacher_probs'].to(config.device)
            output_mask = batch['output_mask'].to(config.device)

            # Get embeddings
            embeddings = embedding_cache.get_embeddings(input_ids)

            # Forward pass
            logits = model(embeddings, mask=attention_mask)

            # Align logits with teacher_probs
            seq_len = logits.shape[1]
            output_len = teacher_probs.shape[1]

            if seq_len >= output_len:
                student_logits = logits[:, -output_len:, :]
            else:
                student_logits = logits
                teacher_probs = teacher_probs[:, :seq_len, :]
                output_mask = output_mask[:, :seq_len]

            # Compute loss
            loss = compute_distillation_loss(
                student_logits, teacher_probs, config.temperature, output_mask
            )

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Configuration
    config = ModelConfig()

    print("=" * 80)
    print("GEMMA DISTILLATION MODEL TRAINING")
    print("=" * 80)
    print(f"Device: {config.device}")
    print(f"Training data: {config.training_data_dir}")
    print()

    # Load tokenizer only (no model needed!)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.teacher_model)

    # Create embedding cache (loads embeddings to GPU once)
    embedding_cache = EmbeddingCache(config.teacher_model, config.device)

    # Build vocabulary
    vocab_builder = VocabularyBuilder(tokenizer, config.training_data_dir)
    vocab_token_ids, token_id_to_vocab_idx = vocab_builder.build()
    config.vocab_size = len(vocab_token_ids)

    # Get all training files and split into train/test
    all_files = sorted(
        glob.glob(f"{config.training_data_dir}/training_data_batch_*.jsonl"))
    split_idx = int(len(all_files) * (1 - config.test_split))
    train_files = all_files[:split_idx]
    test_files = all_files[split_idx:]

    print(f"Training files: {len(train_files)}")
    print(f"Test files: {len(test_files)}")

    # Create datasets
    train_dataset = DistillationDataset(
        train_files, tokenizer, token_id_to_vocab_idx, config.vocab_size
    )
    test_dataset = DistillationDataset(
        test_files, tokenizer, token_id_to_vocab_idx, config.vocab_size
    )

    print(f"Training examples: {len(train_dataset):,}")
    print(f"Test examples: {len(test_dataset):,}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, collate_fn=collate_fn, num_workers=0  # 0 for Windows compatibility
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    # Initialize model
    print("\nInitializing student model...")
    model = DistillationModel(config)
    model = model.to(config.device)

    num_params = model.count_parameters()
    print(f"Student model parameters: {num_params:,} ({num_params/1e6:.1f}M)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Checkpoint manager
    checkpoint_manager = CheckpointManager(config.checkpoint_dir)
    start_epoch, start_step = checkpoint_manager.load_latest(model, optimizer)

    # Training loop
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    for epoch in range(start_epoch, config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        print("-" * 80)

        # Train
        train_loss = train_epoch(
            model, embedding_cache, train_loader,
            optimizer, config, epoch + 1
        )

        print(f"Train Loss: {train_loss:.4f}")

        # Evaluate
        test_loss = evaluate(model, embedding_cache, test_loader, config)
        print(f"Test Loss: {test_loss:.4f}")

        # Save checkpoint
        checkpoint_manager.save(model, optimizer, epoch + 1, 0, test_loss)

        print()

    print("=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
