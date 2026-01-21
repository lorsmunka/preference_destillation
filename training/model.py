from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from transformers import AutoTokenizer
from shared import (
    Utilities,
    MODEL_NAME,
    HIDDEN_DIM,
    NUM_LAYERS,
    NUM_HEADS,
    MAX_SEQ_LENGTH,
    DROPOUT,
)


class RMSNorm(nn.Module):
    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

    def get_num_parameters(self) -> int:
        return self.weight.numel()


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_length: int = MAX_SEQ_LENGTH, base: int = 10000):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_length = max_seq_length

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)

        self._build_cache(max_seq_length)

    def _build_cache(self, seq_length: int):
        positions = torch.arange(seq_length, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def forward(self, seq_length: int):
        if seq_length > self.max_seq_length:
            self._build_cache(seq_length)
            self.max_seq_length = seq_length
        return self.cos_cached[:seq_length], self.sin_cached[:seq_length]


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class Transformer(nn.Module):
    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        num_heads: int = NUM_HEADS,
        max_seq_length: int = MAX_SEQ_LENGTH,
        dropout: float = DROPOUT
    ):
        start_time = time()
        print("Initializing Transformer model...")
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.input_vocab_size = self.tokenizer.vocab_size

        self.vocabulary = Utilities.build_vocabulary(self.tokenizer)
        self.output_vocab_size = self.vocabulary['vocab_size']

        self.output_token_ids = [
            self.vocabulary['token_to_id'][token]
            for token in self.vocabulary['token_list']
        ]

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.max_seq_length = max_seq_length
        self.head_dim = hidden_dim // num_heads

        self.input_embedding = nn.Embedding(self.input_vocab_size, hidden_dim)

        self.rotary_embedding = RotaryEmbedding(self.head_dim, max_seq_length)

        self.transformer_layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, self.head_dim, dropout)
            for _ in range(num_layers)
        ])

        self.layer_norm = RMSNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.output_projection = nn.Linear(hidden_dim, self.output_vocab_size)

        self._init_weights()
        self.model_info()

        elapsed_time = time() - start_time
        print(f"Model initialized -> took {elapsed_time:.2f} seconds.\n")

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_length = input_ids.shape

        x = self.input_embedding(input_ids)
        x = self.dropout(x)

        cos, sin = self.rotary_embedding(seq_length)
        cos = cos.to(x.device)
        sin = sin.to(x.device)

        if attention_mask is None:
            attention_mask = self.create_causal_mask(seq_length, input_ids.device)
        elif attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)

        for layer in self.transformer_layers:
            x = layer(x, attention_mask, cos, sin)

        x = self.layer_norm(x)

        logits = self.output_projection(x)

        return logits

    def create_causal_mask(self, seq_length: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(
            seq_length, seq_length, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_vocabulary(self) -> dict:
        return self.vocabulary

    def get_positions(self) -> dict:
        return self.vocabulary['positions']

    def model_info(self):
        input_embedding_params = self.input_embedding.weight.numel()

        attention_total = sum(layer.attention.get_num_parameters()
                              for layer in self.transformer_layers)
        feedforward_total = sum(layer.feed_forward.get_num_parameters()
                                for layer in self.transformer_layers)
        block_norm_total = sum(
            layer.norm1.get_num_parameters() + layer.norm2.get_num_parameters()
            for layer in self.transformer_layers
        )

        final_norm_total = self.layer_norm.get_num_parameters()

        output_projection_total = self.output_projection.weight.numel()
        if self.output_projection.bias is not None:
            output_projection_total += self.output_projection.bias.numel()

        subtotal = input_embedding_params + attention_total + feedforward_total + \
            block_norm_total + final_norm_total + output_projection_total
        total_parameters = self.get_num_parameters()
        difference = total_parameters - subtotal

        positions = self.vocabulary['positions']

        print("=== Model Parameter Overview ===")
        print(f"Total trainable parameters: {total_parameters:,}")
        print(f"Input embeddings: {input_embedding_params:,}")
        print(f"Transformer Attention: {attention_total:,}")
        print(f"Transformer Feed-Forward: {feedforward_total:,}")
        print(f"RMSNorm layers: {block_norm_total + final_norm_total:,}")
        print(f"Output projection: {output_projection_total:,}")
        print(f"\nSum of all above: {subtotal:,}")
        print(f"Difference (RoPE buffers, non-trainable): {difference:,}")
        print(
            f"\nHyperparameters: hidden_dim={self.hidden_dim}, heads={self.num_heads}, layers={len(self.transformer_layers)}, max_seq_length={self.max_seq_length}")
        print(
            f"Input vocab size={self.input_vocab_size}, Output vocab size={self.output_vocab_size}")
        print(f"\nOutput vocabulary sections:")
        print(f"  Example tokens: {positions['example'][0]}-{positions['example'][1]}")
        print(f"  Whitespace tokens: {positions['whitespace'][0]}-{positions['whitespace'][1]}")
        print(f"  Prompt tokens: {positions['prompt'][0]}-{positions['prompt'][1]}")
        print(f"  Auxiliary tokens: {positions['auxiliary'][0]}-{positions['auxiliary'][1]}")


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int, dropout: float = 0.1):
        super().__init__()

        self.attention = MultiHeadAttention(hidden_dim, num_heads, head_dim, dropout)
        self.feed_forward = FeedForward(hidden_dim, dropout)

        self.norm1 = RMSNorm(hidden_dim)
        self.norm2 = RMSNorm(hidden_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor
    ) -> torch.Tensor:
        attn_output = self.attention(self.norm1(x), attention_mask, cos, sin)
        x = x + self.dropout1(attn_output)

        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout2(ff_output)

        return x

    def get_num_parameters(self) -> int:
        total = 0
        total += self.attention.get_num_parameters()
        total += self.feed_forward.get_num_parameters()
        total += self.norm1.get_num_parameters()
        total += self.norm2.get_num_parameters()
        return total


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int, dropout: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = math.sqrt(self.head_dim)

        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.output = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_length, _ = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        cos = cos[:seq_length].unsqueeze(0).unsqueeze(0)
        sin = sin[:seq_length].unsqueeze(0).unsqueeze(0)
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        attention_output = torch.matmul(attention_weights, V)

        attention_output = attention_output.transpose(1, 2)
        attention_output = attention_output.reshape(batch_size, seq_length, self.hidden_dim)
        output = self.output(attention_output)

        return output

    def get_num_parameters(self) -> int:
        total = 0
        for linear in (self.query, self.key, self.value, self.output):
            total += linear.weight.numel()
        return total


class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        ff_dim = hidden_dim * 4

        self.gate_proj = nn.Linear(hidden_dim, ff_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, ff_dim, bias=False)
        self.down_proj = nn.Linear(ff_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.gelu(self.gate_proj(x))
        up = self.up_proj(x)
        x = self.down_proj(gate * up)
        x = self.dropout(x)
        return x

    def get_num_parameters(self) -> int:
        total = 0
        for linear in (self.gate_proj, self.up_proj, self.down_proj):
            total += linear.weight.numel()
        return total
