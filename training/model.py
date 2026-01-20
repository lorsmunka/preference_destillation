from time import time
import torch
import torch.nn as nn
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

        self.input_embedding = nn.Embedding(self.input_vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_length, hidden_dim)

        self.transformer_layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(hidden_dim)
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
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_length = input_ids.shape

        positions = torch.arange(
            seq_length, device=input_ids.device).unsqueeze(0)

        x = self.input_embedding(input_ids)
        x = x + self.position_embedding(positions)
        x = self.dropout(x)

        if attention_mask is None:
            attention_mask = self.create_causal_mask(
                seq_length, input_ids.device)
        elif attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)

        for layer in self.transformer_layers:
            x = layer(x, attention_mask)

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
        position_embedding_params = self.position_embedding.weight.numel()
        embeddings_total = input_embedding_params + position_embedding_params

        attention_total = sum(layer.attention.get_num_parameters()
                              for layer in self.transformer_layers)
        feedforward_total = sum(layer.feed_forward.get_num_parameters()
                                for layer in self.transformer_layers)
        block_layernorm_total = sum(
            layer.norm1.weight.numel() + layer.norm1.bias.numel() +
            layer.norm2.weight.numel() + layer.norm2.bias.numel()
            for layer in self.transformer_layers
        )

        final_layernorm_total = self.layer_norm.weight.numel() + self.layer_norm.bias.numel()

        output_projection_total = self.output_projection.weight.numel(
        ) + (self.output_projection.bias.numel() or 0)

        subtotal = embeddings_total + attention_total + feedforward_total + \
            block_layernorm_total + final_layernorm_total + output_projection_total
        total_parameters = self.get_num_parameters()
        difference = total_parameters - subtotal

        positions = self.vocabulary['positions']

        print("=== Model Parameter Overview ===")
        print(
            f"Total trainable parameters: {total_parameters:,}")
        print(
            f"Embeddings (input + positional): {embeddings_total:,} ({input_embedding_params:,} + {position_embedding_params:,})")
        print(f"Transformer Attention parameters: {attention_total:,}")
        print(f"Transformer Feed-Forward parameters: {feedforward_total:,}")
        print(f"Transformer Block LayerNorms: {block_layernorm_total:,}")
        print(f"Final LayerNorm parameters: {final_layernorm_total:,}")
        print(f"Output projection parameters: {output_projection_total:,}")
        print(f"\nSum of all above: {subtotal:,}")
        print(f"Difference (if any): {difference:,}")
        print(
            f"\nHyperparameters: Hidden dimension={self.hidden_dim}, Heads={self.num_heads}, Layers={len(self.transformer_layers)}, Max sequence length={self.max_seq_length}")
        print(
            f"Input vocab size={self.input_vocab_size}, Output vocab size={self.output_vocab_size}")
        print(f"\nOutput vocabulary sections:")
        print(f"  Example tokens: {positions['example'][0]}-{positions['example'][1]}")
        print(f"  Whitespace tokens: {positions['whitespace'][0]}-{positions['whitespace'][1]}")
        print(f"  Prompt tokens: {positions['prompt'][0]}-{positions['prompt'][1]}")
        print(f"  Auxiliary tokens: {positions['auxiliary'][0]}-{positions['auxiliary'][1]}")


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.feed_forward = FeedForward(hidden_dim, dropout)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output = self.attention(self.norm1(x), attention_mask)
        x = x + self.dropout1(attn_output)

        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout2(ff_output)

        return x

    def get_num_parameters(self) -> int:
        total = 0
        total += self.attention.get_num_parameters()
        total += self.feed_forward.get_num_parameters()
        for ln in (self.norm1, self.norm2):
            total += ln.weight.numel()
            total += ln.bias.numel()
        return total


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_length, _ = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(batch_size, seq_length, self.num_heads,
                   self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.num_heads,
                   self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads,
                   self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        attention_output = torch.matmul(attention_weights, V)

        attention_output = attention_output.transpose(1, 2)
        attention_output = attention_output.reshape(
            batch_size, seq_length, self.hidden_dim)
        output = self.output(attention_output)

        return output

    def get_num_parameters(self) -> int:
        total = 0
        for linear in (self.query, self.key, self.value, self.output):
            total += linear.weight.numel()
            if linear.bias is not None:
                total += linear.bias.numel()
        return total


class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        ff_dim = hidden_dim * 4

        self.linear1 = nn.Linear(hidden_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

    def get_num_parameters(self) -> int:
        total = 0
        for linear in (self.linear1, self.linear2):
            total += linear.weight.numel()
            if linear.bias is not None:
                total += linear.bias.numel()
        return total
