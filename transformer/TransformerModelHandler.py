import torch
import torch.nn as nn
import math
from typing import Dict, Optional


class TransformerModelHandler(nn.Module):
    def __init__(
        self,
        input_vocab_size: int,
        output_vocab_size: int,
        output_token_ids: list,
        hidden_dim: int = 1024,
        num_layers: int = 16,
        num_heads: int = 16,
        max_seq_length: int = 75,
        dropout: float = 0.1,
        low_rank_dim: int = 256
    ):
        super().__init__()

        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.output_token_ids = output_token_ids
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.max_seq_length = max_seq_length
        self.low_rank_dim = low_rank_dim

        self.input_embedding = nn.Embedding(input_vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_length, hidden_dim)

        self.transformer_layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout, low_rank_dim)
            for _ in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.output_projection = nn.Linear(hidden_dim, output_vocab_size)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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

        if attention_mask is not None:
            attention_mask = self._create_causal_mask(
                seq_length, input_ids.device)
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        else:
            attention_mask = self._create_causal_mask(
                seq_length, input_ids.device)

        for layer in self.transformer_layers:
            x = layer(x, attention_mask)

        x = self.layer_norm(x)

        logits = self.output_projection(x)

        return logits

    def _create_causal_mask(self, seq_length: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(
            seq_length, seq_length, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def map_output_logits_to_full_vocab(self, restricted_logits: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, _ = restricted_logits.shape

        full_vocab_logits = torch.full(
            (batch_size, seq_length, self.input_vocab_size),
            float('-inf'),
            device=restricted_logits.device,
            dtype=restricted_logits.dtype
        )

        output_indices = torch.tensor(
            self.output_token_ids, device=restricted_logits.device)
        full_vocab_logits[:, :, output_indices] = restricted_logits

        return full_vocab_logits

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1, low_rank_dim: int = 256):
        super().__init__()

        self.attention = MultiHeadAttention(
            hidden_dim, num_heads, dropout, low_rank_dim)
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


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1, low_rank_dim: int = 256):
        super().__init__()

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.low_rank_dim = low_rank_dim

        self.query_down = nn.Linear(hidden_dim, low_rank_dim, bias=False)
        self.query_up = nn.Linear(low_rank_dim, hidden_dim)

        self.key_down = nn.Linear(hidden_dim, low_rank_dim, bias=False)
        self.key_up = nn.Linear(low_rank_dim, hidden_dim)

        self.value_down = nn.Linear(hidden_dim, low_rank_dim, bias=False)
        self.value_up = nn.Linear(low_rank_dim, hidden_dim)

        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_length, _ = x.shape

        Q = self.query_up(self.query_down(x))
        K = self.key_up(self.key_down(x))
        V = self.value_up(self.value_down(x))

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

        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(
            batch_size, seq_length, self.hidden_dim)

        output = self.output(attention_output)

        return output


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
