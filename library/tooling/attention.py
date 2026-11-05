"""Where a model looks: the attention weights of a student Transformer and of a Hugging Face teacher,
read on the same text so the two can be compared (nothing in training distils attention, so any
likeness is learned on its own).

    logits, maps = student_attention(model, input_ids)               # [layers, heads, seq, seq]
    rows = TeacherAttention(teacher).rows(input_ids, query_positions)  # [layers, heads, queries, seq]

Both take `query_positions` (keep only those rows) and `column_groups` (add the weights of several
positions into one column, e.g. the teacher's instruction header into one "outside the student's
view" column, or a project's tokens into its lines), so a long input never holds every full map.
The teacher must be loaded with attn_implementation="eager": the fused kernels never return weights.
"""

from typing import List, Optional

import torch

from library.model.transformer import apply_rotary_pos_emb, causal_mask


def group_columns(rows: torch.Tensor, column_groups: List[int], group_count: int) -> torch.Tensor:
    """[..., seq] -> [..., group_count]: the weight of every column added into its group's column."""
    groups = torch.as_tensor(column_groups, device=rows.device)
    grouped = torch.zeros(*rows.shape[:-1], group_count, device=rows.device, dtype=torch.float32)
    return grouped.index_add_(rows.dim() - 1, groups, rows.float())


@torch.no_grad()
def student_attention(model, input_ids: torch.Tensor, query_positions: Optional[List[int]] = None,
                      column_groups: Optional[List[int]] = None, group_count: Optional[int] = None):
    """(logits, maps) of a student Transformer for one sequence (input_ids: [1, seq], remapped).
    maps: [layers, heads, queries, seq] (queries = every position, or `query_positions`); row i is how
    query position i spreads its attention over positions 0..i, and adds up to 1. Computed the way the
    unfused branch of MultiHeadAttention does it, in eval mode (no dropout, no attention noise), and the
    logits come from these very weights: they equal model(input_ids) (tests/test_attention.py)."""
    model.eval()
    seq_length = input_ids.shape[1]
    hidden = model.input_embedding(input_ids)
    cos, sin = model.rotary_embedding(seq_length)
    cos, sin = cos[None, None].to(hidden.device), sin[None, None].to(hidden.device)
    mask = causal_mask(seq_length, hidden.device)
    queries = torch.as_tensor(query_positions, device=hidden.device) if query_positions is not None else None
    maps = []
    for layer in model.transformer_layers:
        attention = layer.attention
        normed = layer.norm1(hidden)

        def split_heads(projection):
            return projection(normed).view(1, seq_length, attention.num_heads, attention.head_dim).transpose(1, 2)

        query, key, value = split_heads(attention.query), split_heads(attention.key), split_heads(attention.value)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        weights = torch.softmax(query @ key.transpose(-2, -1) / attention.scale + mask, dim=-1)
        mixed = (weights @ value).transpose(1, 2).reshape(1, seq_length, attention.hidden_dim)
        hidden = hidden + attention.output(mixed)
        hidden = hidden + layer.feed_forward(layer.norm2(hidden))
        kept = weights[0] if queries is None else weights[0][:, queries]
        maps.append(group_columns(kept, column_groups, group_count) if column_groups is not None else kept)
    logits = model.output_projection(model.layer_norm(hidden))
    return logits, torch.stack(maps)


class TeacherAttention:
    """Catches the attention weights of every text attention layer of a Hugging Face causal LM in one
    forward pass (forward hooks on the attention modules, whose eager forward returns
    (output, weights)). Vision / audio towers are left out."""

    def __init__(self, model):
        self.model = model
        self.layers = [module for module in model.modules()
                       if type(module).__name__.endswith("Attention") and hasattr(module, "q_proj")
                       and not any(part in type(module).__name__ for part in ("Vision", "Audio"))]
        if not self.layers:
            raise ValueError(f"{type(model).__name__}: no attention modules with q_proj found")

    @torch.no_grad()
    def rows(self, input_ids: List[int], query_positions: List[int], column_groups: Optional[List[int]] = None,
             group_count: Optional[int] = None) -> torch.Tensor:
        """[layers, heads, queries, seq or group_count], float32 on the CPU."""
        captured = []

        def keep(module, inputs, output):
            weights = output[1] if isinstance(output, tuple) and len(output) > 1 else None
            if weights is None:
                raise RuntimeError("the attention returned no weights: load the teacher with attn_implementation='eager'")
            kept = weights[0][:, torch.as_tensor(query_positions, device=weights.device)].float()
            if column_groups is not None:
                kept = group_columns(kept, column_groups, group_count)
            captured.append(kept.cpu())

        handles = [layer.register_forward_hook(keep) for layer in self.layers]
        device = next(self.model.parameters()).device
        try:
            self.model(input_ids=torch.tensor([input_ids], device=device), use_cache=False, logits_to_keep=1)
        finally:
            for handle in handles:
                handle.remove()
        return torch.stack(captured)
