"""Distribution metrics — pure tensor functions, computed INSIDE the eval pass.

These used to require a separate post-run tool (`top-k-accruacy-analsy/`) that reloaded the
model and re-ran every batch. The eval pass already holds the student prediction logits and
the teacher logits per step, so all of this is ~free there. No model reload post-run.

All functions take `[steps, vocab]` logits and `[steps]` target indices and return per-step
tensors; `step_distribution_stats` aggregates one example into summable totals.
"""

from typing import Dict

import torch
import torch.nn.functional as F


def per_step_entropy(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Shannon entropy (nats) of each row's softmax distribution. Quantifies how 'soft' vs
    'sharp' the next-token distribution is — the axis the thesis domain comparison rests on."""
    log_probs = F.log_softmax(logits / temperature, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def perplexity_from_ce(mean_ce_nats: float) -> float:
    """Perplexity = exp(cross-entropy in nats). Standard LM reporting unit; low = confident+right."""
    return float(torch.exp(torch.tensor(float(mean_ce_nats))))


def topk_target_hits(student_logits: torch.Tensor, target_indices: torch.Tensor, k: int) -> torch.Tensor:
    """Per step: is the teacher's target token among the student's top-k? -> bool tensor."""
    top_indices = student_logits.topk(k, dim=-1).indices
    return (top_indices == target_indices.unsqueeze(1)).any(dim=1)


def teacher_student_overlap(student_logits: torch.Tensor, teacher_logits: torch.Tensor, k: int) -> torch.Tensor:
    """Per step: fraction of the student's top-k that also appear in the teacher's top-k."""
    student_top = student_logits.topk(k, dim=-1).indices
    teacher_top = teacher_logits.topk(k, dim=-1).indices
    matches = (student_top.unsqueeze(2) == teacher_top.unsqueeze(1)).any(dim=2).sum(dim=1)
    return matches.float() / k


def target_rank(student_logits: torch.Tensor, target_indices: torch.Tensor) -> torch.Tensor:
    """Per step: 1-based rank of the target token in the student distribution (1 = top, lower better)."""
    target_logit = student_logits.gather(1, target_indices.unsqueeze(1))
    return (student_logits > target_logit).sum(dim=1) + 1


def step_distribution_stats(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target_indices: torch.Tensor,
    k: int,
) -> Dict[str, float]:
    """Summable per-example totals for the eval aggregator (compute-once principle)."""
    effective_k = min(k, student_logits.shape[-1])
    return {
        "topk_hits": float(topk_target_hits(student_logits, target_indices, effective_k).sum().item()),
        "overlap_sum": float(teacher_student_overlap(student_logits, teacher_logits, effective_k).sum().item()),
        "target_rank_sum": float(target_rank(student_logits, target_indices).sum().item()),
        "student_entropy_sum": float(per_step_entropy(student_logits).sum().item()),
        "teacher_entropy_sum": float(per_step_entropy(teacher_logits).sum().item()),
        "steps": int(student_logits.shape[0]),
    }
