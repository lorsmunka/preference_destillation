"""The distillation loss: a per-token distribution-matching term (one of several
divergences — the idea #38 zoo) mixed with per-token cross-entropy on the teacher's
chosen token. With every dial at its default this is exactly forward-KL + CE.
"""

import math

import torch
import torch.nn.functional as F


class DistillationLoss:
    def __init__(self, divergence: str = 'forward_kl', entropy_kl_weighting: float = 0.0):
        self.divergence = divergence
        self.entropy_kl_weighting = entropy_kl_weighting

    def compute(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                target_indices: torch.Tensor, student_temperature: float,
                teacher_temperature: float, kl_ratio: float, reduction: str = 'mean'):
        """Returns (divergence term, cross-entropy term, combined loss)."""
        student_log_probs = F.log_softmax(student_logits / student_temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / teacher_temperature, dim=-1)

        divergence_per_token = self.divergence_per_token(student_log_probs, teacher_probs) * (student_temperature ** 2)
        ce_per_token = F.cross_entropy(
            student_logits / student_temperature, target_indices, reduction='none') * (student_temperature ** 2)

        kl_ratio_per_token = self.kl_ratio_per_token(teacher_probs, kl_ratio)
        combined_per_token = kl_ratio_per_token * divergence_per_token + (1 - kl_ratio_per_token) * ce_per_token

        if reduction == 'mean':
            return divergence_per_token.mean(), ce_per_token.mean(), combined_per_token.mean()
        return divergence_per_token.sum(), ce_per_token.sum(), combined_per_token.sum()

    def divergence_per_token(self, student_log_probs: torch.Tensor, teacher_probs: torch.Tensor) -> torch.Tensor:
        """One number per token for the chosen divergence, summed over the vocabulary."""
        if self.divergence == 'forward_kl':  # KL(teacher || student) — the classic Hinton direction
            return F.kl_div(student_log_probs, teacher_probs, reduction='none').sum(dim=-1)
        teacher_log_probs = torch.log(teacher_probs.clamp_min(1e-12))
        student_probs = student_log_probs.exp()
        if self.divergence == 'reverse_kl':  # KL(student || teacher) — mode-seeking
            return (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)
        if self.divergence == 'js':  # Jensen-Shannon: symmetric, bounded
            mixture_log_probs = torch.log((0.5 * (student_probs + teacher_probs)).clamp_min(1e-12))
            teacher_side = (teacher_probs * (teacher_log_probs - mixture_log_probs)).sum(dim=-1)
            student_side = (student_probs * (student_log_probs - mixture_log_probs)).sum(dim=-1)
            return 0.5 * teacher_side + 0.5 * student_side
        if self.divergence == 'tvd':  # total variation distance
            return 0.5 * (student_probs - teacher_probs).abs().sum(dim=-1)
        raise ValueError(f"Unknown divergence {self.divergence!r}; use forward_kl, reverse_kl, js or tvd")

    def kl_ratio_per_token(self, teacher_probs: torch.Tensor, scheduled_ratio: float) -> torch.Tensor:
        """Weight of the divergence term per token. entropy_kl_weighting blends the scheduled
        ratio with the teacher's normalized entropy at that token (0..1, over log V), so
        confident tokens lean on cross-entropy and uncertain tokens on the divergence."""
        token_count = teacher_probs.shape[0]
        if self.entropy_kl_weighting <= 0:
            return torch.full((token_count,), scheduled_ratio, device=teacher_probs.device)
        entropy = -(teacher_probs * torch.log(teacher_probs.clamp_min(1e-12))).sum(dim=-1)
        normalized_entropy = entropy / math.log(teacher_probs.shape[-1])
        weight = self.entropy_kl_weighting
        return (1 - weight) * scheduled_ratio + weight * normalized_entropy
