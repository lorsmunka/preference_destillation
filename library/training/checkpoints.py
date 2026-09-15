"""Checkpoint files. Per-epoch checkpoints hold weights only (13.6 MB for the default
student); the optimizer state (27 MB) is written only into the resume checkpoint, which
is the one place it is needed."""

import os
from time import time

import torch


def save_checkpoint(filepath: str, model, optimizer, epoch: int, current_step: int,
                    train_loss: float, include_optimizer: bool = False) -> None:
    start_time = time()
    checkpoint = {
        'epoch': epoch,
        'current_step': current_step,
        'model_state_dict': model.state_dict(),
        'train_loss': train_loss,
    }
    if include_optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath} -> took {time() - start_time:.2f}s\n")


def load_checkpoint(filepath: str, model, device) -> dict:
    """Loads the weights into `model` and returns the checkpoint dict, so the caller can
    restore the optimizer state (if present) and the step counter."""
    checkpoint = torch.load(filepath, map_location=device, weights_only=True)
    model_state = checkpoint['model_state_dict']
    model_state.pop('rotary_embedding.cos_cached', None)
    model_state.pop('rotary_embedding.sin_cached', None)
    model.load_state_dict(model_state, strict=False)
    return checkpoint
