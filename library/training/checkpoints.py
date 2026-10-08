"""Checkpoint files. Per-epoch checkpoints hold weights only (13.6 MB for the default
student); the optimizer state (27 MB) is written only into the resume checkpoint, which
is the one place it is needed."""

import os
from time import time
from typing import Optional

import torch


def save_checkpoint(filepath: str, model, optimizer, epoch: int, current_step: int,
                    train_loss: Optional[float], include_optimizer: bool = False,
                    steps_per_epoch: Optional[int] = None) -> None:
    start_time = time()
    checkpoint = {
        'epoch': epoch,
        'current_step': current_step,
        'model_state_dict': model.state_dict(),
        'train_loss': train_loss,
    }
    if include_optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if steps_per_epoch is not None:
        checkpoint['steps_per_epoch'] = steps_per_epoch

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    temporary_path = filepath + '.partial'   # written fully, then renamed: a crash never leaves half a checkpoint
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, filepath)
    print(f"Checkpoint saved: {filepath} -> took {time() - start_time:.2f}s\n")


def read_checkpoint(filepath: str, device) -> dict:
    """The checkpoint dict, without touching any model."""
    return torch.load(filepath, map_location=device, weights_only=True)


def load_checkpoint(filepath: str, model, device) -> dict:
    """Loads the weights into `model` (strictly, see Transformer.load_weights) and returns
    the checkpoint dict, so the caller can restore the optimizer state and the step counter."""
    checkpoint = read_checkpoint(filepath, device)
    model.load_weights(checkpoint['model_state_dict'])
    return checkpoint
