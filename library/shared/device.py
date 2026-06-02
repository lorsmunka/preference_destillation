"""The one torch dependency that used to live in config. Kept separate so importing
`library.shared` / domain / tooling stays torch-free."""

import torch


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
