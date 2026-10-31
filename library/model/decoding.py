"""Decoding for the per-pass test and the demo: the built-in Transformer's fast IncrementalDecoder
(key/value cache), or SimpleDecoder for any other student. Both have the same three methods."""

from typing import List

import torch

from library.model.transformer import IncrementalDecoder, Transformer


class SimpleDecoder:
    """Autoregressive decoding for a model without a key/value cache (a custom student): every step
    re-reads each sequence's whole prefix on its own. Slower than IncrementalDecoder, but it needs
    nothing from the model beyond forward(input_ids). Call it under torch.no_grad()."""

    def __init__(self, model, prompts: List[List[int]]):
        self.model = model
        self.device = next(model.parameters()).device
        self.sequences = [list(prompt) for prompt in prompts]
        self.last_logits = self._last_logits()

    def _last_logits(self) -> torch.Tensor:
        return torch.stack([self.model(torch.tensor([sequence], dtype=torch.long, device=self.device))[0, -1]
                            for sequence in self.sequences])

    def next_logits(self) -> torch.Tensor:
        return self.last_logits

    def append(self, input_ids: List[int]) -> None:
        for sequence, input_id in zip(self.sequences, input_ids):
            sequence.append(input_id)
        self.last_logits = self._last_logits()

    def keep_rows(self, rows: List[int]) -> None:
        self.sequences = [self.sequences[row] for row in rows]
        self.last_logits = self.last_logits[rows]


def decoder_for(model, prompts: List[List[int]]):
    if isinstance(model, Transformer):
        return IncrementalDecoder(model, prompts)
    return SimpleDecoder(model, prompts)
