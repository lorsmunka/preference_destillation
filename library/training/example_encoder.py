"""Turns one stored example (sentence + teacher steps) into the tensors the student
consumes. Shared by training and evaluation so they cannot drift apart."""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from library.shared import PROMPT_DELIMITER


@dataclass
class EncodedExample:
    sentence_tokens: List[int]        # teacher tokenizer ids of the prompt (not remapped)
    target_indices: List[int]         # index into the reduced output vocabulary, per step
    input_tensor: torch.Tensor        # [1, prompt + steps - 1], remapped to the compact input vocab
    target_logits_tensor: torch.Tensor   # [steps, V]
    target_indices_tensor: torch.Tensor  # [steps]

    @property
    def num_steps(self) -> int:
        return len(self.target_indices)


class ExampleEncoder:
    def __init__(self, model, device: str):
        self.model = model
        self.tokenizer = model.tokenizer
        self.device = device
        self.vocab_size = model.vocabulary['vocab_size']
        self.output_token_to_index = {
            token: index for index, token in enumerate(model.vocabulary['token_list'])
        }

    def sentence_tokens(self, example: Dict) -> List[int]:
        text = example['sentence'] + PROMPT_DELIMITER
        return self.tokenizer.encode(text, add_special_tokens=False)

    def step_data(self, step: Dict):
        token = step['token']
        logit_vector = step['logits'][:self.vocab_size]
        target_index = self.output_token_to_index.get(token, step['predicted_token_index'])
        token_ids = self.tokenizer.encode(token, add_special_tokens=False)
        token_id = token_ids[0] if token_ids else self.tokenizer.unk_token_id
        return token_id, logit_vector, target_index

    def encode(self, example: Dict) -> EncodedExample:
        sentence_tokens = self.sentence_tokens(example)
        token_ids, target_logits, target_indices = [], [], []
        for step in example['steps']:
            token_id, logit_vector, target_index = self.step_data(step)
            token_ids.append(token_id)
            target_logits.append(logit_vector)
            target_indices.append(target_index)

        input_ids = self.model.remap_input_tokens(sentence_tokens + token_ids[:-1])
        return EncodedExample(
            sentence_tokens=sentence_tokens,
            target_indices=target_indices,
            input_tensor=torch.tensor([input_ids], dtype=torch.long, device=self.device),
            # rows may be float16 numpy views (.npz batches) or lists (.jsonl) — one cast covers both
            target_logits_tensor=torch.from_numpy(np.asarray(target_logits, dtype=np.float32)).to(self.device),
            target_indices_tensor=torch.tensor(target_indices, dtype=torch.long, device=self.device),
        )
