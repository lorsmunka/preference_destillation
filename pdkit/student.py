"""StudentModel — the one place that builds the student, loads a checkpoint, and generates.

Replaces the 4 copies of model construction and the 3 copies of the rollout loop
(evaluate_model.py, inference.py, top-k tool, trainer). Centralizes the checkpoint load so
the rotary-buffer pop (which evaluate_model.py / inference.py were missing) happens everywhere.

Model path only — reloading the model is allowed here because this is the interactive
demo/eval/test path, not post-run log analysis. Not exercised in the analytics test pass
(needs batch data + the gated Gemma tokenizer); ported from the working originals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from .config import load_input_vocabulary
from .domains import Domain, get_domain


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class GenerationResult:
    text: str
    token_strings: List[str] = field(default_factory=list)
    token_ids: List[int] = field(default_factory=list)
    predicted_indices: List[int] = field(default_factory=list)  # output-vocab indices, for token accuracy
    terminated: bool = False
    steps: int = 0


class StudentModel:
    def __init__(self, model, domain: Domain, device: str):
        self.model = model
        self.domain = domain
        self.device = device
        self.tokenizer = model.tokenizer
        self.output_token_ids = model.output_token_ids
        self.vocab_size = model.vocabulary["vocab_size"]
        self.output_token_to_index = {
            token: index for index, token in enumerate(model.vocabulary["token_list"])
        }

    # ── construction ──────────────────────────────────────────────────
    @classmethod
    def from_run(cls, run, checkpoint_path: Path, device: Optional[str] = None) -> "StudentModel":
        from training.model import Transformer  # lazy: pulls torch/tokenizer only on model path

        device = device or pick_device()
        info = run.info
        input_vocabulary = load_input_vocabulary(info["domain"], info["teacher_model"])
        model = Transformer(
            domain=info["domain"],
            teacher_model=info["teacher_model"],
            hidden_dim=info["hidden_dim"],
            num_layers=info["num_layers"],
            num_heads=info["num_heads"],
            dropout=info.get("dropout", 0.15),
            auxiliary_token_percentage=info.get("auxiliary_token_percentage", 1.0),
            input_vocabulary=input_vocabulary,
        ).to(device)
        cls._load_checkpoint(model, Path(checkpoint_path), device)
        model.eval()
        return cls(model, get_domain(info["domain"]), device)

    @staticmethod
    def _load_checkpoint(model, checkpoint_path: Path, device: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        state = dict(checkpoint["model_state_dict"])
        # The fix: these non-persistent rotary buffers must be dropped before load_state_dict.
        # evaluate_model.py / inference.py omitted this; trainer + top-k did it.
        state.pop("rotary_embedding.cos_cached", None)
        state.pop("rotary_embedding.sin_cached", None)
        model.load_state_dict(state, strict=False)

    # ── forward primitives ────────────────────────────────────────────
    def _forward(self, remapped_ids: List[int]) -> torch.Tensor:
        tensor = torch.tensor([remapped_ids], dtype=torch.long, device=self.device)
        return self.model(tensor)[0]  # [seq, vocab]

    def _target_index(self, step: dict) -> int:
        return self.output_token_to_index.get(step["token"], step["predicted_token_index"])

    def _token_id(self, token: str) -> int:
        ids = self.tokenizer.encode(token, add_special_tokens=False)
        return ids[0] if ids else self.tokenizer.unk_token_id

    @torch.no_grad()
    def teacher_forced(self, example: dict) -> Optional[Dict[str, torch.Tensor]]:
        """Single forward pass over ground-truth context. Returns student prediction logits,
        teacher logits and target indices per step — everything the distribution metrics need."""
        steps = example.get("steps", [])
        if not steps:
            return None
        sentence_ids = self.tokenizer.encode(
            self.domain.student_prompt(example["sentence"]), add_special_tokens=False
        )
        token_ids = [self._token_id(s["token"]) for s in steps]
        teacher_logits = [s["logits"][: self.vocab_size] for s in steps]
        target_indices = [self._target_index(s) for s in steps]

        remapped = self.model.remap_input_tokens(sentence_ids + token_ids[:-1])
        logits = self._forward(remapped)
        start = len(sentence_ids) - 1
        prediction_logits = logits[start: start + len(steps), :]
        return {
            "prediction_logits": prediction_logits,
            "teacher_logits": torch.tensor(teacher_logits, dtype=torch.float32, device=self.device),
            "target_indices": torch.tensor(target_indices, dtype=torch.long, device=self.device),
        }

    @torch.no_grad()
    def generate(self, text: str, max_new_tokens: int, temperature: float = 0.0) -> GenerationResult:
        """Free-generation rollout: stop on the domain stop token or after max_new_tokens.
        `terminated` records whether the stop token appeared (the natural-termination signal)."""
        sentence_ids = self.tokenizer.encode(self.domain.student_prompt(text), add_special_tokens=False)
        remapped_context = self.model.remap_input_tokens(sentence_ids)
        generated_ids: List[int] = []
        remapped_generated: List[int] = []
        token_strings: List[str] = []
        predicted_indices: List[int] = []
        terminated = False

        for _ in range(max_new_tokens):
            logits = self._forward(remapped_context + remapped_generated)[-1]
            if temperature:
                probabilities = F.softmax(logits / temperature, dim=-1)
                predicted_index = torch.multinomial(probabilities, num_samples=1).item()
            else:
                predicted_index = torch.argmax(logits).item()

            predicted_indices.append(predicted_index)
            token_id = self.output_token_ids[predicted_index]
            generated_ids.append(token_id)
            remapped_generated.append(self.model.remap_input_tokens([token_id])[0])
            decoded = self.tokenizer.decode([token_id])
            token_strings.append(decoded)

            text_so_far = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
            if decoded == self.domain.stop_token or self.domain.stop_token in text_so_far:
                terminated = True
                break

        return GenerationResult(
            text=self.tokenizer.decode(generated_ids, skip_special_tokens=True),
            token_strings=token_strings,
            token_ids=generated_ids,
            predicted_indices=predicted_indices,
            terminated=terminated,
            steps=len(generated_ids),
        )
