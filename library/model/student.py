"""StudentModel — the one place that builds the student, loads a checkpoint, and generates.

Replaces the 4 copies of model construction and the 3 copies of the rollout loop
(evaluate_model.py, inference.py, top-k tool, trainer). Checkpoints load through
Transformer.load_weights (strict, the same loader the trainer uses), and generation uses the
domain's own stop rule (domain.is_stop), as the trainer's evaluation does.

Model path only — reloading the model is allowed here because this is the interactive
demo/eval/test path, not post-run log analysis. Not exercised in the analytics test pass
(needs batch data + the gated Gemma tokenizer); ported from the working originals.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from library.shared.config import load_input_vocabulary
from library.domain import Domain, as_domain


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
    def from_run(cls, run, checkpoint_path: Path, device: Optional[str] = None,
                 domain: Optional[Domain] = None) -> "StudentModel":
        from library.model.custom_student import build_student  # lazy: pulls torch/tokenizer only on model path

        device = device or pick_device()
        info = run.info
        if domain is None:  # built-in runs resolve by name; custom runs pass the object
            try:
                domain = as_domain(info["domain"])
            except KeyError:
                raise ValueError(
                    f"Run {run.name!r} uses custom domain {info['domain']!r} — pass "
                    f"domain=YourDomain() to analyse it (it isn't one of the built-ins)."
                )
        model = build_student(info, domain, cls._input_vocabulary(run)).to(device)
        cls._check_output_vocabulary(run, model, domain)
        cls._load_checkpoint(model, Path(checkpoint_path), device)
        model.eval()
        return cls(model, domain, device)

    @staticmethod
    def _input_vocabulary(run) -> Optional[dict]:
        """The run's own snapshot (runs trained since 2026-10-08 have one), else the batches'."""
        run_dir = getattr(run, "run_dir", None)
        snapshot_path = Path(run_dir) / "input_vocabulary.json" if run_dir is not None else None
        if snapshot_path is not None and snapshot_path.exists():
            with open(snapshot_path, encoding="utf-8") as file:
                return json.load(file)
        return load_input_vocabulary(run.info["domain"], run.info["teacher_model"])

    @staticmethod
    def _check_output_vocabulary(run, model, domain: Domain) -> None:
        """The rebuilt output vocabulary must be the one the run trained with (recorded in
        info.json since 2026-10-08), or every logit would name the wrong token."""
        from library.shared.batch_files import vocabulary_fingerprint
        from library.shared.vocabulary import build_vocabulary
        recorded = run.info.get("vocabulary_sha1")
        if recorded is None:
            return
        rebuilt = vocabulary_fingerprint(build_vocabulary(model.tokenizer, domain)["token_list"])
        if rebuilt != recorded:
            raise ValueError(
                f"Run {run.name!r} trained with reduced vocabulary {recorded}, but the domain now "
                f"builds {rebuilt}: its tokens changed since, so this checkpoint cannot be read with it.")

    @staticmethod
    def _load_checkpoint(model, checkpoint_path: Path, device: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_weights(checkpoint["model_state_dict"])   # strict; drops stored rotary tables

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
            # rows may be float32 numpy views (.npz batches) or lists (.jsonl): one cast covers both
            "teacher_logits": torch.from_numpy(np.asarray(teacher_logits, dtype=np.float32)).to(self.device),
            "target_indices": torch.tensor(target_indices, dtype=torch.long, device=self.device),
        }

    @torch.no_grad()
    def generate(self, text: str, max_new_tokens: int, temperature: float = 0.0) -> GenerationResult:
        """Free-generation rollout: stop on the domain's stop rule (domain.is_stop, the same
        rule as the trainer's evaluation) or after max_new_tokens. `terminated` records whether
        it stopped on the rule (the natural-termination signal)."""
        from library.model.decoding import decoder_for  # lazy, like build_student in from_run

        sentence_ids = self.tokenizer.encode(self.domain.student_prompt(text), add_special_tokens=False)
        decoder = decoder_for(self.model, [self.model.remap_input_tokens(sentence_ids)])
        generated_ids: List[int] = []
        token_strings: List[str] = []
        predicted_indices: List[int] = []
        generated_text = ""
        terminated = False

        for _ in range(max_new_tokens):
            logits = decoder.next_logits()[0]
            if temperature:
                probabilities = F.softmax(logits / temperature, dim=-1)
                predicted_index = torch.multinomial(probabilities, num_samples=1).item()
            else:
                predicted_index = torch.argmax(logits).item()

            predicted_indices.append(predicted_index)
            token_id = self.output_token_ids[predicted_index]
            generated_ids.append(token_id)
            decoded = self.tokenizer.decode([token_id])
            token_strings.append(decoded)
            generated_text += decoded

            if self.domain.is_stop(decoded, generated_text):
                terminated = True
                break
            decoder.append(self.model.remap_input_tokens([token_id]))

        return GenerationResult(
            text=self.tokenizer.decode(generated_ids, skip_special_tokens=True),
            token_strings=token_strings,
            token_ids=generated_ids,
            predicted_indices=predicted_indices,
            terminated=terminated,
            steps=len(generated_ids),
        )
