from time import time
from typing import Dict, List, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache

from library.shared.vocabulary import (
    build_vocabulary,
    get_response_tokens,
    extract_logits_as_vector,
)
from library.shared.config import teacher_weights_path
from library.shared.device import get_device
from library.domain import as_domain


MAX_THINKING_TOKENS = 1024
THOUGHT_SCAFFOLD = "thought\n<channel|>"


class ModelHandler:
    def __init__(self, model_name: str, domain, load_in_4bit: bool = False,
                 chat_template: bool = False, thinking: bool = False):
        self.model_name = model_name
        self.domain_handler = as_domain(domain)  # accepts a Domain object or a name string
        self.max_generation_steps = self.domain_handler.max_steps
        self.load_in_4bit = load_in_4bit
        self.chat_template = chat_template or thinking
        self.thinking = thinking
        self.tokenizer = None
        self.model = None
        self.vocabulary: Optional[dict] = None
        self.response_tokens: Optional[List[str]] = None
        self.thought_end_token_id: Optional[int] = None
        self.last_generated_text = ""   # kept even when the example is rejected, for diagnostics

        self.load_tokenizer_and_model()

    def load_tokenizer_and_model(self):
        start_time = time()
        weights = teacher_weights_path(self.model_name)   # local re-sharded copy when one exists
        print(f"Loading tokenizer and model{' (4-bit NF4)' if self.load_in_4bit else ''} from {weights}...")
        self.tokenizer = AutoTokenizer.from_pretrained(weights)
        load_kwargs = dict(device_map="auto", dtype=torch.bfloat16)
        if self.load_in_4bit:
            # bitsandbytes NF4 keeps the weights 4-bit at runtime (~7 GB for 12B). Note that
            # compressed-tensors checkpoints (e.g. *-qat-w4a16-ct) cannot be used instead: this
            # transformers stack unpacks them to bf16 on the first forward pass.
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        self.model = AutoModelForCausalLM.from_pretrained(weights, **load_kwargs)
        self.end_token_ids = self.find_end_token_ids()
        if self.thinking:
            self.thought_end_token_id = self.find_thought_end_token()
        elapsed_time = time() - start_time
        print(
            f"Loaded tokenizer and model -> took {elapsed_time:.2f} seconds.\n")

        self.create_vocabulary()

    def find_end_token_ids(self) -> set:
        """EOS plus the end-of-turn token of whichever Gemma generation this is; generation
        stops on any of them (the domain's own stop token normally fires first)."""
        ids = {self.tokenizer.eos_token_id} if self.tokenizer.eos_token_id is not None else set()
        vocabulary = self.tokenizer.get_vocab()
        for token in ("<turn|>", "<end_of_turn>"):
            if token in vocabulary:
                ids.add(vocabulary[token])
        return ids

    def find_thought_end_token(self) -> int:
        """The special token that closes Gemma 4's thinking channel. Fails loudly on a
        tokenizer that has no such token, rather than silently generating without it."""
        candidates = [token for token in self.tokenizer.get_vocab() if "channel" in token and "|" in token]
        closers = [token for token in candidates if token.startswith("<channel")]
        if not closers:
            raise ValueError(f"{self.model_name}: thinking=True but no thought-channel close token in the "
                             f"tokenizer (channel-like tokens: {candidates[:6]})")
        return self.tokenizer.convert_tokens_to_ids(closers[0])

    def create_vocabulary(self):
        start_time = time()
        print("Creating vocabulary...")

        self.vocabulary = build_vocabulary(self.tokenizer, self.domain_handler)
        self.response_tokens = get_response_tokens(
            self.tokenizer, self.domain_handler)

        elapsed_time = time() - start_time
        print(f"Created vocabulary -> took {elapsed_time:.2f} seconds.\n")

    def build_prompt(self, text: str) -> str:
        prompt = self.domain_handler.teacher_prompt(text)
        if not self.chat_template:
            return prompt
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=self.thinking)

    def generate_thinking(self, input_ids: torch.Tensor):
        """Let the teacher think (greedy, plain generate — no logits kept), stopping when the
        thought channel closes. Returns (sequence including the thought, thought text)."""
        with torch.no_grad():
            output = self.model.generate(
                input_ids, attention_mask=torch.ones_like(input_ids), do_sample=False,
                max_new_tokens=MAX_THINKING_TOKENS, eos_token_id=self.thought_end_token_id,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        new_tokens = output[0, input_ids.shape[1]:]
        thought = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return output, thought

    def is_stop(self, last_token_decoded: str, generated_text: str) -> bool:
        return self.domain_handler.is_stop(last_token_decoded, generated_text)

    def generate_training_example(self, text: str) -> Tuple[Optional[Dict], Optional[str]]:
        start_time = time()

        prompt = self.build_prompt(text)
        inputs = self.prepare_inputs_for_device(prompt)

        steps = []
        generated_text = ""
        sequence = inputs['input_ids']
        thinking = ""
        if self.thinking:
            sequence, thinking = self.generate_thinking(sequence)

        # KV-cached decoding: the prompt (and thought) is fed once, then one token per step —
        # the same path model.generate() takes. Re-feeding the whole prefix every step, as the
        # old loop did, gave different (worse) logits on Gemma 4 through bitsandbytes.
        cache = DynamicCache()
        pending = sequence
        last_token_decoded = ""
        skipped_scaffold = ""   # Gemma 4 sometimes opens "thought\n<channel|>" before answering, thinking or not
        while not self.is_stop(last_token_decoded, generated_text) and len(steps) <= self.max_generation_steps:
            token_repr, token_decoded, logit_vector, predicted_token_index, next_token_id = self.generate_single_step(
                pending, cache, total_length=sequence.shape[1])
            if next_token_id in self.end_token_ids:
                break   # end of sequence / end of turn: not part of the response

            if not steps and skipped_scaffold is not None and THOUGHT_SCAFFOLD.startswith(skipped_scaffold + token_decoded):
                skipped_scaffold += token_decoded   # advance past the scaffold without recording it
                pending = torch.tensor([[next_token_id]], device=sequence.device)
                sequence = torch.cat([sequence, pending], dim=1)
                continue
            skipped_scaffold = None

            steps.append({
                "token": token_repr,
                "logits": logit_vector,
                "predicted_token_index": predicted_token_index
            })

            generated_text += token_decoded
            pending = torch.tensor([[next_token_id]], device=sequence.device)
            sequence = torch.cat([sequence, pending], dim=1)
            last_token_decoded = token_decoded

        elapsed_time = time() - start_time
        self.last_generated_text = generated_text

        response_tokens = self.tokenizer.tokenize(generated_text)
        print(
            f"\tGenerated training example length of {len(response_tokens)} tokens -> took {elapsed_time:.2f} seconds.")

        if not all(token in self.response_tokens for token in response_tokens):
            unexpected_tokens = [
                token for token in response_tokens if token not in self.response_tokens]
            print(
                f"\tSkipped: Generated response contains unexpected tokens: {unexpected_tokens}")
            return None, "unexpected_tokens"

        if len(response_tokens) > self.max_generation_steps:
            print(
                f"\tSkipped: Generated response is too long ({len(response_tokens)} tokens).")
            return None, "too_long"

        example = {
            "sentence": text,
            "model_response": generated_text,
            "steps": steps
        }
        if self.thinking:
            example["thinking"] = thinking
        return example, None

    def generate_single_step(self, pending: torch.Tensor, cache: DynamicCache,
                             total_length: int) -> Tuple[str, str, List[float], int, int]:
        """Feed the not-yet-cached tokens (`pending`) through the model with the KV cache and
        return the next token plus the full logit vector over the reduced vocabulary."""
        attention_mask = torch.ones((1, total_length), dtype=torch.long, device=pending.device)
        with torch.no_grad():
            output = self.model(input_ids=pending, attention_mask=attention_mask,
                                past_key_values=cache, use_cache=True)
        logits = output.logits[0, -1, :]

        logit_vector = extract_logits_as_vector(logits, self.vocabulary)
        predicted_token_index = int(max(range(len(logit_vector)), key=lambda i: logit_vector[i]))

        next_token_id = torch.argmax(logits).item()
        token_repr = self.tokenizer.convert_ids_to_tokens([next_token_id])[0]
        token_decoded = self.tokenizer.decode([next_token_id])

        return token_repr, token_decoded, logit_vector, predicted_token_index, next_token_id

    def prepare_inputs_for_device(self, text: str) -> Dict[str, torch.Tensor]:
        inputs = self.tokenizer(text, return_tensors="pt")
        device = get_device()

        if device != "cpu":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        return inputs

    def get_vocabulary(self) -> dict:
        return self.vocabulary

    def get_vocab_size(self) -> int:
        return self.vocabulary['vocab_size']

    def get_positions(self) -> dict:
        return self.vocabulary['positions']
