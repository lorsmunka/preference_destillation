from time import time
from typing import Dict, List, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from shared import Utilities, MAX_GENERATION_STEPS, get_device


class ModelHandler:
    def __init__(self, model_name: str, domain: str):
        self.model_name = model_name
        self.domain = domain
        self.tokenizer = None
        self.model = None
        self.vocabulary: Optional[dict] = None
        self.response_tokens: Optional[List[str]] = None

        self.load_tokenizer_and_model()

    def load_tokenizer_and_model(self):
        start_time = time()
        print("Loading tokenizer and model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            dtype=torch.bfloat16
        )
        elapsed_time = time() - start_time
        print(
            f"Loaded tokenizer and model -> took {elapsed_time:.2f} seconds.\n")

        self.create_vocabulary()

    def create_vocabulary(self):
        start_time = time()
        print("Creating vocabulary...")

        self.vocabulary = Utilities.build_vocabulary(self.tokenizer, self.domain)
        self.response_tokens = Utilities.get_response_tokens(
            self.tokenizer, self.domain)

        elapsed_time = time() - start_time
        print(f"Created vocabulary -> took {elapsed_time:.2f} seconds.\n")

    def build_prompt(self, text: str) -> str:
        if self.domain == "math_word_problem":
            return Utilities.create_math_word_problem_prompt(text)
        return Utilities.create_reddit_sentiment_prompt(text)

    def is_stop_token(self, token_decoded: str, generated_text: str) -> bool:
        if self.domain == "math_word_problem":
            lines = generated_text.strip().split("\n")
            if lines:
                last_line = lines[-1].strip()
                if last_line.startswith("Solution:") and len(last_line) > len("Solution:"):
                    return True
            return False
        return token_decoded == "}"

    def generate_training_example(self, text: str) -> Tuple[Optional[Dict], Optional[str]]:
        start_time = time()

        prompt = self.build_prompt(text)
        inputs = self.prepare_inputs_for_device(prompt)

        steps = []
        generated_text = ""
        current_sequence = inputs['input_ids']

        last_token_decoded = ""
        while not self.is_stop_token(last_token_decoded, generated_text) and len(steps) <= MAX_GENERATION_STEPS:
            token_repr, token_decoded, logit_vector, predicted_token_index, new_sequence = self.generate_single_step(
                current_sequence)

            steps.append({
                "token": token_repr,
                "logits": logit_vector,
                "predicted_token_index": predicted_token_index
            })

            generated_text += token_decoded
            current_sequence = new_sequence
            last_token_decoded = token_decoded

        elapsed_time = time() - start_time

        response_tokens = self.tokenizer.tokenize(generated_text)
        print(
            f"\tGenerated training example length of {len(response_tokens)} tokens -> took {elapsed_time:.2f} seconds.")

        if not all(token in self.response_tokens for token in response_tokens):
            unexpected_tokens = [
                token for token in response_tokens if token not in self.response_tokens]
            print(
                f"\tSkipped: Generated response contains unexpected tokens: {unexpected_tokens}")
            return None, "unexpected_tokens"

        if len(response_tokens) > MAX_GENERATION_STEPS:
            print(
                f"\tSkipped: Generated response is too long ({len(response_tokens)} tokens).")
            return None, "too_long"

        return {
            "sentence": text,
            "model_response": generated_text,
            "steps": steps
        }, None

    def generate_single_step(self, current_sequence: torch.Tensor) -> Tuple[str, str, List[float], int, torch.Tensor]:
        current_inputs = {
            'input_ids': current_sequence,
            'attention_mask': torch.ones_like(current_sequence)
        }

        with torch.no_grad():
            logits = self.model(**current_inputs).logits[0, -1, :]

        logit_vector = Utilities.extract_logits_as_vector(
            logits, self.vocabulary)
        predicted_token_index = int(
            max(range(len(logit_vector)), key=lambda i: logit_vector[i]))

        next_token_id = torch.argmax(logits).item()
        token_repr = self.tokenizer.convert_ids_to_tokens([next_token_id])[0]
        token_decoded = self.tokenizer.decode([next_token_id])

        new_sequence = torch.cat([current_sequence, torch.tensor(
            [[next_token_id]], device=current_sequence.device)], dim=1)

        return token_repr, token_decoded, logit_vector, predicted_token_index, new_sequence

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
