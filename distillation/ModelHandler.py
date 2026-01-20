from time import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from shared import Utilities, MODEL_NAME, MAX_GENERATION_STEPS

from typing import Dict, List, Tuple, Optional


class ModelHandler:
    def __init__(self):
        self.model_name = MODEL_NAME
        self.tokenizer = None
        self.model = None
        self.vocabulary: Optional[dict] = None
        self.json_response_tokens: Optional[List[str]] = None

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

        self.vocabulary = Utilities.build_vocabulary(self.tokenizer)
        self.json_response_tokens = Utilities.get_json_response_tokens(self.tokenizer)

        elapsed_time = time() - start_time
        print(f"Created vocabulary -> took {elapsed_time:.2f} seconds.\n")

    def generate_training_example(self, sentence: str) -> Optional[Dict]:
        start_time = time()

        prompt = Utilities.create_evaluation_prompt(sentence)
        inputs = self.prepare_inputs_for_gpu(prompt)

        steps = []
        generated_text = ""
        current_sequence = inputs['input_ids']

        last_token = ""
        while last_token != "}" and len(steps) <= MAX_GENERATION_STEPS:
            next_token, logit_vector, predicted_token_index, new_sequence = self.generate_single_step(
                current_sequence)

            steps.append({
                "token": next_token,
                "logits": logit_vector,
                "predicted_token_index": predicted_token_index
            })

            generated_text += next_token
            current_sequence = new_sequence
            last_token = next_token

        elapsed_time = time() - start_time

        response_tokens = self.tokenizer.tokenize(generated_text)
        print(
            f"Generated training example length of {len(response_tokens)} tokens -> took {elapsed_time:.2f} seconds.")

        if not all(token in self.json_response_tokens for token in response_tokens):
            unexpected_tokens = [
                token for token in response_tokens if token not in self.json_response_tokens]
            print(
                f"Skipped: Generated response contains unexpected tokens: {unexpected_tokens}")
            return None

        if len(response_tokens) > MAX_GENERATION_STEPS:
            print(
                f"Skipped: Generated response is too long ({len(response_tokens)} tokens).")
            return None

        return {
            "sentence": sentence,
            "model_response": generated_text,
            "steps": steps
        }

    def generate_single_step(self, current_sequence: torch.Tensor) -> Tuple[str, List[float], int, torch.Tensor]:
        current_inputs = {
            'input_ids': current_sequence,
            'attention_mask': torch.ones_like(current_sequence)
        }

        with torch.no_grad():
            logits = self.model(**current_inputs).logits[0, -1, :]

        logit_vector = Utilities.extract_logits_as_vector(logits, self.vocabulary)
        predicted_token_index = int(max(range(len(logit_vector)), key=lambda i: logit_vector[i]))

        next_token_id = torch.argmax(logits).item()
        next_token = self.tokenizer.decode([next_token_id])

        new_sequence = torch.cat([current_sequence, torch.tensor(
            [[next_token_id]], device=current_sequence.device)], dim=1)

        return next_token, logit_vector, predicted_token_index, new_sequence

    def prepare_inputs_for_gpu(self, text: str) -> Dict[str, torch.Tensor]:
        inputs = self.tokenizer(text, return_tensors="pt")

        if torch.cuda.is_available():
            new_inputs = {}
            for k, v in inputs.items():
                new_inputs[k] = v.cuda()
            inputs = new_inputs

        return inputs

    def get_vocabulary(self) -> dict:
        return self.vocabulary

    def get_vocab_size(self) -> int:
        return self.vocabulary['vocab_size']

    def get_positions(self) -> dict:
        return self.vocabulary['positions']
