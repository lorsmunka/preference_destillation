from time import sleep, time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from Utilities import Utilities

from typing import Dict, List, Tuple


class ModelHandler:
    def __init__(self):
        self.model_name = "google/gemma-3-4b-it"
        self.tokenizer = None
        self.model = None
        self.token_to_id_map = None

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

        self.create_vocabulary_map()

    def create_vocabulary_map(self):
        start_time = time()
        print("Creating vocabulary map...")

        self.token_to_id_map = Utilities.build_vocabulary_map(self.tokenizer)

        elapsed_time = time() - start_time
        print(f"Created vocabulary map -> took {elapsed_time:.2f} seconds. \n")

    def generate_training_example(self, sentence):
        start_time = time()

        prompt = Utilities.create_evaluation_prompt(sentence)
        inputs = self.prepare_inputs_for_gpu(prompt)

        steps = []
        generated_text = ""
        current_sequence = inputs['input_ids']

        last_token = ""
        while last_token != "}" and len(steps) < 100:
            next_token, step_probs, new_sequence = self.generate_single_step(
                current_sequence)

            steps.append({
                "token": next_token,
                "probabilities": step_probs
            })

            generated_text += next_token
            current_sequence = new_sequence
            last_token = next_token

        elapsed_time = time() - start_time

        print(
            f"Generated training example -> took {elapsed_time:.2f} seconds.")

        return {
            "sentence": sentence,
            "model_response": generated_text,
            "steps": steps
        }

    def generate_single_step(self, current_sequence):
        current_inputs = {
            'input_ids': current_sequence,
            'attention_mask': torch.ones_like(current_sequence)
        }

        with torch.no_grad():
            logits = self.model(
                **current_inputs).logits[0, -1, :]

        token_names, token_logits = Utilities.extract_logits(
            logits, self.token_to_id_map)
        step_probs = self.sort_by_logits(token_names, token_logits)

        next_token_id = torch.argmax(logits).item()
        next_token = self.tokenizer.decode([next_token_id])

        new_sequence = torch.cat([current_sequence, torch.tensor(
            [[next_token_id]], device=current_sequence.device)], dim=1)

        return next_token, step_probs, new_sequence

    def sort_by_logits(self, token_names: List[str], token_logits: List[float]) -> Dict[str, float]:
        return dict(sorted(zip(token_names, token_logits), key=lambda x: x[1], reverse=True))

    def prepare_inputs_for_gpu(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt")

        if torch.cuda.is_available():
            new_inputs = {}
            for k, v in inputs.items():
                new_inputs[k] = v.cuda()
            inputs = new_inputs

        return inputs
