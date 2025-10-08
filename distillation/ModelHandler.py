from time import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelHandler:
    def __init__(self):
        self.model_name = "google/gemma-3-4b-it"
        self.tokenizer = None
        self.model = None

        self.load_tokenizer_and_model()

    def load_tokenizer_and_model(self):
        start_time = time()
        print("\nLoading tokenizer and model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            dtype=torch.bfloat16
        )
        elapsed_time = time() - start_time
        print(
            f"Loaded tokenizer and model -> took {elapsed_time:.2f} seconds.\n")

    def generate_training_example(self, sentence):
        start_time = time()

        elapsed_time = time() - start_time
        print(
            f"Generated training example -> took {elapsed_time:.2f} seconds.")

        return {
            "sentence": sentence,
            "generated_text": f"Generated text for: {sentence}"
        }
