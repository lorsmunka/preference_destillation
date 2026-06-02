import json
from time import time


class InputHandler:

    def __init__(self, file_path: str):
        self.inputs = []
        self.input_count = 0
        self.load_inputs(file_path)

    def load_inputs(self, file_path: str):
        start_time = time()
        print("Loading inputs...")
        inputs = []
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line.strip())
                inputs.append(data["text"])

        self.inputs = inputs
        self.input_count = len(inputs)

        elapsed_time = time() - start_time
        print(
            f"Loaded {len(inputs):,} -> took {elapsed_time:.2f} seconds. \n")

    def get_input(self, index):
        return self.inputs[index]
