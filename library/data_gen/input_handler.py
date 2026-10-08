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
        records = []
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                records.append(json.loads(line.strip()))

        self.records = records   # whole corpus records; a domain with ground truth puts "answer" there
        self.inputs = [record["text"] for record in records]
        inputs = self.inputs
        self.input_count = len(inputs)

        elapsed_time = time() - start_time
        print(
            f"Loaded {len(inputs):,} -> took {elapsed_time:.2f} seconds. \n")

    def get_input(self, index):
        return self.inputs[index]

    def get_target(self, index):
        """The corpus record's gold "target" text (teacher-forced scoring), or None."""
        return self.records[index].get("target")

    def get_teacher_mode(self, index):
        """The record's own teacher mode ("generate" / "forced"), or None to use the domain's."""
        return self.records[index].get("teacher")

    def get_ground_truth(self, index):
        """The corpus record's "answer", when the domain has one (else None)."""
        return self.records[index].get("answer")

    def get_metadata(self, index):
        """The record's split, item id and turn type (None where the record has none), carried
        into the batches so training can split on them instead of on file order."""
        record = self.records[index]
        return {"split": record.get("split"), "item_id": record.get("item_id"), "turn_type": record.get("turn_type")}
