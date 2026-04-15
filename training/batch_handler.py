import json
import os
from time import time


class BatchHandler:
    def __init__(self, batches_dir: str, training_test_ratio: float, max_training_examples: int = None, batch_size: int = 32):
        self.batches_dir = batches_dir
        self.training_test_ratio = training_test_ratio
        self.max_training_examples = max_training_examples
        self.batch_size = batch_size

    def get_batch(self, index):
        batch_index = index + 1
        print(f"Loading batch {batch_index}...")
        start_time = time()

        with open(os.path.join(self.batches_dir, f"batch_{batch_index}.jsonl"), "r", encoding="utf-8") as file:
            batch = [json.loads(line.strip()) for line in file]

        elapsed_time = time() - start_time
        print(
            f"Loaded batch {batch_index} -> took {elapsed_time:.2f} seconds.")

        return batch

    def get_batch_count(self):
        files = os.listdir(self.batches_dir)
        batch_files = [f for f in files if f.startswith(
            "batch_") and f.endswith(".jsonl")]
        return len(batch_files)

    def _effective_batch_count(self):
        total_batches = self.get_batch_count()
        if self.max_training_examples is not None:
            max_batches = self.max_training_examples // self.batch_size
            return min(total_batches, max_batches)
        return total_batches

    def get_training_batches_radius(self):
        effective = self._effective_batch_count()
        return 0, int(effective * self.training_test_ratio)

    def get_test_batches_radius(self):
        effective = self._effective_batch_count()
        return int(effective * self.training_test_ratio), effective
