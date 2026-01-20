import json
from time import time

from shared import BATCHES_DIR, BATCH_SIZE, TRAINING_TEST_RATIO


class BatchHandler:
    def get_batch(self, index):
        batch_index = index + 1
        print(f"Loading batch {batch_index}...")
        start_time = time()
        batch = []

        with open(f"{BATCHES_DIR}/batch_{batch_index}.jsonl", "r", encoding="utf-8") as file:
            batch = [json.loads(line.strip()) for line in file]

        elapsed_time = time() - start_time
        print(
            f"Loaded batch {batch_index} -> took {elapsed_time:.2f} seconds.")

        return batch

    def get_batch_size(self):
        return BATCH_SIZE

    def get_batch_count(self):
        import os
        files = os.listdir(BATCHES_DIR)
        batch_files = [f for f in files if f.startswith(
            "batch_") and f.endswith(".jsonl")]
        return len(batch_files)

    def get_training_batches_radius(self):
        total_batches = self.get_batch_count()
        return 0, int(total_batches * TRAINING_TEST_RATIO)

    def get_test_batches_radius(self):
        total_batches = self.get_batch_count()
        return int(total_batches * TRAINING_TEST_RATIO), total_batches
