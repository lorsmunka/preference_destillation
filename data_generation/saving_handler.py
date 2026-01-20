import os
import json
from time import time

from shared import BATCHES_DIR, Logger


class SavingHandler:
    def __init__(self, logger: Logger):
        self.logger = logger

    def save_batch(self, batch_examples):
        start_time = time()

        if not os.path.exists(BATCHES_DIR):
            os.makedirs(BATCHES_DIR)

        path = f"{BATCHES_DIR}/batch_{self.logger.batch_count}.jsonl"
        with open(path, "w", encoding="utf-8") as file:
            for example in batch_examples:
                file.write(f"{json.dumps(example)}\n")

        elapsed_time = time() - start_time
        print(f"\nSaved batch to {path} -> took {elapsed_time:.2f} seconds.\n")
