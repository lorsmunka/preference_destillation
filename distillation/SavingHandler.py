import os
import json
from time import time

from shared import DISTILLATION_BATCHES_DIR, DistillationTelemetryHandler


class SavingHandler:
    def __init__(self, telemetry_handler: DistillationTelemetryHandler):
        self.telemetry_handler = telemetry_handler

    def save_batch(self, batch_examples):
        start_time = time()

        if not os.path.exists(DISTILLATION_BATCHES_DIR):
            os.makedirs(DISTILLATION_BATCHES_DIR)

        path = f"{DISTILLATION_BATCHES_DIR}/batch_{self.telemetry_handler.batch_count}.jsonl"
        with open(path, "w", encoding="utf-8") as file:
            for example in batch_examples:
                file.write(f"{json.dumps(example)}\n")

        elapsed_time = time() - start_time
        print(f"\nSaved batch to {path} -> took {elapsed_time:.2f} seconds.\n")
