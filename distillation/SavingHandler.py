import os
import json
from time import time
from TelemetryHandler import TelemetryHandler

OUTPUT_DIR = "./distillation_batches"


class SavingHandler:
    def __init__(self, telemetry_handler: TelemetryHandler):
        self.telemetry_handler = telemetry_handler

    def save_batch(self, batch_examples):
        start_time = time()

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        path = f"{OUTPUT_DIR}/batch_{self.telemetry_handler.batch_count}.jsonl"
        with open(path, "w", encoding="utf-8") as file:
            for example in batch_examples:
                file.write(f"{json.dumps(example)}\n")

        elapsed_time = time() - start_time
        print(f"Saved batch to {path} -> took {elapsed_time:.2f} seconds.")
