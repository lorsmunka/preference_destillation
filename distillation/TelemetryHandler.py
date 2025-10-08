import json
import os
from time import time


class TelemetryHandler:
    def __init__(self):
        self.processed_sentence_count = 0
        self.total_runtime_seconds = 0
        self.session_count = 0
        self.session_start_time = None
        self.load_save()

    @property
    def current_batch_count(self):
        return self.processed_sentence_count % 32

    @property
    def batch_count(self):
        return self.processed_sentence_count // 32

    def load_save(self):
        start_time = time()
        print("Loading telemetry...")
        if os.path.exists("./telemetry/telemetry.json"):
            with open("./telemetry/telemetry.json", "r", encoding="utf-8") as file:
                data = json.load(file)
                self.processed_sentence_count = data.get(
                    "processed_sentence_count", 0)
                self.total_runtime_seconds = data.get(
                    "total_runtime_seconds", 0)
                self.session_count = data.get("session_count", 0)

        elapsed_time = time() - start_time
        print(f"Loaded telemetry -> took {elapsed_time:.2f} seconds.")
        print(f"Processed sentences: {self.processed_sentence_count:,}")
        print(f"Total runtime (seconds): {self.total_runtime_seconds:,}")
        print(f"Session count: {self.session_count:,}")
        print(f"Current batch count: {self.current_batch_count}")
