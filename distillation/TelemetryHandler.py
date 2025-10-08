import json
import os
from time import time, sleep

TELEMETRY_DIR = "./telemetry"
TELEMETRY_FILE_PATH = f"./{TELEMETRY_DIR}/telemetry.json"


class TelemetryHandler:
    def __init__(self):
        self.processed_sentence_count = 0
        self.successful_sentence_count = 0
        self.total_runtime_seconds = 0
        self.session_count = 0
        self.session_start_time = None
        self.load_save()

    @property
    def current_batch_sentence_count(self):
        return self.successful_sentence_count % 32

    @property
    def batch_count(self):
        return self.successful_sentence_count // 32

    def load_save(self):
        start_time = time()
        print("Loading telemetry...")
        if os.path.exists(TELEMETRY_FILE_PATH):
            with open(TELEMETRY_FILE_PATH, "r", encoding="utf-8") as file:
                data = json.load(file)
                self.processed_sentence_count = data.get(
                    "processed_sentence_count", 0)
                self.successful_sentence_count = data.get(
                    "successful_sentence_count", 0)
                self.total_runtime_seconds = data.get(
                    "total_runtime_seconds", 0)
                self.session_count = data.get("session_count", 0)

        self.session_count += 1
        self.session_start_time = time()

        elapsed_time = time() - start_time
        print(f"Loaded telemetry -> took {elapsed_time:.2f} seconds.\n")
        print("== Telemetry ==")
        print(f"Processed sentences: {self.processed_sentence_count:,}")
        print(f"Successful sentences: {self.successful_sentence_count:,}")
        print(f"Total runtime (seconds): {self.total_runtime_seconds:2,.2f}")
        print(f"Session count: {self.session_count:,}")
        print(f"Current batch count: {self.current_batch_sentence_count}")
        print(f"Waiting 2 seconds before continuing... \n")
        sleep(2)

    def save(self):
        start_time = time()
        print("Saving telemetry...")

        if not os.path.exists(TELEMETRY_DIR):
            os.makedirs(TELEMETRY_DIR)

        self.total_runtime_seconds += time() - self.session_start_time
        with open(TELEMETRY_FILE_PATH, "w", encoding="utf-8") as file:
            json.dump({
                "processed_sentence_count": self.processed_sentence_count,
                "successful_sentence_count": self.successful_sentence_count,
                "total_runtime_seconds": self.total_runtime_seconds,
                "session_count": self.session_count
            }, file, indent=4)

        elapsed_time = time() - start_time
        print(
            f"Saved telemetry to {TELEMETRY_FILE_PATH} -> took {elapsed_time:.2f} seconds.\n")
