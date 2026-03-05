import json
import os

from queue_runner import QueueRunner
from shared import ExitListener, get_output_dir, DATA_GENERATION_QUEUE_PATH

with open(DATA_GENERATION_QUEUE_PATH, "r", encoding="utf-8") as file:
    queue = json.load(file)

exit_listener = ExitListener()

for queue_element in queue:
    domain = queue_element["domain"]
    model_name = queue_element["model_name"]

    info_path = os.path.join(get_output_dir(domain, model_name), "info.json")

    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as file:
            info = json.load(file)
        if info.get("status") == "completed":
            print(f"Skipping completed: {domain} ({model_name})")
            continue

    print(f"\n{'=' * 60}")
    print(f"Starting: {domain} ({model_name})")
    print(f"{'=' * 60}\n")

    runner = QueueRunner(queue_element, exit_listener)
    runner.run()

    print(f"\nCompleted: {domain} ({model_name})\n")

exit_listener.stop()
print("Bye!")
