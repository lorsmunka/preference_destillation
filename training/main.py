import json
import os

from training_runner import TrainingRunner
from shared import ExitListener, TRAINING_QUEUE_PATH, get_training_run_dir


with open(TRAINING_QUEUE_PATH, "r", encoding="utf-8") as file:
    queue = json.load(file)

exit_listener = ExitListener()

for queue_element in queue:
    run_name = queue_element["run_name"]
    run_dir = get_training_run_dir(run_name)
    info_path = os.path.join(run_dir, "info.json")

    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as file:
            info = json.load(file)
        if info.get("status") == "completed":
            print(f"Skipping completed run: {run_name}")
            continue

    runner = TrainingRunner(queue_element, exit_listener)
    should_continue = runner.run()

    if not should_continue:
        break

exit_listener.stop()
print("Bye!")
