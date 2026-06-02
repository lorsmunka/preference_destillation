"""Training entry point — edit the RUNS list and run `python train.py`.

Replaces the old `training/main.py` + `training_queue.json`: run configs are plain Python now.
Each dict is one run; runs whose info.json says "completed" are skipped, so the list is
resumable. New eval-time knobs (optional per run): `eval_top_k` (default 20),
`eval_cap_multiple` (default 2.0 — natural-termination cap = 2x teacher length).
"""

import json
import os

from shared import ExitListener, get_training_run_dir
from training.training_runner import TrainingRunner

RUNS = [
    {
        "domain": "reddit_comment_sentiment",
        "teacher_model": "google/gemma-3-4b-it",
        "hidden_dim": 48,
        "num_layers": 3,
        "num_heads": 1,
        "dropout": 0.08,
        "epoch_count": 3,
        "batch_size": 32,
        "learning_rate": 0.0015,
        "lr_warmup_ratio": 0.1,
        "max_training_examples": 8000,
        "training_test_ratio": 0.8,
        "auxiliary_token_percentage": 1.0,
        "kl_ratio_start": 0.99,
        "kl_ratio_end": 0.5,
        "distillation_temperature_start": 1.0,
        "distillation_temperature_end": 1.0,
        "run_name": "example-sentiment-kl99to50",
        "experiment": "example",
        "description": "Example run — KL 0.99->0.50 annealing on sentiment",
        # "eval_top_k": 20,
        # "eval_cap_multiple": 2.0,
    },
]


def main():
    exit_listener = ExitListener()
    for config in RUNS:
        info_path = os.path.join(get_training_run_dir(config["run_name"]), "info.json")
        if os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as file:
                if json.load(file).get("status") == "completed":
                    print(f"Skipping completed run: {config['run_name']}")
                    continue
        if not TrainingRunner(config, exit_listener).run():
            break
    exit_listener.stop()
    print("Done.")


if __name__ == "__main__":
    main()
