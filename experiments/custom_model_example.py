"""How a custom student model is queued (2026-10-31, user: "I want a clean test file"): the built-in
Transformer and a custom LSTM side by side on the math tutor data. The LSTM lives in its own file,
library/model/custom/lstm_student.py; this file only queues it.

    python experiments/custom_model_example.py     # builds both students and prints them; trains nothing

To train them, pass RUNS to seeded_run_training like any other experiment.
"""

import sys

from _common import size_settings

from library import TrainingRun
from library.domain.math_tutor.full import FullDomain
from library.model.custom.lstm_student import LstmStudent
from library.model.custom_student import build_student
from library.shared import load_input_vocabulary

DOMAIN = FullDomain("v1")
TEACHER = "google/gemma-4-12B-it"
SHARED = dict(domain=DOMAIN, teacher_model=TEACHER, experiment="custom-model-example", epoch_count=3,
              batch_size=32, lr_warmup_ratio=0.1, kl_ratio_start=1.0, kl_ratio_end=1.0, shuffle_seed=1)

RUNS = [
    # the built-in Transformer, as every run so far
    TrainingRun(**SHARED, **size_settings("s"), run_name="custom-example-transformer-s"),

    # a custom model: model= is the class, model_settings= go to its __init__
    TrainingRun(**SHARED, run_name="custom-example-lstm", learning_rate=0.001,
                model=LstmStudent, model_settings={"hidden": 384, "layers": 2, "dropout": 0.1}),
]

RECORDED_KEYS = ["model", "model_settings", "hidden_dim", "num_layers", "num_heads", "dropout"]


def main():
    input_vocabulary = load_input_vocabulary(DOMAIN.name, TEACHER)
    for run in RUNS:
        settings = run.as_dict()
        print(f"\n=== {run.run_name} ===")
        student = build_student(settings, DOMAIN, input_vocabulary)
        print(f"Parameters: {student.get_num_parameters() / 1e6:.2f}M")
        print("info.json would record:", {key: settings[key] for key in RECORDED_KEYS if key in settings})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
