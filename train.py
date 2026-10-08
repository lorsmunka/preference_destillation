"""Training entry point — edit the RUNS list and run `python train.py`.

Run configs are typed `TrainingRun` objects (not dicts): pass a Domain object as `domain=`
(no magic strings), everything else has a default. Runs whose info.json says "completed"
are skipped, so the list is resumable.

For a named experiment (its own runs plus the analysis that reads them), write a script
under experiments/ instead of editing this list — see experiments/postgen_kl_anneal.py.
"""

from library import TrainingRun
from library.domain import REDDIT_SENTIMENT
from library.training.training_runner import run_training

RUNS = [
    TrainingRun(
        domain=REDDIT_SENTIMENT,
        run_name="example-sentiment-kl99to50",
        experiment="example",
        hidden_dim=48,
        num_layers=3,
        num_heads=1,
        dropout=0.08,
        epoch_count=3,
        batch_size=32,
        learning_rate=0.0015,
        lr_warmup_ratio=0.1,
        max_training_examples=8000,
        training_test_ratio=0.8,
        auxiliary_token_percentage=1.0,
        kl_ratio_start=0.99,
        kl_ratio_end=0.5,
        distillation_temperature_start=1.0,
        distillation_temperature_end=1.0,
        description="Example run — KL 0.99->0.50 annealing on sentiment",
    ),
]


def main():
    if run_training(RUNS):
        print("Done.")
    else:
        print("Stopped. Run it again to resume.")


if __name__ == "__main__":
    main()
