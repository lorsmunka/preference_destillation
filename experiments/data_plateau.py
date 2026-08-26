"""E5 — Where does distillation data stop paying? (board D3/D4)

D3 (2404.03263): distillation's benefit "tends not to occur if the task is data-limited".
D4 (2504.14772): small models saturate on excess synthetic data, plateau ~5k-30k, but
size- and task-dependent and never charted for reduced-vocab generative students.

Claim under test: task accuracy against teacher-example count flattens (or turns down)
well before the 500k examples the corpus holds — so every other experiment on this board
can be run at a small fraction of the data without changing its conclusion.

Body, vocabulary and schedule are held fixed; only `max_training_examples` moves.

    python experiments/data_plateau.py

5 data sizes x 5 seeds = 25 runs. Epoch count is held at 3, so larger sizes see strictly
more gradient steps — the axis is total teacher data, which is what costs money to make.
"""

from _common import seeded_run_training

from library import TrainingRun
from library.domain import REDDIT_SENTIMENT

EXPERIMENT = "ideaD-data-plateau"

DATA_SIZES = [256, 512, 1024, 2048, 4096]
SEEDS = [1, 2, 3, 4, 5]

FIXED = dict(
    domain=REDDIT_SENTIMENT,
    hidden_dim=48,
    num_layers=3,
    num_heads=1,
    dropout=0.08,
    epoch_count=3,
    batch_size=32,
    learning_rate=0.0015,
    lr_warmup_ratio=0.1,
    training_test_ratio=0.8,
    auxiliary_token_percentage=1.0,
    kl_ratio_start=0.99,
    kl_ratio_end=0.5,
    distillation_temperature_start=1.0,
    distillation_temperature_end=1.0,
)


def build_runs():
    seeded = []
    for size in DATA_SIZES:
        for seed in SEEDS:
            run = TrainingRun(
                **FIXED,
                max_training_examples=size,
                run_name=f"ideaD-data{size:05d}-s{seed}",
                experiment=EXPERIMENT,
                description=f"Data plateau: {size} teacher examples, seed {seed}",
            )
            seeded.append((seed, run))
    return seeded


if __name__ == "__main__":
    runs = build_runs()
    print(f"{EXPERIMENT}: {len(runs)} runs "
          f"({len(DATA_SIZES)} data sizes x {len(SEEDS)} seeds)\n")
    seeded_run_training(runs)
    print("\nDone. Analyse with: python experiments/report_ideas.py")
