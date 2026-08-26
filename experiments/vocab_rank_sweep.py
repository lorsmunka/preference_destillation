"""E1 — Does shrinking the OUTPUT vocabulary relieve the softmax/output-rank bottleneck?

Board ideas A1-A4 (ANCHOR), OQ1 (fidelity gap), S2 (vocab as a capacity axis).

Claim under test: at fixed student capacity, lowering the target distribution's rank by
shrinking the output vocabulary makes the distillation target easier to express, so
fidelity and task accuracy should IMPROVE (or at least not degrade) as V falls — the
mitigation the bottleneck papers leave to future work.

The sweep rides on `auxiliary_token_percentage`: the task-essential tokens (example +
whitespace + prompt = 64 on reddit) are always retained, and only the auxiliary bank
varies. So every run can still express the task; what changes is how much extra output
rank the head has to carry.

    python experiments/vocab_rank_sweep.py

5 auxiliary levels x 5 seeds = 25 runs, ~3 min each. Completed runs are skipped.
"""

from _common import seeded_run_training

from library import TrainingRun
from library.domain import REDDIT_SENTIMENT

EXPERIMENT = "ideaA-vocab-rank-sweep"

# (auxiliary_token_percentage, expected V on reddit) — V is confirmed from model_info
# after training rather than trusted from this table.
AUXILIARY_LEVELS = [
    (0.0, 64),
    (0.1, 113),
    (0.25, 193),
    (0.5, 300),
    (1.0, 525),
]

SEEDS = [1, 2, 3, 4, 5]

MATCHED_BODY = dict(
    domain=REDDIT_SENTIMENT,
    hidden_dim=48,
    num_layers=3,
    num_heads=1,
    dropout=0.08,
    epoch_count=3,
    batch_size=32,
    learning_rate=0.0015,
    lr_warmup_ratio=0.1,
    max_training_examples=2048,
    training_test_ratio=0.8,
    # Pure-ish KL so the measurement is about matching the teacher DISTRIBUTION, which is
    # what the rank argument is about. CE would score token identity and hide the effect.
    kl_ratio_start=0.99,
    kl_ratio_end=0.5,
    distillation_temperature_start=1.0,
    distillation_temperature_end=1.0,
)


def build_runs():
    seeded = []
    for auxiliary_percentage, expected_vocab in AUXILIARY_LEVELS:
        for seed in SEEDS:
            run = TrainingRun(
                **MATCHED_BODY,
                auxiliary_token_percentage=auxiliary_percentage,
                run_name=f"ideaA-vocab-aux{int(auxiliary_percentage * 100):03d}-s{seed}",
                experiment=EXPERIMENT,
                description=f"Output-vocab sweep: aux={auxiliary_percentage} (V~{expected_vocab}), seed {seed}",
            )
            seeded.append((seed, run))
    return seeded


if __name__ == "__main__":
    runs = build_runs()
    print(f"{EXPERIMENT}: {len(runs)} runs "
          f"({len(AUXILIARY_LEVELS)} vocab levels x {len(SEEDS)} seeds)\n")
    seeded_run_training(runs)
    print("\nDone. Analyse with: python experiments/report_ideas.py")
