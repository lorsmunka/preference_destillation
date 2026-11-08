"""The game-language agent's first student (2026-11-08, user: "train a KL to CE flutter agent on the scoring, it just
needs to semi work"): one student on Gemma's own checked blocks (experiments/game_gemma_run.py: 39,274 train +
1,323 test records, teacher-forced with the reference card in front), KL share 0.9 -> 0.1 as the coding agent's
"kd", size m, 10 passes. Runs on the rented card right after the scoring (experiments/cloud_game_student.sh);
the report (compile / clean rate of its own blocks in the app) comes later at home.

    python experiments/game_student.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import seeded_run_training, size_settings  # noqa: E402
from library import TrainingRun  # noqa: E402
from library.domain.game_dsl import GameDslDomain  # noqa: E402

TEACHER = "google/gemma-4-12B-it"
DOMAIN = GameDslDomain("card")
SEED = 1
SIZE = "m"
RUN = TrainingRun(
    run_name=f"game-agent-kd-{SIZE}-s{SEED}", domain=DOMAIN, teacher_model=TEACHER, experiment="game-agent-v1",
    **size_settings(SIZE), dropout=0.1, epoch_count=10, batch_size=32, lr_warmup_ratio=0.1,
    max_training_examples=None, shuffle_seed=SEED, kl_ratio_start=0.9, kl_ratio_end=0.1,
    description=f"game-language agent v1: Gemma's checked blocks, KL 0.9 -> 0.1, size {SIZE}, seed {SEED}")


def main():
    from library.shared.build_input_vocabulary import build_input_vocabulary
    from library.shared.config import load_input_vocabulary
    if load_input_vocabulary(DOMAIN.name, TEACHER) is None:
        build_input_vocabulary(DOMAIN, TEACHER)
    sys.exit(0 if seeded_run_training([(SEED, RUN)]) else 1)


if __name__ == "__main__":
    main()
