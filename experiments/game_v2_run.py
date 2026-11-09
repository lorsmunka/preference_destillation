"""The game-language agent v2 on rented cards (2026-11-09; the corpus from library/domain/game_dsl/v2_corpus.py):

  score [max] [group]   Gemma scores the corpus's targets teacher-forced (the card2 prompt) -> batches/game_dsl_v2/
                        max = how many records (default all), group = records per forward pass (default 8, A40 48 GB)
  student <size>        one student on the scored batches (sizes below), KL 0.9 -> 0.1, PASSES passes

    python experiments/game_v2_run.py score 600 8        (the speed test)
    python experiments/game_v2_run.py student s10
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TEACHER = "google/gemma-4-12B-it"
PASSES = 6
SEED = 1
# the user's sizes (2026-11-09): ~5M, 10M, 15M, 20M, 30M parameters (hidden width, layers, heads, learning rate)
SIZES = {"s5": (192, 6, 4, 0.0012), "s10": (256, 6, 4, 0.001), "s15": (320, 7, 5, 0.0009),
         "s20": (384, 8, 6, 0.0008), "s30": (448, 10, 7, 0.0007)}


def score(max_examples=None, group=8):
    from library import GenerationJob
    from library.data_gen.generator import DistillationDataGenerator
    from library.domain.game_dsl import GameDslV2Domain
    from library.shared import ExitListener
    job = GenerationJob(domain=GameDslV2Domain(), teacher_model=TEACHER, load_in_4bit=False, chat_template=True,
                        max_examples=max_examples or 1_000_000, batch_size=32, generation_batch_size=group,
                        description="game_dsl v2: Gemma's checked blocks for the user-point-of-view asks, "
                                    "teacher-forced, card2 prompt, bf16")
    exit_listener = ExitListener()
    DistillationDataGenerator(job, exit_listener).run()
    exit_listener.stop()


def student(size):
    from _common import seeded_run_training
    from library import TrainingRun
    from library.domain.game_dsl import GameDslV2Domain
    from library.shared.build_input_vocabulary import build_input_vocabulary
    from library.shared.config import load_input_vocabulary
    domain = GameDslV2Domain()
    if load_input_vocabulary(domain.name, TEACHER) is None:
        build_input_vocabulary(domain, TEACHER)
    hidden_dim, num_layers, num_heads, learning_rate = SIZES[size]
    run = TrainingRun(
        run_name=f"game-agent-v2-{size}-s{SEED}", domain=domain, teacher_model=TEACHER, experiment="game-agent-v2",
        hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads, learning_rate=learning_rate,
        dropout=0.1, epoch_count=PASSES, batch_size=32, lr_warmup_ratio=0.1, max_training_examples=None,
        shuffle_seed=SEED, kl_ratio_start=0.9, kl_ratio_end=0.1,
        description=f"game-language agent v2: Gemma's checked blocks (user-point-of-view asks), KL 0.9 -> 0.1, "
                    f"size {size}, seed {SEED}")
    sys.exit(0 if seeded_run_training([(SEED, run)]) else 1)


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "score"
    if action == "score":
        score(int(sys.argv[2]) if len(sys.argv) > 2 else None, int(sys.argv[3]) if len(sys.argv) > 3 else 8)
    elif action == "student":
        student(sys.argv[2])
