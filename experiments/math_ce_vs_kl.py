"""Pure KL vs pure CE on the full math-tutor data (2026-10-27, user: "10-10 passes on pure ce vs pure kl and
graph at round 20m size"): two students, identical but for what they learn from.
  kl  Gemma's whole probability list at every position (KL share 1.0 throughout)
  ce  only the word Gemma wrote (KL share 0)
Size m (16M), 10 passes; each seed gives both students the same start weights and data order, so the two
are read as a pair (KL minus CE per seed). Tested after every pass on every test reply
(max_test_examples=None: 8,415 replies, not the default tail of 1,008), so the two lines of the graph are
far apart only if the difference is real. Seed 1 done 2026-10-29 (KL .700 vs CE .684), seed 2 2026-10-31
(KL .719 vs CE .688), seed 3 2026-11-02 (KL .697 vs CE .677): KL ahead at all 30 paired tests.
A third student added 2026-11-02 (user: "Should we do a kl to ce?" ... "yes"):
  kl-to-ce  the KL share slides from 1 to 0 over the 10 passes (the framework's default schedule, 0.9 -> 0.1,
            runs the same way), so it tests whether ending on the written word helps after learning the list

    python experiments/math_ce_vs_kl.py              # trains what is missing (one after the other), then reports
    python experiments/math_ce_vs_kl.py kl           # one student per process: two train side by side on one card
    python experiments/math_ce_vs_kl.py kl-to-ce 1   # ... and only seed 1 of it (seeds: 1, 2, 3 or 1,2)
"""

import sys

from _common import RESULTS_DIR, seeded_run_training, size_settings, write_rows

from library import RunStore, TrainingRun
from library.domain.math_tutor.full import FullDomain
from library.tooling.render.plots import plot_tests_by_pass

TEACHER = "google/gemma-4-12B-it"
DOMAIN = FullDomain("v1")
EXPERIMENT = "math-agent"
ARMS = {"kl": (1.0, 1.0), "ce": (0.0, 0.0), "kl-to-ce": (1.0, 0.0)}   # arm -> (kl_ratio_start, kl_ratio_end)
LABELS = {"kl": "KL", "ce": "CE", "kl-to-ce": "KL to CE"}
COMPARISONS = [("kl", "ce"), ("kl-to-ce", "kl"), ("kl-to-ce", "ce")]
SIZE = "m"
SEEDS = [1, 2, 3]
STUDENT = dict(dropout=0.1, epoch_count=10, batch_size=32, lr_warmup_ratio=0.1, max_training_examples=None,
               max_test_examples=None)
CHOSEN = sys.argv[1].split(",") if len(sys.argv) > 1 else list(ARMS)
CHOSEN_SEEDS = [int(seed) for seed in sys.argv[2].split(",")] if len(sys.argv) > 2 else SEEDS
RESULT_COLUMNS = ["row", "run", "arm", "seed", "pass", "task_accuracy", "majority_baseline", "test_error",
                  "test_token_accuracy"]


def run_name(arm, seed):
    return f"{EXPERIMENT}-{arm}-{SIZE}-s{seed}"


def describe(arm):
    start, end = ARMS[arm]
    return f"pure {arm} (KL share {start})" if start == end else f"{arm} (KL share {start} -> {end})"


def build_runs(arms, seeds):
    return [(seed, TrainingRun(**STUDENT, **size_settings(SIZE), domain=DOMAIN, teacher_model=TEACHER,
                               shuffle_seed=seed, kl_ratio_start=ARMS[arm][0], kl_ratio_end=ARMS[arm][1],
                               run_name=run_name(arm, seed), experiment=EXPERIMENT,
                               description=f"{describe(arm)}, full math tutor (v1), size {SIZE}, "
                                           f"seed {seed}, every test reply"))
            for seed in seeds for arm in arms]


def finished_seeds():
    from library.training.training_runner import is_completed
    return [seed for seed in SEEDS if all(is_completed(run_name(arm, seed)) for arm in ARMS)]


def report():
    """Every pass of every student for every finished seed: task accuracy next to the most common answer,
    test error, token accuracy; then the paired differences at the last pass, seed by seed, and the graph."""
    store = RunStore()
    seeds = finished_seeds()
    runs = {(arm, seed): store.get(run_name(arm, seed)) for arm in ARMS for seed in seeds}
    rows = []
    for (arm, seed), run in runs.items():
        for number, evaluation in enumerate(run.eval_epochs, start=1):
            rows.append({"row": f"{run.name}-pass{number}", "run": run.name, "arm": arm, "seed": seed,
                         "pass": number, "task_accuracy": evaluation.task_accuracy,
                         "majority_baseline": run.majority_baseline(evaluation),
                         "test_error": round(evaluation.ce_loss, 4),
                         "test_token_accuracy": evaluation.teacher_forced_accuracy})
    for row in rows:
        print(", ".join(f"{column}={row[column]}" for column in RESULT_COLUMNS[1:]))
    write_rows("math_agent.csv", RESULT_COLUMNS, rows)
    for first, second in COMPARISONS:
        print(f"\n{LABELS[first]} minus {LABELS[second]} at the last pass (same seed = same start weights and data order):")
        for seed in seeds:
            first_last, second_last = runs[(first, seed)].eval_epochs[-1], runs[(second, seed)].eval_epochs[-1]
            print(f"  seed {seed}: task accuracy {100 * (first_last.task_accuracy - second_last.task_accuracy):+.1f} "
                  f"points, test error {first_last.ce_loss - second_last.ce_loss:+.3f}")
    seed_text = f"seed {seeds[0]}" if len(seeds) == 1 else f"seeds {', '.join(map(str, seeds))} (thick: mean, thin: each seed)"
    size_text = f"{runs[('kl', seeds[0])].params_millions():.1f}M"
    graph = plot_tests_by_pass({LABELS[arm]: [runs[(arm, seed)] for seed in seeds] for arm in ARMS},
                               RESULTS_DIR / "math_agent.png",
                               title=f"KL vs CE vs KL to CE, math tutor, size {SIZE} ({size_text}), {seed_text}")
    print(f"Graph: {graph}")


def main():
    unknown = [arm for arm in CHOSEN if arm not in ARMS]
    if unknown:
        raise SystemExit(f"unknown students {unknown}: choose from {list(ARMS)}")
    unknown_seeds = [seed for seed in CHOSEN_SEEDS if seed not in SEEDS]
    if unknown_seeds:
        raise SystemExit(f"unknown seeds {unknown_seeds}: choose from {SEEDS}")
    if not seeded_run_training(build_runs(CHOSEN, CHOSEN_SEEDS)):
        return
    if finished_seeds() == SEEDS:
        report()
    else:
        print("This student is done; the report follows when the other one is (rerun with no arguments).")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
