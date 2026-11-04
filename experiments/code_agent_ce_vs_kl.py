"""Pure KL vs pure CE on the coding agent's real data (2026-11-05, user: "Okay, start the coding ones"):
the math tutor's experiment (experiments/math_ce_vs_kl.py) on the second agent. Two students, identical but
for what they learn from:
  kl  Gemma's whole probability list at every position (KL share 1.0 throughout)
  ce  only the word Gemma wrote (KL share 0)
Size m (as on the math tutor), 10 passes, seed 1: both students get the same start weights and data order,
so they are read as a pair. Tested after every pass on every test record (max_test_examples=None: the
4 test projects the students never see). Task accuracy here = the student's scope covers exactly
Gemma's lines (strict: an extra or a missing line counts as wrong).

    python experiments/code_agent_ce_vs_kl.py        # kl, then ce (one after the other), then the report
    python experiments/code_agent_ce_vs_kl.py kl     # one student only
    python experiments/code_agent_ce_vs_kl.py ce

Not side by side on the 4070: alone a student trains at 2.3 s per batch in ~5 GB; two together filled the
card and slowed to 8 and 22 s per batch (timed 2026-11-05).
"""

import sys

from _common import RESULTS_DIR, seeded_run_training, size_settings, write_rows

from library import RunStore, TrainingRun
from library.domain.code_agent import CodeAgentDomain
from library.tooling.render.plots import plot_tests_by_pass

TEACHER = "google/gemma-4-12B-it"
DOMAIN = CodeAgentDomain("v1")
EXPERIMENT = "code-agent"
ARMS = {"kl": (1.0, 1.0), "ce": (0.0, 0.0)}   # arm -> (kl_ratio_start, kl_ratio_end)
LABELS = {"kl": "KL", "ce": "CE"}
SIZE = "m"
SEEDS = [1]
STUDENT = dict(dropout=0.1, epoch_count=10, batch_size=32, lr_warmup_ratio=0.1, max_training_examples=None,
               max_test_examples=None)
CHOSEN = sys.argv[1].split(",") if len(sys.argv) > 1 else list(ARMS)
RESULT_COLUMNS = ["row", "run", "arm", "seed", "pass", "task_accuracy", "majority_baseline", "test_error",
                  "test_token_accuracy"]


def run_name(arm, seed):
    return f"{EXPERIMENT}-{arm}-{SIZE}-s{seed}"


def build_runs(arms):
    return [(seed, TrainingRun(**STUDENT, **size_settings(SIZE), domain=DOMAIN, teacher_model=TEACHER,
                               shuffle_seed=seed, kl_ratio_start=ARMS[arm][0], kl_ratio_end=ARMS[arm][1],
                               run_name=run_name(arm, seed), experiment=EXPERIMENT,
                               description=f"pure {arm} (KL share {ARMS[arm][0]}), coding agent (v1), size {SIZE}, "
                                           f"seed {seed}, every test record"))
            for seed in SEEDS for arm in arms]


def finished_seeds():
    from library.training.training_runner import is_completed
    return [seed for seed in SEEDS if all(is_completed(run_name(arm, seed)) for arm in ARMS)]


def report():
    """Every pass of both students: task accuracy next to the most common answer, test error, token
    accuracy; then KL minus CE at the last pass, seed by seed, and the graph."""
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
    write_rows("code_agent.csv", RESULT_COLUMNS, rows)
    print("\nKL minus CE at the last pass (same seed = same start weights and data order):")
    for seed in seeds:
        kl_last, ce_last = runs[("kl", seed)].eval_epochs[-1], runs[("ce", seed)].eval_epochs[-1]
        print(f"  seed {seed}: task accuracy {100 * (kl_last.task_accuracy - ce_last.task_accuracy):+.1f} points, "
              f"test error {kl_last.ce_loss - ce_last.ce_loss:+.3f}")
    seed_text = f"seed {seeds[0]}" if len(seeds) == 1 else f"seeds {', '.join(map(str, seeds))} (thick: mean, thin: each seed)"
    size_text = f"{runs[('kl', seeds[0])].params_millions():.1f}M"
    graph = plot_tests_by_pass({LABELS[arm]: [runs[(arm, seed)] for seed in seeds] for arm in ARMS},
                               RESULTS_DIR / "code_agent.png",
                               title=f"Pure KL vs pure CE, coding agent, size {SIZE} ({size_text}), {seed_text}")
    print(f"Graph: {graph}")


def main():
    unknown = [arm for arm in CHOSEN if arm not in ARMS]
    if unknown:
        raise SystemExit(f"unknown students {unknown}: choose from {list(ARMS)}")
    if not seeded_run_training(build_runs(CHOSEN)):
        return
    if finished_seeds() == SEEDS:
        report()
    else:
        print("This student is done; the report follows when the other one is (rerun with no arguments).")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
