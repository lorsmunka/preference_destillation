"""E-01 — Does the DIRECTION of KL/CE annealing matter, and does it depend on domain entropy?

Five arms on kl_ratio_start -> kl_ratio_end, on every domain, at several seeds. No new
library code: the direction is already expressed by the start/end pair.

    python experiments/anneal_direction.py                 # all domains, cheap ones first
    python experiments/anneal_direction.py reddit math     # only these domains

Reddit and math run at 10 seeds (~3 min each); post-gen at 3 seeds (~3 h each). Completed
runs are skipped, so the script is re-runnable and doubles as the record of the experiment.
The entropy law (experiment_ledger E-01) predicts no separation between arms where
exp(H) ~ 1 (reddit, math) and separation, if any, only on post-gen (exp(H) ~ 2.7).
"""

import statistics
import sys

from _common import best_and_final, seeded_run_training, write_rows

from library import Analysis, TrainingRun
from library.domain import MATH_WORD_PROBLEM, POST_GENERATION, REDDIT_SENTIMENT

EXPERIMENT = "e01-anneal-direction"

# arm name -> (kl_ratio_start, kl_ratio_end)
ARMS = {
    "ce-to-kl": (0.1, 0.9),
    "kl-to-ce": (0.9, 0.1),
    "flat-50": (0.5, 0.5),
    "pure-ce": (0.0, 0.0),
    "pure-kl": (1.0, 1.0),
}

# Cheap domains first so results arrive early; post-gen follows the existing 3.8k/10-epoch
# recipe from postgen_kl_anneal.py so its numbers stay comparable with those runs.
DOMAINS = {
    "reddit": dict(domain=REDDIT_SENTIMENT, seeds=range(1, 11), epoch_count=3, max_training_examples=2048),
    "math": dict(domain=MATH_WORD_PROBLEM, seeds=range(1, 11), epoch_count=3, max_training_examples=2048),
    "postgen": dict(domain=POST_GENERATION, seeds=range(1, 4), epoch_count=10, max_training_examples=3840),
}

MATCHED_BODY = dict(
    hidden_dim=48,
    num_layers=3,
    num_heads=1,
    dropout=0.08,
    batch_size=32,
    learning_rate=0.0015,
    lr_warmup_ratio=0.1,
    training_test_ratio=0.8,
    auxiliary_token_percentage=1.0,
    distillation_temperature_start=1.0,
    distillation_temperature_end=1.0,
)

RESULT_COLUMNS = [
    "run", "domain", "arm", "seed", "kl_ratio_start", "kl_ratio_end",
    "task_accuracy_best", "task_accuracy_final",
    "teacher_student_overlap_final", "mean_target_rank_final",
    "student_entropy_final", "teacher_entropy_final",
]


def run_name(domain_key, arm, seed):
    return f"e01-{domain_key}-{arm}-s{seed}"


def build_runs(domain_keys):
    seeded = []
    for domain_key in domain_keys:
        spec = DOMAINS[domain_key]
        for arm, (kl_start, kl_end) in ARMS.items():
            for seed in spec["seeds"]:
                run = TrainingRun(
                    **MATCHED_BODY,
                    domain=spec["domain"],
                    epoch_count=spec["epoch_count"],
                    max_training_examples=spec["max_training_examples"],
                    kl_ratio_start=kl_start,
                    kl_ratio_end=kl_end,
                    run_name=run_name(domain_key, arm, seed),
                    experiment=EXPERIMENT,
                    description=f"E-01 anneal direction: {domain_key}, {arm} ({kl_start}->{kl_end}), seed {seed}",
                )
                seeded.append((seed, run))
    return seeded


def collect_rows(domain_keys):
    rows = []
    for domain_key in domain_keys:
        for arm, (kl_start, kl_end) in ARMS.items():
            for seed in DOMAINS[domain_key]["seeds"]:
                name = run_name(domain_key, arm, seed)
                try:
                    run = Analysis(name).run
                except Exception:
                    continue
                best, final = best_and_final(run)
                if best is None:
                    continue
                rows.append({
                    "run": name, "domain": domain_key, "arm": arm, "seed": seed,
                    "kl_ratio_start": kl_start, "kl_ratio_end": kl_end,
                    "task_accuracy_best": best.task_accuracy,
                    "task_accuracy_final": final.task_accuracy,
                    "teacher_student_overlap_final": final.teacher_student_overlap,
                    "mean_target_rank_final": final.mean_target_rank,
                    "student_entropy_final": final.student_entropy,
                    "teacher_entropy_final": final.teacher_entropy,
                })
    return rows


def print_summary(rows):
    """Mean +/- std per (domain, arm) — the table the ledger entry quotes."""
    print(f"\n{'domain':8} {'arm':10} {'n':>3} {'task best':>14} {'task final':>14} {'overlap':>10} {'rank':>8}")
    for domain_key in DOMAINS:
        for arm in ARMS:
            group = [r for r in rows if r["domain"] == domain_key and r["arm"] == arm]
            if not group:
                continue
            def mean_std(key):
                values = [r[key] for r in group if r[key] is not None]
                if not values:
                    return "     -"
                mean = statistics.mean(values)
                std = statistics.stdev(values) if len(values) > 1 else 0.0
                return f"{mean:6.3f}+/-{std:5.3f}"
            print(f"{domain_key:8} {arm:10} {len(group):3d} {mean_std('task_accuracy_best'):>14} "
                  f"{mean_std('task_accuracy_final'):>14} {mean_std('teacher_student_overlap_final'):>10} "
                  f"{mean_std('mean_target_rank_final'):>8}")


def main():
    domain_keys = [key for key in sys.argv[1:] if key in DOMAINS] or list(DOMAINS)
    runs = build_runs(domain_keys)
    print(f"{EXPERIMENT}: {len(runs)} runs over {domain_keys} "
          f"({len(ARMS)} arms x seeds per domain)\n")
    seeded_run_training(runs)

    rows = collect_rows(domain_keys)
    if rows:
        write_rows("e01_anneal_direction.csv", RESULT_COLUMNS, rows)
        print_summary(rows)
    else:
        print("No completed runs yet.")


if __name__ == "__main__":
    main()
