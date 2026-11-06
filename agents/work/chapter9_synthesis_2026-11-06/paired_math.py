"""Read-only: seed-matched KL vs CE (and KL->CE) comparisons for the math tutor full run, from runs_evals.csv."""
import csv
import math
import statistics
from pathlib import Path

rows = list(csv.DictReader(open(Path(__file__).parent / "runs_evals.csv", encoding="utf-8")))
by_run = {}
for row in rows:
    by_run.setdefault(row["run"], []).append(row)

metrics = ["task_accuracy", "ce_loss", "teacher_forced_accuracy", "student_accuracy", "teacher_student_overlap",
           "mean_target_rank", "student_entropy", "topk_accuracy", "validity_rate", "termination_rate"]


def value(run, epoch, metric):
    for row in by_run[run]:
        if int(row["epoch"]) == epoch:
            return float(row[metric])
    return None


def t_paired(differences):
    mean = statistics.mean(differences)
    if len(differences) < 2:
        return mean, None, None
    sd = statistics.stdev(differences)
    t = mean / (sd / math.sqrt(len(differences))) if sd > 0 else float("inf")
    return mean, sd, t


print("FINAL PASS (epoch index 9), KL vs CE, seeds 1-3")
for metric in metrics:
    kl = [value(f"math-agent-kl-m-s{seed}", 9, metric) for seed in (1, 2, 3)]
    ce = [value(f"math-agent-ce-m-s{seed}", 9, metric) for seed in (1, 2, 3)]
    differences = [a - b for a, b in zip(kl, ce)]
    mean, sd, t = t_paired(differences)
    print(f"{metric:26s} KL {statistics.mean(kl):.4f} (sd {statistics.stdev(kl):.4f})  CE {statistics.mean(ce):.4f}"
          f" (sd {statistics.stdev(ce):.4f})  diff {mean:+.4f} sd {sd:.4f} t {t:.1f}  per seed {[round(d, 4) for d in differences]}")

print()
print("PASSES WHERE KL BEATS CE (30 pairs), per metric (higher better except ce_loss, mean_target_rank)")
for metric in metrics:
    lower_better = metric in ("ce_loss", "mean_target_rank")
    wins = 0
    total = 0
    for seed in (1, 2, 3):
        for epoch in range(10):
            a = value(f"math-agent-kl-m-s{seed}", epoch, metric)
            b = value(f"math-agent-ce-m-s{seed}", epoch, metric)
            total += 1
            if (a < b) if lower_better else (a > b):
                wins += 1
    print(f"{metric:26s} KL better at {wins}/{total}")

print()
print("KL->CE vs KL and vs CE, seeds 1-2, per pass (task_accuracy diff, ce_loss diff)")
for seed in (1, 2):
    line = []
    for epoch in range(10):
        klce_t = value(f"math-agent-kl-to-ce-m-s{seed}", epoch, "task_accuracy")
        kl_t = value(f"math-agent-kl-m-s{seed}", epoch, "task_accuracy")
        ce_t = value(f"math-agent-ce-m-s{seed}", epoch, "task_accuracy")
        klce_l = value(f"math-agent-kl-to-ce-m-s{seed}", epoch, "ce_loss")
        kl_l = value(f"math-agent-kl-m-s{seed}", epoch, "ce_loss")
        ce_l = value(f"math-agent-ce-m-s{seed}", epoch, "ce_loss")
        line.append(f"p{epoch + 1}: vsKL {klce_t - kl_t:+.3f}/{klce_l - kl_l:+.3f} vsCE {klce_t - ce_t:+.3f}/{klce_l - ce_l:+.3f}")
    print(f"seed {seed}:")
    for item in line:
        print("   ", item)

print()
print("KL->CE final pass, other metrics, seeds 1-2 vs KL / CE same seeds")
for metric in metrics:
    klce = [value(f"math-agent-kl-to-ce-m-s{seed}", 9, metric) for seed in (1, 2)]
    kl = [value(f"math-agent-kl-m-s{seed}", 9, metric) for seed in (1, 2)]
    ce = [value(f"math-agent-ce-m-s{seed}", 9, metric) for seed in (1, 2)]
    print(f"{metric:26s} KL->CE {statistics.mean(klce):.4f}  KL {statistics.mean(kl):.4f}  CE {statistics.mean(ce):.4f}")

print()
print("Pass at which the per-pass gain in task accuracy falls below 1 point (mean of 3 seeds)")
for arm in ("kl", "ce"):
    means = [statistics.mean(value(f"math-agent-{arm}-m-s{seed}", epoch, "task_accuracy") for seed in (1, 2, 3)) for epoch in range(10)]
    gains = [round(means[i] - means[i - 1], 4) for i in range(1, 10)]
    print(arm, [round(m, 4) for m in means], "gains", gains)
