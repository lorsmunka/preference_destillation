"""E-01 — Does the DIRECTION of KL/CE annealing matter, and does it depend on domain entropy?

Five arms on kl_ratio_start -> kl_ratio_end, on every domain, at several seeds. No new
library code: the direction is already expressed by the start/end pair.

    python experiments/anneal_direction.py                 # all domains, cheap ones first
    python experiments/anneal_direction.py reddit math     # only these domains

Reddit and math run at 10 seeds (~3 min each); post-gen at 3 seeds (~3 h each). Completed
runs are skipped, so the script is re-runnable and doubles as the record of the experiment.
The entropy law (experiment_ledger E-01) predicts no separation between arms where
exp(H) ~ 1 (reddit, math) and separation, if any, only on post-gen (exp(H) ~ 2.7).

Task accuracy is reported beside its majority baseline (always giving the most common
answer on the run's test set) and the excess over it: on reddit and math the students sit
at that floor (code review X-C1). Arms are compared pairwise with Welch's t-test and a
Bonferroni correction over the 10 pairs (X-M1). Spreads are sample std across seeds.
"""

import itertools
import sys

from _common import best_and_final, mean_std, seeded_run_training, write_rows

from library import RunStore, TrainingRun
from library.domain import MATH_WORD_PROBLEM, POST_GENERATION, REDDIT_SENTIMENT
from library.shared.metrics.task import answer_type_accuracy
from library.tooling.run import test_split_teacher_entropy
from library.tooling.significance import bonferroni, welch_t_test

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
    "majority_baseline", "task_accuracy_excess_best", "task_accuracy_excess_final",
    "boolean_accuracy_final", "numeric_accuracy_final",
    "teacher_student_overlap_final", "mean_target_rank_final",
    "student_entropy_final", "teacher_entropy_final",
]

# (domain, metric) pairs the arms are compared on, pairwise (X-M1).
PAIRWISE_TESTS = [
    ("reddit", "task_accuracy_excess_best"), ("reddit", "task_accuracy_excess_final"),
    ("math", "task_accuracy_excess_best"), ("math", "task_accuracy_excess_final"),
    ("postgen", "teacher_student_overlap_final"),
]
PAIR_COLUMNS = ["comparison", "domain", "metric", "arm_a", "arm_b", "n_a", "n_b", "mean_a", "mean_b",
                "difference", "welch_t", "degrees_of_freedom", "p_value", "p_bonferroni", "significant_after_correction"]
SIGNIFICANCE_LEVEL = 0.05


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


def completed_runs(domain_key, store=None):
    """The completed runs of one domain, as (arm, seed, Run)."""
    store = store or RunStore()
    found = []
    for arm in ARMS:
        for seed in DOMAINS[domain_key]["seeds"]:
            try:
                run = store.get(run_name(domain_key, arm, seed))
            except FileNotFoundError:
                continue
            if run.status == "completed":
                found.append((arm, seed, run))
    return found


def collect_rows(domain_keys):
    rows = []
    store = RunStore()
    for domain_key in domain_keys:
        for arm, seed, run in completed_runs(domain_key, store):
            kl_start, kl_end = ARMS[arm]
            best, final = best_and_final(run)
            if best is None:
                continue
            baseline = run.majority_baseline(final)
            answer_types = run.answer_type_accuracy(final)
            rows.append({
                "run": run.name, "domain": domain_key, "arm": arm, "seed": seed,
                "kl_ratio_start": kl_start, "kl_ratio_end": kl_end,
                "task_accuracy_best": best.task_accuracy,
                "task_accuracy_final": final.task_accuracy,
                "majority_baseline": baseline,
                "task_accuracy_excess_best": best.task_accuracy - baseline if baseline is not None else None,
                "task_accuracy_excess_final": final.task_accuracy - baseline if baseline is not None else None,
                "boolean_accuracy_final": answer_types.get("boolean_accuracy"),
                "numeric_accuracy_final": answer_types.get("numeric_accuracy"),
                "teacher_student_overlap_final": final.teacher_student_overlap,
                "mean_target_rank_final": final.mean_target_rank,
                "student_entropy_final": final.student_entropy,
                "teacher_entropy_final": final.teacher_entropy,
            })
    return rows


def format_mean_std(values, signed=False):
    mean, std, _ = mean_std(values)
    if mean is None:
        return "     -"
    number = f"{mean:+6.3f}" if signed else f"{mean:6.3f}"
    return f"{number}+/-{std:5.3f}" if std is not None else number


def print_summary(rows):
    """Mean +/- sample std per (domain, arm) — the table the ledger entry quotes."""
    print(f"\n{'domain':8} {'arm':10} {'n':>3} {'task best':>14} {'task final':>14} {'baseline':>9} "
          f"{'excess best':>15} {'excess final':>15} {'overlap':>14} {'rank':>14}")
    for domain_key in DOMAINS:
        for arm in ARMS:
            group = [r for r in rows if r["domain"] == domain_key and r["arm"] == arm]
            if not group:
                continue
            baseline = mean_std([r["majority_baseline"] for r in group])[0]
            print(f"{domain_key:8} {arm:10} {len(group):3d} "
                  f"{format_mean_std([r['task_accuracy_best'] for r in group]):>14} "
                  f"{format_mean_std([r['task_accuracy_final'] for r in group]):>14} "
                  f"{(f'{baseline:.3f}' if baseline is not None else '-'):>9} "
                  f"{format_mean_std([r['task_accuracy_excess_best'] for r in group], signed=True):>15} "
                  f"{format_mean_std([r['task_accuracy_excess_final'] for r in group], signed=True):>15} "
                  f"{format_mean_std([r['teacher_student_overlap_final'] for r in group]):>14} "
                  f"{format_mean_std([r['mean_target_rank_final'] for r in group]):>14}")
    print("baseline = accuracy of always giving the most common answer on the runs' test set; "
          "excess = task accuracy - baseline; +/- = sample std across seeds.")
    print_test_sets(rows)


def print_test_sets(rows):
    """Which sample the teacher entropy and the baseline come from (code review X-M8, X-C1)."""
    store = RunStore()
    for domain_key in DOMAINS:
        runs = [run for _, _, run in completed_runs(domain_key, store)]
        if not runs:
            continue
        entropy = test_split_teacher_entropy(runs)
        if entropy:
            print(f"{domain_key}: teacher entropy {entropy['nats']:.4f} nats, exp(H) {entropy['exp_nats']:.3f} "
                  f"({entropy['sample']})")
        composition = answer_type_accuracy(runs[0].test_set_answer_counts()) if domain_key == "math" else {}
        if composition:
            print(f"{domain_key}: test set {composition['boolean_examples'] + composition['numeric_examples']} "
                  f"questions = {composition['boolean_examples']} true/false (the commoner value alone scores "
                  f"{composition['boolean_majority_baseline']:.3f} of them) + {composition['numeric_examples']} numeric. "
                  f"Per-type student accuracy is in eval logs written since the fix; older runs: "
                  f"Analysis(run).evaluate().extra")


def pairwise_rows(rows):
    """Every pair of arms per (domain, metric) in PAIRWISE_TESTS: Welch's t-test, with the
    p-value Bonferroni-corrected over the pairs of that domain and metric (X-M1)."""
    arm_pairs = list(itertools.combinations(ARMS, 2))
    results = []
    for domain_key, metric in PAIRWISE_TESTS:
        values = {arm: [r[metric] for r in rows
                        if r["domain"] == domain_key and r["arm"] == arm and r[metric] is not None]
                  for arm in ARMS}
        for arm_a, arm_b in arm_pairs:
            test = welch_t_test(values[arm_a], values[arm_b])
            if test is None:
                continue
            corrected = bonferroni(test.p_value, len(arm_pairs))
            mean_a, mean_b = mean_std(values[arm_a])[0], mean_std(values[arm_b])[0]
            results.append({
                "comparison": f"{domain_key}|{metric}|{arm_a}|{arm_b}", "domain": domain_key, "metric": metric,
                "arm_a": arm_a, "arm_b": arm_b, "n_a": len(values[arm_a]), "n_b": len(values[arm_b]),
                "mean_a": mean_a, "mean_b": mean_b, "difference": mean_a - mean_b,
                "welch_t": test.t, "degrees_of_freedom": test.degrees_of_freedom,
                "p_value": test.p_value, "p_bonferroni": corrected,
                "significant_after_correction": corrected < SIGNIFICANCE_LEVEL,
            })
    return results


def print_pairwise(results):
    print(f"\nPairwise arm comparisons — Welch's t-test, Bonferroni over 10 pairs per row group "
          f"(significant: corrected p < {SIGNIFICANCE_LEVEL})")
    print(f"{'domain':8} {'metric':30} {'arm a':9} {'arm b':9} {'diff':>8} {'t':>6} {'df':>5} {'p':>7} {'p corr':>7}")
    for result in results:
        print(f"{result['domain']:8} {result['metric']:30} {result['arm_a']:9} {result['arm_b']:9} "
              f"{result['difference']:+8.4f} {result['welch_t']:6.2f} {result['degrees_of_freedom']:5.1f} "
              f"{result['p_value']:7.4f} {result['p_bonferroni']:7.4f}{'  *' if result['significant_after_correction'] else ''}")
    for domain_key, metric in PAIRWISE_TESTS:
        group = [r for r in results if r["domain"] == domain_key and r["metric"] == metric]
        if group:
            surviving = [f"{r['arm_a']} vs {r['arm_b']}" for r in group if r["significant_after_correction"]]
            print(f"  {domain_key} {metric}: {len(surviving)}/{len(group)} pairs differ after correction"
                  + (f" ({', '.join(surviving)})" if surviving else ""))


def report(domain_keys):
    """Read the completed runs back: the CSVs, the summary table, the corrected comparison."""
    rows = collect_rows(domain_keys)
    if not rows:
        print("No completed runs yet.")
        return
    write_rows("e01_anneal_direction.csv", RESULT_COLUMNS, rows)
    print_summary(rows)
    pairs = pairwise_rows(rows)
    if pairs:
        write_rows("e01_anneal_direction_pairs.csv", PAIR_COLUMNS, pairs)
        print_pairwise(pairs)


def main():
    domain_keys = [key for key in sys.argv[1:] if key in DOMAINS] or list(DOMAINS)
    runs = build_runs(domain_keys)
    print(f"{EXPERIMENT}: {len(runs)} runs over {domain_keys} "
          f"({len(ARMS)} arms x seeds per domain)\n")
    if not seeded_run_training(runs):
        print("Stopped — re-run the same command to resume; the report runs once every run is done.")
        return
    report(domain_keys)


if __name__ == "__main__":
    main()
