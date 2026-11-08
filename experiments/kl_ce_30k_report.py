"""The report of the KL vs CE runs at 30k examples (experiments/kl_ce_30k.py), for thesis 9.1: every pass of every
run (the curves), the final pass, and the seed-paired differences between the arms at the final pass (a paired t-test
over the seeds, df = seeds - 1). Results through _common.write_rows into experiments/results/:

  kl_ce_30k_curves.csv   one row per run and pass
  kl_ce_30k_final.csv    one row per run (its final pass)
  kl_ce_30k_paired.csv   one row per domain, metric and pair of arms: mean difference (first minus second), its
                         sample std, t, p, and how many seeds favour the first arm

    python experiments/kl_ce_30k_report.py
"""

import math
import statistics
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import write_rows  # noqa: E402
from kl_ce_30k import ARMS, SEEDS, run_name  # noqa: E402
from library import RunStore  # noqa: E402
from library.tooling.significance import student_t_two_sided_p  # noqa: E402

METRICS = ["task_accuracy", "ce_loss", "topk_accuracy", "teacher_student_overlap", "mean_target_rank",
           "teacher_forced_accuracy", "student_accuracy", "student_entropy"]
LOWER_IS_BETTER = {"ce_loss", "mean_target_rank"}


def paired(differences):
    count = len(differences)
    mean = statistics.mean(differences)
    if count < 2:
        return mean, None, None, None
    spread = statistics.stdev(differences)
    if spread == 0:
        return mean, 0.0, None, None
    t = mean / (spread / math.sqrt(count))
    return mean, spread, t, student_t_two_sided_p(t, count - 1)


def main():
    store = RunStore()
    curves, finals, pairs = [], [], []
    final_values = {}          # (domain, arm, seed) -> {metric: value}
    for domain in ARMS:
        for arm in ARMS[domain]:
            for seed in SEEDS:
                name = run_name(domain, arm, seed)
                try:
                    run = store.get(name)
                except Exception:      # noqa: BLE001 - not trained (yet)
                    continue
                if run.status != "completed" or not run.eval_epochs:
                    continue
                for record in run.eval_epochs:
                    row = {"key": f"{name}|pass{record.epoch + 1}", "run": name, "domain": domain, "arm": arm,
                           "seed": seed, "pass": record.epoch + 1, "majority_baseline": record.majority_baseline}
                    row.update({metric: getattr(record, metric, None) for metric in METRICS})
                    curves.append(row)
                last = run.eval_epochs[-1]
                final = {metric: getattr(last, metric, None) for metric in METRICS}
                final_values[(domain, arm, seed)] = final
                finals.append({"run": name, "domain": domain, "arm": arm, "seed": seed, "pass": last.epoch + 1,
                               "majority_baseline": last.majority_baseline, **final})
        for first, second in combinations(ARMS[domain], 2):
            seeds = [seed for seed in SEEDS if (domain, first, seed) in final_values
                     and (domain, second, seed) in final_values]
            if not seeds:
                continue
            for metric in METRICS:
                values = [(final_values[(domain, first, seed)][metric], final_values[(domain, second, seed)][metric])
                          for seed in seeds]
                if any(a is None or b is None for a, b in values):
                    continue
                differences = [a - b for a, b in values]
                mean, spread, t, p = paired(differences)
                better = sum((d < 0) if metric in LOWER_IS_BETTER else (d > 0) for d in differences)
                pairs.append({"key": f"{domain}|{metric}|{first}-vs-{second}", "domain": domain, "metric": metric,
                              "first": first, "second": second, "seeds": len(seeds), "mean_difference": mean,
                              "std_difference": spread, "t": t, "p": p, "seeds_first_better": better})
    write_rows("kl_ce_30k_curves.csv", ["key", "run", "domain", "arm", "seed", "pass", "majority_baseline", *METRICS],
               curves)
    write_rows("kl_ce_30k_final.csv", ["run", "domain", "arm", "seed", "pass", "majority_baseline", *METRICS], finals)
    write_rows("kl_ce_30k_paired.csv", ["key", "domain", "metric", "first", "second", "seeds", "mean_difference",
                                        "std_difference", "t", "p", "seeds_first_better"], pairs)
    for domain in ARMS:
        print(f"== {domain}: final pass, mean over seeds")
        for arm in ARMS[domain]:
            values = [final_values[(domain, arm, seed)] for seed in SEEDS if (domain, arm, seed) in final_values]
            if values:
                print(f"  {arm:9s} n={len(values)}  " + "  ".join(
                    f"{metric}={statistics.mean(value[metric] for value in values):.4f}"
                    for metric in ("task_accuracy", "ce_loss", "student_accuracy")))
    print(f"{len(curves)} pass rows, {len(finals)} runs, {len(pairs)} paired comparisons -> experiments/results/")


if __name__ == "__main__":
    main()
