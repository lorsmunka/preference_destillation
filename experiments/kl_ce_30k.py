"""KL vs CE on 30k training examples (thesis 9.1; requested by the thesis session 2026-11-08, plan approved by the
user the same day: 30k examples, 3 seeds, all four arms). E-01 at 2,048 examples sat at the floor (every arm gave
the commonest answer), so the arms could not be compared there.

Every run copies a reference run's recorded settings (its info.json) and changes only the run name, the KL ratio
schedule, max_training_examples (30,000) and the seed (torch seed = shuffle_seed = 1, 2, 3, so the arms pair by
seed). The test set is the fixed tail of the batch directory, the same as the reference runs'.

  reddit  reference e01-reddit-pure-kl-s1 (3.4M student, 3 passes); arms as E-01: KL 1->1, CE 0->0, KL->CE 0.9->0.1,
          CE->KL 0.1->0.9. Gate first: pure KL seed 1 must clearly beat .626 (always the commonest label).
  math    reference math-agent-kl-to-ce-m-s1 (size m, 10 passes); arms as the math agent runs: KL 1->1, CE 0->0,
          KL->CE 1->0, CE->KL 0->1. On a rented card: the first 937 training files + all 266 test files on the
          network volume math-30k (the same test list as the full runs; experiments/upload_math_30k.sh).

    python experiments/kl_ce_30k.py reddit gate     # pure KL seed 1 only
    python experiments/kl_ce_30k.py reddit          # all 12 (completed runs are skipped)
    python experiments/kl_ce_30k.py math            # all 12
"""

import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import seeded_run_training  # noqa: E402
from library import TrainingRun  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = 30_000
SEEDS = (1, 2, 3)
REFERENCE = {"reddit": "e01-reddit-pure-kl-s1", "math": "math-agent-kl-to-ce-m-s1"}
ARMS = {
    "reddit": {"pure-kl": (1.0, 1.0), "pure-ce": (0.0, 0.0), "kl-to-ce": (0.9, 0.1), "ce-to-kl": (0.1, 0.9)},
    "math": {"pure-kl": (1.0, 1.0), "pure-ce": (0.0, 0.0), "kl-to-ce": (1.0, 0.0), "ce-to-kl": (0.0, 1.0)},
}
GATE_BASELINE = 0.626      # reddit: always giving the commonest label (E-01's majority baseline)


def reference_settings(domain_key):
    info = json.loads((REPO / "runs" / REFERENCE[domain_key] / "info.json").read_text(encoding="utf-8"))
    config = info.get("config", info)
    fields = {field.name for field in dataclasses.fields(TrainingRun)}
    return {key: value for key, value in config.items() if key in fields}


def domain_object(domain_key, name):
    if domain_key == "math":
        from library.domain.math_tutor.full import FullDomain
        return FullDomain("v1")
    from library.domain.registry import as_domain
    return as_domain(name)


def run_name(domain_key, arm, seed):
    return f"kl-ce-30k-{domain_key}-{arm}-s{seed}"


def build_runs(domain_key, gate=False):
    settings = reference_settings(domain_key)
    domain = domain_object(domain_key, settings.pop("domain"))
    runs = []
    for seed in SEEDS:
        for arm, (start, end) in ARMS[domain_key].items():
            if gate and (arm, seed) != ("pure-kl", 1):
                continue
            changed = dict(settings, run_name=run_name(domain_key, arm, seed), domain=domain,
                           experiment=f"kl-ce-30k-{domain_key}", kl_ratio_start=start, kl_ratio_end=end,
                           max_training_examples=EXAMPLES, shuffle_seed=seed,
                           description=f"KL vs CE at 30k examples: {domain_key}, {arm} ({start}->{end}), seed {seed}; "
                                       f"settings of {REFERENCE[domain_key]}")
            runs.append((seed, TrainingRun(**changed)))
    return runs


if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else "reddit"
    chosen = sys.argv[2] if len(sys.argv) > 2 else None      # "gate", or one run's name (one process per run)
    runs = build_runs(key, gate=chosen == "gate")
    if chosen and chosen != "gate":
        runs = [(seed, run) for seed, run in runs if run.run_name == chosen]
    sys.exit(0 if seeded_run_training(runs) else 1)
