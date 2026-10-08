"""E2 — Does the KL/CE ratio control a self-consistency dividend? (board G1)

Claim under test: KL-heavy distillation preserves the teacher's spread, so sampling
several answers and voting buys accuracy; CE-heavy distillation mode-collapses, so
voting buys nothing. If true, the measured CE~=KL tie at one-shot decoding (ledger #1,
Menon) hides a test-time gap. A null is also a result: tiny students may be too small to
carry usable diversity, which would be a counterpoint to the 7B-scale finding (2604.16027).

Math is the only domain with headroom — reddit's labels are near-deterministic, so
sampling cannot disagree usefully.

    python experiments/self_consistency_dividend.py

Phase 1 trains 2 loss arms x 5 seeds. Phase 2 is inference only: greedy vs majority vote
over k samples at several temperatures, on the held-out split.
"""

from collections import Counter

from _common import seeded_run_training, write_rows

from library import RunStore, TrainingRun
from library.domain import MATH_WORD_PROBLEM
from library.shared.batch_files import read_batch

EXPERIMENT = "ideaG-self-consistency"

SEEDS = [1, 2, 3, 4, 5]
VOTE_EXAMPLES = 100      # held-out problems scored per run
SAMPLES_PER_EXAMPLE = 5  # k in "sample k and vote"
TEMPERATURES = [0.7, 1.0]

LOSS_ARMS = {
    # Pure CE: match the argmax token, no distribution information.
    "purece": dict(kl_ratio_start=0.0, kl_ratio_end=0.0,
                   distillation_temperature_start=1.0, distillation_temperature_end=1.0),
    # Pure KL: match the full teacher distribution.
    "purekl": dict(kl_ratio_start=1.0, kl_ratio_end=1.0,
                   distillation_temperature_start=1.0, distillation_temperature_end=1.0),
}

FIXED = dict(
    domain=MATH_WORD_PROBLEM,
    hidden_dim=48,
    num_layers=3,
    num_heads=1,
    dropout=0.08,
    epoch_count=3,
    batch_size=32,
    learning_rate=0.0015,
    lr_warmup_ratio=0.1,
    max_training_examples=4096,
    training_test_ratio=0.8,
    auxiliary_token_percentage=1.0,
)


def build_runs():
    seeded = []
    for arm, loss_config in LOSS_ARMS.items():
        for seed in SEEDS:
            run = TrainingRun(
                **FIXED,
                **loss_config,
                run_name=f"ideaG-{arm}-s{seed}",
                experiment=EXPERIMENT,
                description=f"Self-consistency arm {arm}, seed {seed}",
            )
            seeded.append((seed, run))
    return seeded


# ── phase 2: sampling + voting ────────────────────────────────────────
def test_examples(run, limit):
    """Held-out examples for this run: the first `limit` of its test set by the trainer's
    own split rule (Run.split_files), the fixed test tail no run trained on."""
    from library.training.batch_handler import TEST, select_records
    examples = []
    for path in run.split_files()[1]:
        for example in select_records(read_batch(path), TEST):
            examples.append(example)
            if len(examples) >= limit:
                return examples
    return examples


def vote(solutions):
    """Majority answer among the extracted solutions, ignoring failures to produce one."""
    usable = [s for s in solutions if s is not None]
    if not usable:
        return None
    return Counter(usable).most_common(1)[0][0]


def measure_run(run_name, metric_factory):
    from library import Analysis
    from library.model.student import StudentModel

    analysis = Analysis(run_name)
    run = analysis.run
    label, checkpoint_path = analysis.store.resolve_checkpoint(run_name, "latest")
    student = StudentModel.from_run(run, checkpoint_path)
    extractor = metric_factory()
    examples = test_examples(run, VOTE_EXAMPLES)

    greedy_correct = 0
    scored = 0
    true_answers = Counter()
    voted_correct = {temperature: 0 for temperature in TEMPERATURES}
    distinct_total = {temperature: 0 for temperature in TEMPERATURES}

    for example in examples:
        truth = extractor._extract_solution(example.get("model_response", ""))
        if truth is None:
            continue
        scored += 1
        true_answers[truth] += 1

        greedy = student.generate(example["sentence"], max_new_tokens=student.domain.max_steps,
                                  temperature=0.0)
        if extractor._extract_solution(greedy.text) == truth:
            greedy_correct += 1

        for temperature in TEMPERATURES:
            solutions = []
            for _ in range(SAMPLES_PER_EXAMPLE):
                sample = student.generate(example["sentence"],
                                          max_new_tokens=student.domain.max_steps,
                                          temperature=temperature)
                solutions.append(extractor._extract_solution(sample.text))
            distinct_total[temperature] += len({s for s in solutions if s is not None})
            if vote(solutions) == truth:
                voted_correct[temperature] += 1

    if scored == 0:
        return None

    majority_baseline = max(true_answers.values()) / scored   # always giving the most common answer
    row = {
        "run": run_name,
        "arm": run.info.get("description", "").split()[2].strip(",") if run.info.get("description") else "",
        "checkpoint": label,
        "scored_examples": scored,
        "greedy_accuracy": greedy_correct / scored,
        "majority_baseline": majority_baseline,
        "greedy_excess": greedy_correct / scored - majority_baseline,
        "student_entropy": run.final_eval.student_entropy if run.final_eval else None,
        "logged_task_accuracy": run.final_eval.task_accuracy if run.final_eval else None,
    }
    for temperature in TEMPERATURES:
        key = str(temperature).replace(".", "")
        row[f"voted_accuracy_t{key}"] = voted_correct[temperature] / scored
        row[f"vote_gain_t{key}"] = voted_correct[temperature] / scored - greedy_correct / scored
        row[f"mean_distinct_t{key}"] = distinct_total[temperature] / scored
    return row


def run_phase_two():
    from library.shared.metrics.task import MathMetric

    columns = ["run", "arm", "checkpoint", "scored_examples", "greedy_accuracy",
               "majority_baseline", "greedy_excess", "student_entropy", "logged_task_accuracy"]
    for temperature in TEMPERATURES:
        key = str(temperature).replace(".", "")
        columns += [f"voted_accuracy_t{key}", f"vote_gain_t{key}", f"mean_distinct_t{key}"]

    store = RunStore()
    completed = {run.name for run in store.completed()}
    rows = []
    for arm in LOSS_ARMS:
        for seed in SEEDS:
            run_name = f"ideaG-{arm}-s{seed}"
            if run_name not in completed:
                print(f"Skipping {run_name}: not a completed run")
                continue
            print(f"Sampling {run_name} ...")
            row = measure_run(run_name, MathMetric)
            if row:
                row["arm"] = arm
                rows.append(row)
    if rows:
        write_rows("ideaG_self_consistency.csv", columns, rows)
    return rows


def main():
    runs = build_runs()
    print(f"{EXPERIMENT}: {len(runs)} runs ({len(LOSS_ARMS)} arms x {len(SEEDS)} seeds)\n")
    if not seeded_run_training(runs):
        print("Stopped — re-run to resume; phase 2 runs once every run is trained.")
        return
    print(f"\nPhase 2: greedy vs vote-of-{SAMPLES_PER_EXAMPLE} on {VOTE_EXAMPLES} held-out problems\n")
    run_phase_two()
    print("\nDone. Analyse with: python experiments/report_ideas.py")


if __name__ == "__main__":
    main()
