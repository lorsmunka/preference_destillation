"""Cross-experiment report for the five tested ideas.

Reads whatever cohorts exist in runs/ (completed runs only) and writes both CSVs
(experiments/results/) and a markdown summary (thesis_work/results_first_pass.md).
Log-only — no checkpoints reloaded, so it is cheap to re-run while training is still going.

    python experiments/report_ideas.py

Covers: E1 vocab-rank sweep, E3 validity-vs-value gap (board B1), E5 data plateau,
and the E2 self-consistency table if its phase-2 CSV exists.

Every task accuracy is shown beside its majority baseline (always giving the most common
label on the run's test set) and the excess over it (code review X-C1). Spreads are the
sample std across seeds. Teacher entropy is read from the test-split eval logs (X-M8).
"""

import csv
from collections import defaultdict
from pathlib import Path

from _common import RESULTS_DIR, best_and_final, mean_std, write_rows

from library import RunStore
from library.shared.metrics.task import INVALID_LABEL
from library.tooling.run import test_split_teacher_entropy
from library.training.evaluator import majority_baseline

REPORT_PATH = Path(__file__).resolve().parents[1] / "thesis_work" / "results_first_pass.md"


def format_stat(mean, std, scale=100, digits=2, signed=False):
    if mean is None:
        return "—"
    number = f"{mean * scale:+.{digits}f}" if signed else f"{mean * scale:.{digits}f}"
    return f"{number}±{std * scale:.{digits}f}" if std is not None else number


def format_value(value, scale=100, digits=2):
    return f"{value * scale:.{digits}f}" if value is not None else "—"


# ── B1: structural validity vs value accuracy ─────────────────────────
def confusion_breakdown(confusion_matrices):
    """Per-category value accuracy, invalid share and majority-label share from a logged
    confusion matrix."""
    breakdown = {}
    for category, matrix in (confusion_matrices or {}).items():
        correct = total = invalid = 0
        for true_label, predictions in matrix.items():
            for predicted_label, count in predictions.items():
                total += count
                if predicted_label == INVALID_LABEL:
                    invalid += count
                elif predicted_label == true_label:
                    correct += count
        if total:
            breakdown[category] = {
                "accuracy": correct / total,
                "invalid_share": invalid / total,
                "majority_baseline": majority_baseline({category: matrix}),
                "total": total,
            }
    return breakdown


def scored_test_examples(run):
    """How many test examples the run's logged task accuracy was scored on."""
    first_field = next(iter(run.test_set_answer_counts().values()), {})
    return sum(sum(row.values()) for row in first_field.values()) or None


def cohort_rows(runs, group_key, group_name):
    """Aggregate a cohort into one row per group value, mean±std across seeds."""
    groups = defaultdict(list)
    for run in runs:
        groups[group_key(run)].append(run)

    rows = []
    for value in sorted(groups):
        members = groups[value]
        best_evals, final_evals, baselines, excess_best, excess_final = [], [], [], [], []
        for run in members:
            best, final = best_and_final(run)
            if not final:
                continue
            best_evals.append(best)
            final_evals.append(final)
            baseline = run.majority_baseline(final)
            baselines.append(baseline)
            excess_best.append(best.task_accuracy - baseline if baseline is not None else None)
            excess_final.append(final.task_accuracy - baseline if baseline is not None else None)
        if not final_evals:
            continue

        def stat(records, field):
            return mean_std([getattr(record, field, None) for record in records])

        task_best = stat(best_evals, "task_accuracy")
        task_final = stat(final_evals, "task_accuracy")
        baseline = mean_std(baselines)
        excess_best_stat = mean_std(excess_best)
        excess_final_stat = mean_std(excess_final)
        validity = stat(final_evals, "validity_rate")
        overlap = stat(final_evals, "teacher_student_overlap")
        entropy = stat(final_evals, "student_entropy")
        perplexity = stat(final_evals, "perplexity")
        kl = stat(final_evals, "kl_loss")
        tf = stat(final_evals, "teacher_forced_accuracy")
        student = stat(final_evals, "student_accuracy")
        termination = stat(final_evals, "termination_rate")
        teacher_entropy = test_split_teacher_entropy(members)

        # B1: the gap between "is it valid JSON" and "are the values right"
        gap_values = [
            (record.validity_rate - record.task_accuracy)
            for record in final_evals
            if record.validity_rate is not None
        ]
        gap_mean, gap_std, _ = mean_std(gap_values)

        per_category = defaultdict(list)
        per_category_baseline = defaultdict(list)
        for record in final_evals:
            for category, values in confusion_breakdown(record.confusion_matrices).items():
                per_category[category].append(values["accuracy"])
                per_category_baseline[category].append(values["majority_baseline"])

        row = {
            group_name: value,
            "runs": len(final_evals),
            "test_examples": scored_test_examples(members[0]),
            "task_best": task_best[0], "task_best_std": task_best[1],
            "task_final": task_final[0], "task_final_std": task_final[1],
            "majority_baseline": baseline[0],
            "task_best_excess": excess_best_stat[0], "task_best_excess_std": excess_best_stat[1],
            "task_final_excess": excess_final_stat[0], "task_final_excess_std": excess_final_stat[1],
            "validity": validity[0], "validity_std": validity[1],
            "validity_value_gap": gap_mean, "validity_value_gap_std": gap_std,
            "teacher_student_overlap": overlap[0], "teacher_student_overlap_std": overlap[1],
            "student_entropy": entropy[0],
            "teacher_entropy": teacher_entropy["nats"] if teacher_entropy else None,
            "perplexity": perplexity[0], "kl_loss": kl[0],
            "teacher_forced_accuracy": tf[0], "student_accuracy": student[0],
            "termination_rate": termination[0],
        }
        for category, accuracies in per_category.items():
            row[f"value_accuracy_{category}"] = mean_std(accuracies)[0]
            row[f"majority_baseline_{category}"] = mean_std(per_category_baseline[category])[0]
        rows.append(row)
    return rows


def all_columns(rows):
    """Every key any row has, in first-seen order (rows differ when a log lacks matrices)."""
    columns = []
    for row in rows:
        columns += [key for key in row if key not in columns]
    return columns


def load_probe():
    path = RESULTS_DIR / "ideaA_bottleneck_probe.csv"
    if not path.exists():
        return {}
    by_vocab = defaultdict(list)
    with open(path, "r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            def number(key):
                try:
                    return float(row[key])
                except (TypeError, ValueError, KeyError):
                    return None
            by_vocab[int(float(row["output_vocab_size"]))].append({
                "ece": number("ece"),
                "effective_rank": number("effective_rank"),
                "head_gradient_fraction": number("head_gradient_fraction"),
                "overconfidence": number("overconfidence"),
            })
    summary = {}
    for vocab, entries in by_vocab.items():
        summary[vocab] = {
            field: mean_std([entry[field] for entry in entries])
            for field in ("ece", "effective_rank", "head_gradient_fraction", "overconfidence")
        }
    return summary


def render_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def teacher_entropy_note(runs):
    entropy = test_split_teacher_entropy(runs)
    if not entropy:
        return []
    return [f"Teacher entropy {entropy['nats']:.4f} nats (exp {entropy['exp_nats']:.3f}): {entropy['sample']}.", ""]


def main():
    store = RunStore()
    completed = store.completed()
    vocab_runs = [r for r in completed if r.name.startswith("ideaA-")]
    data_runs = [r for r in completed if r.name.startswith("ideaD-")]

    sections = ["# First-pass results — five tested ideas",
                "",
                "Auto-generated by `experiments/report_ideas.py` from completed runs. Mean±std "
                "across seeds (sample std, n-1), percentages unless noted. **baseline** = task "
                "accuracy of always giving the most common label on the run's test set; "
                "**excess** = task accuracy minus baseline, in percentage points. Best epoch is "
                "chosen on the test split itself (no validation split), so it is optimistic.", ""]

    # ── E1 + E4 ───────────────────────────────────────────────────────
    if vocab_runs:
        rows = cohort_rows(vocab_runs, lambda r: r.output_vocab_size, "output_vocab_size")
        write_rows("ideaA_vocab_sweep.csv", all_columns(rows), rows)

        probe = load_probe()
        sections += [
            f"## E1 — Output vocabulary vs the softmax/rank bottleneck ({len(vocab_runs)} runs)",
            "",
            "Matched body (48h/3L/1H), 2048 examples, KL 0.99->0.50. Only the auxiliary "
            "bank varies, so every arm can still express the task.", "",
        ] + teacher_entropy_note(vocab_runs)
        table_rows = []
        for row in rows:
            vocab = int(row["output_vocab_size"])
            probe_entry = probe.get(vocab, {})

            def probe_stat(field, scale=1, digits=3):
                entry = probe_entry.get(field)
                if not entry or entry[0] is None:
                    return "—"
                return f"{entry[0] * scale:.{digits}f}"

            table_rows.append([
                str(vocab), str(row["runs"]),
                format_stat(row["task_best"], row["task_best_std"]),
                format_stat(row["task_final"], row["task_final_std"]),
                format_value(row["majority_baseline"]),
                format_stat(row["task_best_excess"], row["task_best_excess_std"], signed=True),
                format_stat(row["task_final_excess"], row["task_final_excess_std"], signed=True),
                format_stat(row["validity"], row["validity_std"]),
                format_stat(row["teacher_student_overlap"], row["teacher_student_overlap_std"]),
                f"{row['student_entropy']:.3f}" if row["student_entropy"] is not None else "—",
                probe_stat("ece"), probe_stat("effective_rank", digits=2),
                probe_stat("head_gradient_fraction"),
            ])
        sections += [render_table(
            ["V", "n", "task best %", "task final %", "baseline %", "excess best pp", "excess final pp",
             "validity %", "overlap %", "student H", "ECE", "eff. rank", "head grad frac"], table_rows), ""]

    # ── E3 / B1 ───────────────────────────────────────────────────────
    if vocab_runs:
        rows = cohort_rows(vocab_runs, lambda r: r.output_vocab_size, "output_vocab_size")
        sections += ["## E3 — Structural validity vs value accuracy (board B1)", "",
                     "The gap is `validity_rate - task_accuracy`: output that parses as the "
                     "right shape but carries wrong leaf values. Each field's value accuracy is "
                     "shown as accuracy / that field's majority-label share.", ""]
        categories = sorted({key.replace("value_accuracy_", "") for row in rows
                             for key in row if key.startswith("value_accuracy_")})
        table_rows = []
        for row in rows:
            cells = [str(int(row["output_vocab_size"])),
                     format_stat(row["validity"], row["validity_std"]),
                     format_stat(row["task_final"], row["task_final_std"]),
                     format_stat(row["validity_value_gap"], row["validity_value_gap_std"])]
            for category in categories:
                cells.append(f"{format_value(row.get(f'value_accuracy_{category}'))} / "
                             f"{format_value(row.get(f'majority_baseline_{category}'))}")
            table_rows.append(cells)
        sections += [render_table(["V", "validity %", "task %", "gap pp"] + [f"{c} % / share %" for c in categories],
                                  table_rows), ""]

    # ── E5 ────────────────────────────────────────────────────────────
    if data_runs:
        rows = cohort_rows(data_runs,
                           lambda r: r.info.get("max_training_examples", 0), "examples")
        write_rows("ideaD_data_plateau.csv", all_columns(rows), rows)
        sections += [f"## E5 — Data plateau ({len(data_runs)} runs)", "",
                     "Body, vocabulary and schedule fixed; only teacher-example count moves. These "
                     "runs predate the fixed test set: each size was scored on its own test set "
                     "(the tail of its own data, see `test ex.`), so compare the excess column, "
                     "not raw task accuracy (code review X-H1).", ""] + teacher_entropy_note(data_runs)
        table_rows = [[
            str(int(row["examples"])), str(row["runs"]), str(row["test_examples"] or "—"),
            format_stat(row["task_best"], row["task_best_std"]),
            format_stat(row["task_final"], row["task_final_std"]),
            format_value(row["majority_baseline"]),
            format_stat(row["task_best_excess"], row["task_best_excess_std"], signed=True),
            format_stat(row["task_final_excess"], row["task_final_excess_std"], signed=True),
            format_stat(row["validity"], row["validity_std"]),
            format_value(row["teacher_forced_accuracy"]),
            format_value(row["student_accuracy"]),
        ] for row in rows]
        sections += [render_table(
            ["examples", "n", "test ex.", "task best %", "task final %", "baseline %", "excess best pp",
             "excess final pp", "validity %", "tf %", "student %"],
            table_rows), ""]

    # ── E2 ────────────────────────────────────────────────────────────
    path = RESULTS_DIR / "ideaG_self_consistency.csv"
    if path.exists():
        by_arm = defaultdict(list)
        with open(path, "r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                by_arm[row["arm"]].append(row)
        sections += ["## E2 — Self-consistency dividend (board G1)", "",
                     "Greedy vs majority vote over k samples on held-out math problems.", ""]
        temperature_keys = sorted({key for rows in by_arm.values() for row in rows
                                   for key in row if key.startswith("vote_gain_t")})
        table_rows = []
        for arm, rows in sorted(by_arm.items()):
            def column(name):
                return mean_std([float(row[name]) for row in rows
                                 if row.get(name) not in (None, "")])
            greedy = column("greedy_accuracy")
            baseline = column("majority_baseline")   # rows written before the X-C1 fix lack it: "—"
            cells = [arm, str(len(rows)), format_stat(greedy[0], greedy[1]), format_value(baseline[0])]
            for key in temperature_keys:
                gain = column(key)
                cells.append(format_stat(gain[0], gain[1]))
            entropy = column("student_entropy")
            cells.append(f"{entropy[0]:.3f}" if entropy[0] is not None else "—")
            table_rows.append(cells)
        sections += [render_table(
            ["arm", "n", "greedy %", "baseline %"] + [f"gain {k.replace('vote_gain_t', 't')} pp" for k in temperature_keys]
            + ["student H"], table_rows), ""]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(sections), encoding="utf-8")
    print("\n".join(sections))
    print(f"\n-> {REPORT_PATH}")


if __name__ == "__main__":
    main()
