"""Cross-experiment report for the five tested ideas.

Reads whatever cohorts exist in runs/ and writes both CSVs (experiments/results/) and a
markdown summary (thesis_work/results_first_pass.md). Log-only — no checkpoints reloaded,
so it is cheap to re-run while training is still going.

    python experiments/report_ideas.py

Covers: E1 vocab-rank sweep, E3 validity-vs-value gap (board B1), E5 data plateau,
and the E2 self-consistency table if its phase-2 CSV exists.
"""

import csv
import statistics
from collections import defaultdict
from pathlib import Path

from _common import RESULTS_DIR, best_and_final, write_rows

from library import RunStore

REPORT_PATH = Path(__file__).resolve().parents[1] / "thesis_work" / "results_first_pass.md"

INVALID_LABEL = "<invalid>"


def mean_std(values):
    usable = [v for v in values if v is not None]
    if not usable:
        return None, None, 0
    mean = statistics.mean(usable)
    std = statistics.pstdev(usable) if len(usable) > 1 else 0.0
    return mean, std, len(usable)


def format_stat(mean, std, scale=100, digits=2):
    if mean is None:
        return "—"
    return f"{mean * scale:.{digits}f}±{std * scale:.{digits}f}"


# ── B1: structural validity vs value accuracy ─────────────────────────
def confusion_breakdown(confusion_matrices):
    """Per-category value accuracy and invalid share from a logged confusion matrix."""
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
                "total": total,
            }
    return breakdown


def cohort_rows(runs, group_key, group_name):
    """Aggregate a cohort into one row per group value, mean±std across seeds."""
    groups = defaultdict(list)
    for run in runs:
        groups[group_key(run)].append(run)

    rows = []
    for value in sorted(groups):
        members = groups[value]
        best_evals, final_evals = [], []
        for run in members:
            best, final = best_and_final(run)
            if best:
                best_evals.append(best)
            if final:
                final_evals.append(final)
        if not final_evals:
            continue

        def stat(records, field):
            return mean_std([getattr(record, field, None) for record in records])

        task_best = stat(best_evals, "task_accuracy")
        task_final = stat(final_evals, "task_accuracy")
        validity = stat(final_evals, "validity_rate")
        overlap = stat(final_evals, "teacher_student_overlap")
        entropy = stat(final_evals, "student_entropy")
        teacher_entropy = stat(final_evals, "teacher_entropy")
        perplexity = stat(final_evals, "perplexity")
        kl = stat(final_evals, "kl_loss")
        tf = stat(final_evals, "teacher_forced_accuracy")
        student = stat(final_evals, "student_accuracy")
        termination = stat(final_evals, "termination_rate")

        # B1: the gap between "is it valid JSON" and "are the values right"
        gap_values = [
            (record.validity_rate - record.task_accuracy)
            for record in final_evals
            if record.validity_rate is not None
        ]
        gap_mean, gap_std, _ = mean_std(gap_values)

        per_category = defaultdict(list)
        for record in final_evals:
            for category, values in confusion_breakdown(record.confusion_matrices).items():
                per_category[category].append(values["accuracy"])

        row = {
            group_name: value,
            "runs": len(members),
            "task_best": task_best[0], "task_best_std": task_best[1],
            "task_final": task_final[0], "task_final_std": task_final[1],
            "validity": validity[0], "validity_std": validity[1],
            "validity_value_gap": gap_mean, "validity_value_gap_std": gap_std,
            "teacher_student_overlap": overlap[0], "teacher_student_overlap_std": overlap[1],
            "student_entropy": entropy[0], "teacher_entropy": teacher_entropy[0],
            "perplexity": perplexity[0], "kl_loss": kl[0],
            "teacher_forced_accuracy": tf[0], "student_accuracy": student[0],
            "termination_rate": termination[0],
        }
        for category, accuracies in per_category.items():
            row[f"value_accuracy_{category}"] = mean_std(accuracies)[0]
        rows.append(row)
    return rows


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


def main():
    store = RunStore()
    completed = [r for r in store.all() if r.status == "completed"]
    vocab_runs = [r for r in completed if r.name.startswith("ideaA-")]
    data_runs = [r for r in completed if r.name.startswith("ideaD-")]

    sections = ["# First-pass results — five tested ideas",
                "",
                "Auto-generated by `experiments/report_ideas.py`. Mean±std across seeds, "
                "percentages unless noted.", ""]

    # ── E1 + E4 ───────────────────────────────────────────────────────
    if vocab_runs:
        rows = cohort_rows(vocab_runs, lambda r: r.output_vocab_size, "output_vocab_size")
        columns = list(rows[0].keys())
        write_rows("ideaA_vocab_sweep.csv", columns, rows)

        probe = load_probe()
        sections += [
            f"## E1 — Output vocabulary vs the softmax/rank bottleneck ({len(vocab_runs)} runs)",
            "",
            "Matched body (48h/3L/1H), 2048 examples, KL 0.99->0.50. Only the auxiliary "
            "bank varies, so every arm can still express the task.", "",
        ]
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
                format_stat(row["validity"], row["validity_std"]),
                format_stat(row["teacher_student_overlap"], row["teacher_student_overlap_std"]),
                f"{row['student_entropy']:.3f}" if row["student_entropy"] is not None else "—",
                probe_stat("ece"), probe_stat("effective_rank", digits=2),
                probe_stat("head_gradient_fraction"),
            ])
        sections += [render_table(
            ["V", "n", "task best %", "task final %", "validity %", "overlap %",
             "student H", "ECE", "eff. rank", "head grad frac"], table_rows), ""]

    # ── E3 / B1 ───────────────────────────────────────────────────────
    if vocab_runs:
        rows = cohort_rows(vocab_runs, lambda r: r.output_vocab_size, "output_vocab_size")
        sections += ["## E3 — Structural validity vs value accuracy (board B1)", "",
                     "The gap is `validity_rate - task_accuracy`: output that parses as the "
                     "right shape but carries wrong leaf values.", ""]
        categories = sorted({key.replace("value_accuracy_", "") for row in rows
                             for key in row if key.startswith("value_accuracy_")})
        table_rows = []
        for row in rows:
            cells = [str(int(row["output_vocab_size"])),
                     format_stat(row["validity"], row["validity_std"]),
                     format_stat(row["task_final"], row["task_final_std"]),
                     format_stat(row["validity_value_gap"], row["validity_value_gap_std"])]
            for category in categories:
                value = row.get(f"value_accuracy_{category}")
                cells.append(f"{value * 100:.2f}" if value is not None else "—")
            table_rows.append(cells)
        sections += [render_table(["V", "validity %", "task %", "gap pp"] + [f"{c} %" for c in categories],
                                  table_rows), ""]

    # ── E5 ────────────────────────────────────────────────────────────
    if data_runs:
        rows = cohort_rows(data_runs,
                           lambda r: r.info.get("max_training_examples", 0), "examples")
        write_rows("ideaD_data_plateau.csv", list(rows[0].keys()), rows)
        sections += [f"## E5 — Data plateau ({len(data_runs)} runs)", "",
                     "Body, vocabulary and schedule fixed; only teacher-example count moves.", ""]
        table_rows = [[
            str(int(row["examples"])), str(row["runs"]),
            format_stat(row["task_best"], row["task_best_std"]),
            format_stat(row["task_final"], row["task_final_std"]),
            format_stat(row["validity"], row["validity_std"]),
            f"{row['teacher_forced_accuracy'] * 100:.2f}" if row["teacher_forced_accuracy"] else "—",
            f"{row['student_accuracy'] * 100:.2f}" if row["student_accuracy"] else "—",
        ] for row in rows]
        sections += [render_table(
            ["examples", "n", "task best %", "task final %", "validity %", "tf %", "student %"],
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
            cells = [arm, str(len(rows)), format_stat(greedy[0], greedy[1])]
            for key in temperature_keys:
                gain = column(key)
                cells.append(format_stat(gain[0], gain[1]))
            entropy = column("student_entropy")
            cells.append(f"{entropy[0]:.3f}" if entropy[0] is not None else "—")
            table_rows.append(cells)
        sections += [render_table(
            ["arm", "n", "greedy %"] + [f"gain {k.replace('vote_gain_t', 't')} pp" for k in temperature_keys]
            + ["student H"], table_rows), ""]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(sections), encoding="utf-8")
    print("\n".join(sections))
    print(f"\n-> {REPORT_PATH}")


if __name__ == "__main__":
    main()
