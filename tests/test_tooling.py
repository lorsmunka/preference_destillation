"""The tooling / reporting fixes from the code review (X-C1, X-H1, X-H2, X-M1, X-M3, X-M4,
X-M6, X-M7, X-L1, X-L2), one test class per area: the majority baseline beside task
accuracy, the trainer's split rule in analysis, the code_recall ground truth, Welch +
Bonferroni, seed cohorts, results CSVs that are never wiped, metric parsing, the log record
fields, the dashboard. CPU only; synthetic runs and batches in a temporary folder (nothing
is read from or written to runs/, batches/ or experiments/results/).

    python -m unittest discover -s tests -p "test_tooling.py"
"""

import csv
import json
import os
import random
import re
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import _common
import progress
import library.tooling.run as run_module
from library.domain.code_recall.corpus import ProgramBuilder
from library.shared.logging.records import EvalEpoch, parse_record
from library.shared.metrics.task import (INVALID_LABEL, OTHER_LABEL, ClassificationMetric, ExactAnswerMetric,
                                         MathMetric, answer_type_accuracy, extract_answer)
from library.tooling import RunStore
from library.tooling.render import comparison_table, experiment_summary, run_summary
from library.tooling.significance import bonferroni, student_t_two_sided_p, welch_t_test

REDDIT_TRUTH = '{"tone": "neutral", "sentiment": "negative", "safety": "safe", "toxicity": "respectful"}'
# safety: 3 of 4 true labels are "safe"; toxicity: 2 and 2 -> pooled majority baseline 5/8.
CONFUSION = {
    "safety": {"harmful": {"harmful": 0, "safe": 1, OTHER_LABEL: 0, INVALID_LABEL: 0},
               "safe": {"harmful": 0, "safe": 3, OTHER_LABEL: 0, INVALID_LABEL: 0}},
    "toxicity": {"toxic": {"toxic": 0, "respectful": 2, OTHER_LABEL: 0, INVALID_LABEL: 0},
                 "respectful": {"toxic": 0, "respectful": 2, OTHER_LABEL: 0, INVALID_LABEL: 0}},
}


def eval_record(epoch, task_accuracy, confusion_matrices=None, **extra):
    return {"type": "eval_epoch", "epoch": epoch, "avg_loss": 0.1, "teacher_forced_accuracy": 0.9,
            "student_accuracy": 0.8, "task_accuracy": task_accuracy,
            "confusion_matrices": confusion_matrices or {}, "total_steps": 100, **extra}


def write_run(runs_dir, name, info, records=(), state=None):
    run_dir = Path(runs_dir) / name
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "info.json").write_text(json.dumps({"run_name": name, **info}), encoding="utf-8")
    with open(run_dir / "logs" / "training.jsonl", "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")
    if state is not None:
        (run_dir / "logs" / "state.json").write_text(json.dumps(state), encoding="utf-8")
    return run_dir


class TemporaryFolderTest(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp(prefix="test_tooling_"))
        self.runs = self.folder / "runs"
        self.runs.mkdir()

    def tearDown(self):
        shutil.rmtree(self.folder, ignore_errors=True)


class MajorityBaselineTest(TemporaryFolderTest):
    """X-C1: task accuracy is reported beside the majority baseline and the excess."""

    def test_summary_and_table_show_baseline_and_excess_from_logged_matrices(self):
        write_run(self.runs, "reddit-run", {"domain": "reddit_comment_sentiment", "status": "completed"},
                  [eval_record(0, 0.5, CONFUSION)])
        run = RunStore(self.runs).get("reddit-run")
        self.assertAlmostEqual(run.majority_baseline(), 5 / 8)
        self.assertAlmostEqual(run.task_accuracy_excess(), 0.5 - 5 / 8)
        summary = run_summary(run)
        self.assertIn("task=50.00% (majority baseline 62.50%, excess -12.50pp)", summary)
        table = comparison_table([run])
        self.assertIn("baseline", table.splitlines()[0])
        self.assertIn("62.50%", table)

    def test_logged_baseline_is_used_as_is(self):
        write_run(self.runs, "new-run", {"domain": "reddit_comment_sentiment", "status": "completed"},
                  [eval_record(0, 0.7, CONFUSION, majority_baseline=0.6, task_accuracy_excess=0.1)])
        run = RunStore(self.runs).get("new-run")
        self.assertEqual((run.majority_baseline(), run.task_accuracy_excess()), (0.6, 0.1))

    def test_free_form_run_has_no_baseline(self):
        write_run(self.runs, "fixed-tail-run", {"domain": "post_generation", "status": "completed",
                                                "split_rule": "fixed_tail"}, [eval_record(0, 0.99)])
        run = RunStore(self.runs).get("fixed-tail-run")
        self.assertIsNone(run.majority_baseline())
        self.assertIn("majority baseline —", run_summary(run))


class SplitRuleTest(TemporaryFolderTest):
    """X-H1: analysis uses the trainer's split_batch_files; the baseline of an old log is
    taken from the test set that log was actually scored on."""

    ANSWERS = {4: ["True", "5"], 9: ["True", "True"], 10: ["True", "7"]}   # batch index -> true answers

    def setUp(self):
        super().setUp()
        self.batches = self.folder / "batches"
        self.batches.mkdir()
        for index in range(1, 11):
            answers = self.ANSWERS.get(index, ["1", "2"])
            with open(self.batches / f"batch_{index}.jsonl", "w", encoding="utf-8") as file:
                for answer in answers:
                    file.write(json.dumps({"sentence": f"problem {index}", "steps": [],
                                           "model_response": f"A=1\nSolution: {answer};"}) + "\n")
        self.original_batches_dir = run_module.batches_dir
        run_module.batches_dir = lambda domain, teacher_model: self.batches
        run_module._TEST_SET_ANSWER_COUNTS.clear()
        # The rule before the fix: test = the last 20% of the first 8 // 2 = 4 files -> file 4.
        write_run(self.runs, "old-math-run",
                  {"domain": "math_word_problem", "status": "completed", "teacher_model": "teacher",
                   "max_training_examples": 8, "batch_size": 2, "training_test_ratio": 0.8},
                  [eval_record(0, 0.75)])
        self.run = RunStore(self.runs).get("old-math-run")

    def tearDown(self):
        run_module.batches_dir = self.original_batches_dir
        run_module._TEST_SET_ANSWER_COUNTS.clear()
        super().tearDown()

    def test_split_files_is_the_fixed_tail_whatever_max_training_examples_is(self):
        training_files, test_files = self.run.split_files()
        self.assertEqual([path.name for path in training_files],
                         ["batch_1.jsonl", "batch_2.jsonl", "batch_3.jsonl", "batch_4.jsonl"])
        self.assertEqual([path.name for path in test_files], ["batch_9.jsonl", "batch_10.jsonl"])

    def test_old_log_baseline_comes_from_the_test_set_it_was_scored_on(self):
        self.assertEqual([path.name for path in self.run.logged_test_files()], ["batch_4.jsonl"])
        self.assertAlmostEqual(self.run.majority_baseline(), 0.5)          # "True" 1 of 2
        self.assertAlmostEqual(self.run.task_accuracy_excess(), 0.25)
        composition = answer_type_accuracy(self.run.test_set_answer_counts())
        self.assertEqual((composition["boolean_examples"], composition["numeric_examples"]), (1, 1))

    def test_analysis_batch_paths_use_the_split_and_roles(self):
        from library.tooling.analysis import _batch_paths
        self.assertEqual([(path.name, role) for path, role in _batch_paths(self.run, "test", None)],
                         [("batch_9.jsonl", "test"), ("batch_10.jsonl", "test")])
        self.assertEqual(len(_batch_paths(self.run, "all", None)), 6)


class CodeRecallTruthTest(unittest.TestCase):
    """X-H2: the while loop's closing += / -= is an assignment of its variable."""

    def test_last_assignment_is_the_last_line_assigning_the_variable(self):
        loop_step_was_last = 0
        for seed in range(300):
            program = ProgramBuilder(random.Random(seed)).build()
            recorded_lines = {}
            for assignment in program.assignments:
                recorded_lines.setdefault(assignment["name"], []).append(assignment["line"])
            for name, lines in recorded_lines.items():
                pattern = rf"\s*{re.escape(name)}\s*(=(?!=)|\+=|-=)"
                in_text = [number for number, text in enumerate(program.lines, start=1) if re.match(pattern, text)]
                self.assertEqual(lines, in_text, (seed, name))
                loop_step_was_last += bool(re.match(rf"\s*{re.escape(name)} (\+=|-=)", program.lines[lines[-1] - 1]))
        self.assertGreater(loop_step_was_last, 0)   # the case the bug got wrong does occur

    def test_toy_while_loop(self):
        class ScriptedRandom(random.Random):
            """Picks the while branch with a decrement, then plain choices."""
            def choice(self, sequence):
                if "while" in sequence and "assign" in sequence:
                    return "while"
                if "decrement" in sequence:
                    return "decrement"
                return sequence[0]

            def randint(self, low, high):
                return low

            def random(self):
                return 0.99

        program = ProgramBuilder(ScriptedRandom())
        program.assignments.append({"name": "count", "line": 1, "value": "5", "scope": "main"})
        program.emit("count = 5", 0)
        program.block(0, "main", ["count"])
        step_lines = [number for number, text in enumerate(program.lines, start=1) if text.strip().startswith("count -=")]
        self.assertTrue(step_lines)
        self.assertEqual(max(a["line"] for a in program.assignments if a["name"] == "count"), step_lines[-1])


class MetricParsingTest(unittest.TestCase):
    """X-M6, X-M7 and the <other> column (X-L2)."""

    def test_non_object_json_is_invalid_not_a_crash(self):
        metric = ClassificationMetric()
        for text in ['"tone"', "5", "null", "[1, 2]"]:
            metric.update([text], REDDIT_TRUTH)
        result = metric.result()
        self.assertEqual((result.examples, result.validity_rate, result.accuracy), (4, 0.0, 0.0))
        self.assertEqual(result.confusion["tone"]["neutral"][INVALID_LABEL], 4)

    def test_out_of_set_and_missing_values_are_counted_in_other(self):
        metric = ClassificationMetric()
        metric.update(['{"tone": "meh", "sentiment": "negative", "safety": "safe"}'], REDDIT_TRUTH)
        confusion = metric.result().confusion
        self.assertEqual(confusion["tone"]["neutral"][OTHER_LABEL], 1)         # value outside the set
        self.assertEqual(confusion["toxicity"]["respectful"][OTHER_LABEL], 1)  # field missing
        self.assertEqual(sum(confusion["safety"]["safe"].values()), 1)

    def test_answer_extraction_prefers_the_answer_line(self):
        cases = {"Answer: 15;": "15", " Answer: 15;": "15", "\nAnswer: 15;": "15", " 15;": "15",
                 "24 - 9 = 15\nAnswer: 15;": "15", "foo;\nAnswer: 15;": "15", "24-9=15\nAnswer:15;": "15",
                 "Answer: 15; Answer: 16;": "15", "Answer: 15": None, "Tennis;": "tennis"}
        for text, expected in cases.items():
            self.assertEqual(extract_answer(text), expected, repr(text))
        metric = ExactAnswerMetric()
        metric.update(["▁Answer", ":", "▁15", ";"], "24 - 9 = 15\nAnswer: 15;")
        metric.update(["▁16", ";"], " 15;")
        result = metric.result()
        self.assertEqual((result.correct, result.examples), (1, 2))
        self.assertEqual(result.confusion, {"answer": {"15": {"15": 1, "16": 1}}})

    def test_math_answer_types_from_the_solution_matrix(self):
        metric = MathMetric()
        metric.update(["Solution: True;"], "Solution: True;")
        metric.update(["Solution: 5;"], "Solution: False;")
        metric.update(["no answer"], "Solution: 12;")
        metric.update(["Solution: 12;"], "Solution: 12;")
        result = metric.result()
        self.assertEqual(result.confusion["solution"]["12"], {"12": 1, INVALID_LABEL: 1})
        self.assertEqual(result.extra, {"boolean_examples": 2, "boolean_accuracy": 0.5,
                                        "boolean_majority_baseline": 0.5,
                                        "numeric_examples": 2, "numeric_accuracy": 0.5})
        self.assertEqual(answer_type_accuracy({}), {})


class LogRecordTest(unittest.TestCase):
    def test_eval_epoch_parses_the_baseline_fields(self):
        record = parse_record(eval_record(2, 0.64, majority_baseline=0.62, task_accuracy_excess=0.02))
        self.assertIsInstance(record, EvalEpoch)
        self.assertEqual((record.majority_baseline, record.task_accuracy_excess), (0.62, 0.02))
        legacy = parse_record({"type": "eval_epoch", "epoch": 0, "classification_accuracy": 0.6})
        self.assertEqual((legacy.task_accuracy, legacy.majority_baseline, legacy.task_accuracy_excess), (0.6, None, None))


class CohortTest(TemporaryFolderTest):
    """X-M3 and the X-L2 statistics: cohorts are tag + configuration, completed runs only."""

    def setUp(self):
        super().setUp()
        base = {"domain": "reddit_comment_sentiment", "experiment": "sweep", "status": "completed", "kl_ratio_end": 0.5}
        write_run(self.runs, "sweep-low-s1", {**base, "kl_ratio_start": 0.1, "shuffle_seed": 1,
                                              "description": "seed 1"}, [eval_record(0, 0.60, CONFUSION)])
        write_run(self.runs, "sweep-low-s2", {**base, "kl_ratio_start": 0.1, "shuffle_seed": 2,
                                              "description": "seed 2"}, [eval_record(0, 0.64, CONFUSION)])
        write_run(self.runs, "sweep-low-s3", {**base, "kl_ratio_start": 0.1, "status": "in_progress"},
                  [eval_record(0, 0.10, CONFUSION)])
        write_run(self.runs, "sweep-high-s1", {**base, "kl_ratio_start": 0.9}, [eval_record(0, 0.70, CONFUSION)])
        write_run(self.runs, "other-s1", {**base, "experiment": "other", "kl_ratio_start": 0.1},
                  [eval_record(0, 0.50, CONFUSION)])
        self.store = RunStore(self.runs)

    def test_cohorts_split_a_tag_by_configuration_and_skip_unfinished_runs(self):
        cohorts = self.store.by_experiment()
        self.assertEqual(sorted(cohorts), ["other", "sweep kl_ratio_start=0.1", "sweep kl_ratio_start=0.9"])
        self.assertEqual(sorted(run.name for run in cohorts["sweep kl_ratio_start=0.1"].runs),
                         ["sweep-low-s1", "sweep-low-s2"])
        self.assertEqual(len(self.store.experiment("sweep")), 3)

    def test_stats_use_the_sample_std_and_none_when_undefined(self):
        cohorts = self.store.by_experiment()
        low = cohorts["sweep kl_ratio_start=0.1"].stats("task_accuracy")
        self.assertAlmostEqual(low.mean, 0.62)
        self.assertAlmostEqual(low.std, 0.0282842712, places=8)   # sample std of 0.60, 0.64
        single = cohorts["sweep kl_ratio_start=0.9"].stats("task_accuracy")
        self.assertEqual((single.mean, single.std), (0.70, None))
        empty = cohorts["other"].stats("perplexity")
        self.assertEqual((empty.mean, empty.std, empty.n), (None, None, 0))
        excess = cohorts["sweep kl_ratio_start=0.1"].stats("task_accuracy_excess")
        self.assertAlmostEqual(excess.mean, 0.62 - 5 / 8)
        text = experiment_summary(list(cohorts.values()), metrics=("task_accuracy", "perplexity"))
        self.assertIn("perplexity=—", text)
        comparison = cohorts["sweep kl_ratio_start=0.1"].compare(cohorts["sweep kl_ratio_start=0.9"], "task_accuracy")
        self.assertIsNone(comparison.cohens_d)

    def test_training_hours_count_every_session(self):
        write_run(self.runs, "resumed", {"status": "completed", "started_at": "2026-01-01T10:00:00",
                                         "completed_at": "2026-01-01T11:00:00"},
                  state={"total_runtime_seconds": 7200})
        self.assertAlmostEqual(RunStore(self.runs).get("resumed").training_hours, 2.0)


class SignificanceTest(unittest.TestCase):
    """X-M1: Welch's t-test and the Bonferroni correction (checked against scipy offline)."""

    def test_t_distribution_tail(self):
        self.assertAlmostEqual(student_t_two_sided_p(2.228, 10), 0.05, places=3)
        self.assertAlmostEqual(student_t_two_sided_p(0.0, 7), 1.0)

    def test_welch_and_bonferroni(self):
        result = welch_t_test([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0])
        self.assertAlmostEqual(result.t, -1.7320508, places=6)                # scipy.stats.ttest_ind(..., equal_var=False)
        self.assertAlmostEqual(result.degrees_of_freedom, 4.4117647, places=6)
        self.assertAlmostEqual(result.p_value, 0.1515805048, places=8)
        self.assertIsNone(welch_t_test([1.0], [2.0, 3.0]))
        self.assertIsNone(welch_t_test([1.0, 1.0], [2.0, 2.0]))
        self.assertEqual(bonferroni(0.2, 10), 1.0)
        self.assertAlmostEqual(bonferroni(0.004, 10), 0.04)


class WriteRowsTest(unittest.TestCase):
    """X-M4: write_rows merges and never wipes a results CSV."""

    def setUp(self):
        self.folder = Path(tempfile.mkdtemp(prefix="test_write_rows_"))
        self.original_results_dir = _common.RESULTS_DIR
        _common.RESULTS_DIR = self.folder

    def tearDown(self):
        _common.RESULTS_DIR = self.original_results_dir
        shutil.rmtree(self.folder, ignore_errors=True)

    def read(self, name):
        with open(self.folder / name, encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            return reader.fieldnames, list(reader)

    def test_fewer_columns_keep_the_stored_data(self):
        _common.write_rows("t.csv", ["run", "a", "b"], [{"run": f"r{i}", "a": i, "b": i * 10} for i in range(5)])
        _common.write_rows("t.csv", ["run", "a"], [{"run": "r9", "a": 9, "unlisted": 1}, {"run": "r0", "a": 100}])
        fieldnames, rows = self.read("t.csv")
        self.assertEqual(fieldnames, ["run", "a", "b"])
        by_run = {row["run"]: row for row in rows}
        self.assertEqual(len(rows), 6)
        self.assertEqual((by_run["r1"]["a"], by_run["r1"]["b"]), ("1", "10"))
        self.assertEqual((by_run["r0"]["a"], by_run["r9"]["b"]), ("100", ""))
        self.assertEqual([path.name for path in self.folder.iterdir()], ["t.csv"])   # no temporary file left

    def test_a_different_key_column_is_refused_without_touching_the_file(self):
        _common.write_rows("t.csv", ["run", "a"], [{"run": "r1", "a": 1}])
        before = (self.folder / "t.csv").read_text(encoding="utf-8")
        with self.assertRaises(ValueError):
            _common.write_rows("t.csv", ["name", "a"], [{"name": "x", "a": 2}])
        self.assertEqual((self.folder / "t.csv").read_text(encoding="utf-8"), before)


class DashboardTest(TemporaryFolderTest):
    """X-L2: an abandoned in_progress run is shown as stopped, never as the current run;
    the planned total comes from this experiment's own queue."""

    def setUp(self):
        super().setUp()
        self.original_runs_dir = progress.RUNS_DIR
        progress.RUNS_DIR = self.runs

    def tearDown(self):
        progress.RUNS_DIR = self.original_runs_dir
        super().tearDown()

    def test_stale_in_progress_run_is_stopped_and_not_current(self):
        stale = write_run(self.runs, "abandoned", {"status": "in_progress", "started_at": "2026-01-02T00:00:00"})
        write_run(self.runs, "finished", {"status": "completed", "started_at": "2026-01-01T00:00:00"})
        old = time.time() - (progress.STALE_AFTER_MINUTES + 5) * 60
        for path in [stale / "info.json", stale / "logs" / "training.jsonl"]:
            os.utime(path, (old, old))
        infos = progress.run_infos()
        self.assertEqual(infos["abandoned"][0]["status"], "stopped")
        self.assertEqual(progress.current_run(infos), "finished")

    def test_planned_total_comes_from_the_experiments_own_queue(self):
        queues = [{"tag": "other", "kind": "training", "total": 10}, {"tag": "mine", "kind": "training", "total": 25}]
        self.assertEqual(progress.planned_run_count("mine", queues, None), 25)
        self.assertIsNone(progress.planned_run_count("missing", queues, None))


if __name__ == "__main__":
    unittest.main()
