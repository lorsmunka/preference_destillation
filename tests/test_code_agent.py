"""The coding agent (library/domain/code_agent): randomized project versions keep exact true places,
the scope and explanation checks, the two-step harness with a fake model, and its training records.
CPU only, no model weights.

    python -m unittest discover -s tests -p "test_code_agent.py"
"""

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from library.domain.code_agent import CodeAgentDomain, ScopeMetric
from library.domain.code_agent.checks import (canonical_scope, check_explanation, check_scope, made_up_names, parse_scope,
                                              score_scope, usable_ranges)
from library.domain.code_agent.harness import CodeAgentSession, transcript_records
from library.domain.code_agent.project import (check_base, identifiers, known_names, load_base, make_version, project_list,
                                               rename, reorder, symbol_lines)
from library.domain.code_agent.requests import version_requests
from library.agent.runner import run_sessions


class BaseProjects(unittest.TestCase):
    def test_every_base_project_is_consistent(self):
        for project in project_list():
            self.assertEqual(check_base(load_base(project)), [], project)

    def test_a_version_is_the_same_every_time_and_differs_from_the_next(self):
        base = load_base("banking")
        self.assertEqual(make_version(base, 7).files, make_version(base, 7).files)
        self.assertNotEqual(make_version(base, 7).files, make_version(base, 8).files)

    def test_every_topic_place_exists_in_every_version(self):
        for project in project_list():
            base = load_base(project)
            for number in range(40):
                version = make_version(base, number)   # make_version itself checks definitions and renames
                places = symbol_lines(version.files)
                for topic in base["topics"]:
                    for place in topic["places"]:
                        self.assertIn(version.place(place), places, (version.id, place))


class Renaming(unittest.TestCase):
    def test_names_in_code_and_f_strings_are_renamed(self):
        text = 'def withdraw(amount):\n    print(f"{withdraw.__name__} {amount}")\n    return withdraw\n'
        renamed = rename(text, {"withdraw": "take_out"})
        self.assertNotIn("withdraw", renamed)
        self.assertIn('f"{take_out.__name__} {amount}"', renamed)

    def test_plain_words_stay_in_strings_and_code_like_names_change_there(self):
        text = '"""Make a Transaction for a withdraw."""\nkind = "withdraw"\n\n\nclass Transaction:\n    pass\n'
        renamed = rename(text, {"withdraw": "take_out", "Transaction": "Payment"})
        self.assertIn('"""Make a Payment for a withdraw."""', renamed)
        self.assertIn('kind = "withdraw"', renamed)
        self.assertIn("class Payment:", renamed)


class Reordering(unittest.TestCase):
    TEXT = ('"""Doc."""\n\nimport os\n\nLIMIT = 3\n\n\n# helper\ndef a():\n    return 1\n\n\ndef b():\n    return 2\n\n\n'
            'class C:\n    def __init__(self):\n        self.x = 1\n\n    def m(self):\n        return 1\n\n    def n(self):\n'
            '        return 2\n\n\ndef d():\n    return 4\n\n\nif __name__ == "__main__":\n    d()\n')

    def test_definitions_move_but_the_head_and_the_main_guard_stay(self):
        for seed in range(20):
            result = reorder(self.TEXT, random.Random(seed))
            lines = result.splitlines()
            self.assertEqual(lines[:5], ['"""Doc."""', "", "import os", "", "LIMIT = 3"])
            self.assertEqual(lines[-2:], ['if __name__ == "__main__":', "    d()"])
            self.assertEqual(set(symbol_lines({"f.py": lines})), set(symbol_lines({"f.py": self.TEXT.splitlines()})))
            self.assertLess(result.index("# helper"), result.index("def a():"))   # a comment moves with its function
            self.assertLess(result.index("def __init__"), result.index("def m(") if "def m(" in result else 0)

    def test_the_order_does_change(self):
        orders = {tuple(line for line in reorder(self.TEXT, random.Random(seed)).splitlines() if line.startswith("def "))
                  for seed in range(20)}
        self.assertGreater(len(orders), 1)


FILES = {"accounts.py": [f"line {number}" for number in range(1, 41)], "fees.py": [f"line {number}" for number in range(1, 21)]}


class ScopeChecks(unittest.TestCase):
    def test_a_good_scope_parses_and_passes(self):
        text = "<scope>\naccounts.py 12-30\nfees.py 7\n</scope>"
        self.assertEqual(parse_scope(text), {"accounts.py": [(12, 30)], "fees.py": [(7, 7)]})
        self.assertIsNone(check_scope(text, FILES))

    def test_bad_scopes_are_refused(self):
        self.assertIn("no file", check_scope("<scope>\nbank.py 1-3\n</scope>", FILES))
        self.assertIn("outside", check_scope("<scope>\nfees.py 15-25\n</scope>", FILES))
        self.assertIn("start after end", check_scope("<scope>\nfees.py 9-3\n</scope>", FILES))
        self.assertIn("cut off", check_scope("<scope>\nfees.py 1-3\n", FILES))
        self.assertEqual(check_scope("<scope>\nthe fees file\n</scope>", FILES), "no file ranges")

    def test_usable_ranges_clip_to_the_files(self):
        self.assertEqual(usable_ranges("<scope>\nfees.py 15-25\nbank.py 1-2\n</scope>", FILES), {"fees.py": [(15, 20)]})

    def test_the_canonical_scope_merges_and_sorts(self):
        self.assertEqual(canonical_scope("<scope>\nfees.py 7-9\naccounts.py 3-5\naccounts.py 6-8\n</scope>"),
                         "accounts.py 3-8; fees.py 7-9")
        self.assertIsNone(canonical_scope("<explanation>x</explanation>"))

    def test_scores_count_two_places_in_one_file(self):
        places = {("fees.py", "a"): (1, 4), ("fees.py", "b"): (10, 13)}
        score = score_scope({"fees.py": [(1, 4)]}, [("fees.py", "a"), ("fees.py", "b")], places)
        self.assertEqual((score["recall"], score["precision"], score["places_hit"]), (0.5, 1.0, 1))

    def test_the_metric_compares_scopes_and_skips_explanations(self):
        metric = ScopeMetric()
        metric.update(["<scope>", "\n", "fees.py", "▁", "7", "-", "9", "\n", "</scope>"], "<scope>\nfees.py 7-9\n</scope>")
        metric.update(["<explanation>"], "<explanation>It adds.</explanation>")
        self.assertEqual((metric.correct, metric.examples), (1, 1))


class ExplanationChecks(unittest.TestCase):
    def test_made_up_names_are_found(self):
        names = {"withdraw", "check_withdrawal", "Account"}
        self.assertEqual(made_up_names("The `withdraw` method calls check_withdrawal() and apply_fee() on a BankManager.", names),
                         ["BankManager", "apply_fee"])
        self.assertIsNone(check_explanation("<explanation>`withdraw` takes money out.</explanation>", names))
        self.assertIn("cut off", check_explanation("<explanation>`withdraw` takes", names))

    def test_python_words_are_not_made_up(self):
        self.assertEqual(made_up_names("It enters a `while` loop and calls `len()` on the `list`.", set()), [])

    def test_string_values_count_as_names(self):
        version = make_version(load_base("banking"), 0)
        self.assertIn("transfer_in", known_names(version.files))


class ScriptedModel:
    """Replies in order, whatever is asked."""

    def __init__(self, replies):
        self.replies = list(replies)

    def generate(self, requests):
        return [self.replies.pop(0) for _ in requests]


class Harness(unittest.TestCase):
    def setUp(self):
        self.base = load_base("banking")
        self.version = make_version(self.base, 0)
        self.item = {"id": "banking~v0#1", "project": "banking", "version": 0, "split": "train",
                     **version_requests(self.base, self.version)[1]}   # an explain request
        file_name, name = self.item["truth"][0]
        self.start, self.end = symbol_lines(self.version.files)[(file_name, name)]
        self.file_name = file_name

    def test_scope_then_explanation_gives_two_records(self):
        session = CodeAgentSession(self.item, self.version)
        model = ScriptedModel([f"{self.file_name} {self.start}-{self.end}\n</scope>",
                               "<explanation>It checks the amount first.</explanation>"])
        run_sessions(model, [session])
        self.assertEqual(session.turn["route"], "explain")
        self.assertTrue(session.turn["route_correct"])
        records = transcript_records(self.item, self.version, session.turn)
        self.assertEqual([record["turn_type"] for record in records], ["scope", "explain"])
        self.assertTrue(records[0]["text"].startswith("<project>"))
        self.assertTrue(records[1]["text"].startswith("<context>"))
        self.assertIn(f'<file name="{self.file_name}">', records[1]["text"])
        self.assertEqual(records[1]["target"], "<explanation>It checks the amount first.</explanation>")

    def test_a_bad_scope_is_asked_for_again_once(self):
        session = CodeAgentSession(self.item, self.version)
        run_sessions(ScriptedModel(["nowhere.py 1-2\n</scope>", f"{self.file_name} {self.start}-{self.end}\n</scope>",
                                    "<found/>"]), [session])
        self.assertEqual(len(session.turn["rejected"]), 1)
        self.assertIsNone(session.turn["scope_check"])
        self.assertEqual(session.turn["route"], "find")
        self.assertFalse(session.turn["route_correct"])

    def test_a_little_junk_before_the_tag_is_dropped(self):
        session = CodeAgentSession(self.item, self.version)
        run_sessions(ScriptedModel([f"{self.file_name} {self.start}-{self.end}\n</scope>",
                                    "<channel|><explanation>It checks.</explanation>"]), [session])
        self.assertEqual(session.turn["output"], "<explanation>It checks.</explanation>")
        self.assertEqual(session.turn["route"], "explain")

    def test_an_unusable_scope_ends_the_item_without_records(self):
        session = CodeAgentSession(self.item, self.version)
        run_sessions(ScriptedModel(["nowhere.py 1-2\n</scope>", "nowhere.py 3-4\n</scope>"]), [session])
        self.assertTrue(session.done)
        self.assertEqual(transcript_records(self.item, self.version, session.turn), [])

    def test_the_domain_reads_both_steps(self):
        domain = CodeAgentDomain()
        self.assertTrue(domain.structural_validity("<scope>\nfees.py 7-9\n</scope>"))
        self.assertTrue(domain.structural_validity("<found/>"))
        self.assertFalse(domain.structural_validity("<explanation>no end"))
        self.assertIn("<project>", domain.teacher_prompt("<project>\n</project>"))


class TeacherSetup(unittest.TestCase):
    def test_a_pass_keeps_its_first_setup_and_refuses_another(self):
        import tempfile
        from library.agent import runner
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "v1.jsonl"
            first = runner.teacher_setup(path)
            self.assertEqual(runner.teacher_setup(path), first)
            self.assertEqual(runner.recorded_setup(path), first)
            saved = runner.FULL_PRECISION_MEMORY_GB
            runner.FULL_PRECISION_MEMORY_GB = 0 if first else 10_000   # a card that gives the other setup
            try:
                with self.assertRaises(SystemExit):
                    runner.teacher_setup(path)
            finally:
                runner.FULL_PRECISION_MEMORY_GB = saved

    def test_replies_without_a_setup_are_refused(self):
        import tempfile
        from library.agent import runner
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "old.jsonl"
            path.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                runner.teacher_setup(path)


if __name__ == "__main__":
    unittest.main()
