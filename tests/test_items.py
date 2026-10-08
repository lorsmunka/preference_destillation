"""The math-tutor item fixes from the code review (I-H1, I-M1, I-M2, I-M3, I-L1, I-L2, I-L5),
one test class per fix, plus the validator's own broken-item cases. CPU only, no data files.

    python -m unittest discover -s tests -p "test_items.py"
"""

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from library.domain.math_tutor import validator_cases
from library.domain.math_tutor.benchmarks import check as range_check
from library.domain.math_tutor.functions import NotAllowed, execute
from library.domain.math_tutor.items import validate
from library.domain.math_tutor.prune import is_heldout
from library.domain.math_tutor.renumber import fixed_values, renumber_item, replace_in_text, slots_of
from library.domain.math_tutor.reword import expand
from library.domain.math_tutor.similarity import normalise

GOOD = validator_cases.GOOD


def with_follow_up(question: str, solve: str) -> dict:
    return {**GOOD, "follow_ups": [{"question": question, "solve": solve}]}


class HeldOutShapeInAnyTurn(unittest.TestCase):   # I-H1
    def test_follow_up_with_a_heldout_shape_holds_the_item_out(self):
        item = with_follow_up("She sells 5 more and then bakes 8. How many muffins does she have now?",
                              "def solve(state):\n    now = state['left'] - 5 + 8\n    return {**state, 'left': now, 'answer': now}")
        validated, reason = validate(item)
        self.assertIsNone(reason)
        self.assertEqual(validated["problem_shape"], "sub")
        self.assertEqual(validated["shape"], "sub|sub>add")
        self.assertTrue(is_heldout(validated))

    def test_operations_chained_across_turns_are_not_held_out(self):
        validated, reason = validate(GOOD)   # "sub" then "add": sub>add across two turns, never inside one
        self.assertIsNone(reason)
        self.assertEqual(validated["shape"], "sub|add")
        self.assertFalse(is_heldout(validated))

    def test_heldout_problem_shape_is_still_held_out(self):
        self.assertTrue(is_heldout({"shape": "add>mul|sub"}))


class HyphenatedNumberKeepsItsUnit(unittest.TestCase):   # I-M1
    def test_only_the_number_half_changes(self):
        self.assertEqual(replace_in_text("He has 11 ten-dollar notes.", {11: 20, 10: 13}), "He has 20 thirteen-dollar notes.")
        self.assertEqual(replace_in_text("A three-day trip.", {3: 5}), "A five-day trip.")
        self.assertEqual(replace_in_text("Ten-kilogram bags.", {10: 7}), "Seven-kilogram bags.")

    def test_number_compounds_still_change_whole(self):
        self.assertEqual(replace_in_text("She read twenty-one pages.", {21: 34}), "She read thirty-four pages.")
        self.assertEqual(replace_in_text("Twenty-one pages.", {21: 9}), "Nine pages.")


class FixedValuesAreNotRenumbered(unittest.TestCase):   # I-M2
    def test_fixed_values_are_found(self):
        self.assertEqual(fixed_values("She pays with a $20 note."), {20})
        self.assertEqual(fixed_values("He has 11 ten-dollar notes and two $2 coins."), {10, 2})
        self.assertEqual(fixed_values("The show starts at 3 o'clock; the shop opens at 9am."), {3, 9})
        self.assertEqual(fixed_values("It opens at six in the morning."), {6})
        self.assertEqual(fixed_values("She lives on the 9th floor."), {9})
        self.assertEqual(fixed_values("Each dog has 4 legs and each cart has two wheels, and a three-legged stool."), {4, 2, 3})
        self.assertEqual(fixed_values("An octopus with 8 arms and a square with four sides."), {8, 4})
        self.assertEqual(fixed_values("I am sure 12 cars came, and 14 children ate in the afternoon."), set())

    def test_banknote_is_never_renumbered(self):
        item = {**GOOD, "id": "S09-case-0001", "slice": "S09", "context": "shop",
                "problem": "Ava buys 3 books at $6 each and pays with a $20 note. How much change does she get?",
                "solve": "def solve():\n    cost = 3 * 6\n    change = 20 - cost\n    return {'cost': cost, 'answer': change}",
                "follow_ups": []}
        validated, reason = validate(item)
        self.assertIsNone(reason)
        validated["split"] = "train"
        self.assertEqual(slots_of(validated), [3, 6])
        copies, outcome = renumber_item(validated, random.Random(0))
        self.assertEqual(outcome, "ok")
        for copy in copies:
            self.assertIn("$20 note", copy["problem"])
            self.assertIn("20 - cost", copy["solve"])


class SeededSiblingsShareAFamily(unittest.TestCase):   # I-M3
    def test_siblings_from_one_seed_share_the_seed_id(self):
        first = {**GOOD, "id": "S02-seeded-b01-0001", "source": "seeded:asdiv-nluds-0001"}
        second = {**GOOD, "id": "S01-seeded-b02-0007", "slice": "S01", "source": "seeded:asdiv-nluds-0001",
                  "problem": "A baker had 19 muffins and made 36 of them. How many muffins are left?"}
        families = {validate(item)[0]["family"] for item in (first, second)}
        self.assertEqual(families, {"asdiv-nluds-0001"})

    def test_other_items_are_their_own_family(self):
        self.assertEqual(validate(GOOD)[0]["family"], GOOD["id"])

    def test_a_rewording_keeps_the_seed_family(self):
        original = {**GOOD, "id": "S02-seeded-b01-0001", "source": "seeded:asdiv-nluds-0001"}
        record = {"id": "S02-seeded-b01-0001~w1", "of": original["id"],
                  "problem": "There were 36 muffins in the shop and 19 got sold. How many remain?",
                  "follow_up_questions": ["The baker adds 8 more. How many are there now?"]}
        reworded, error = expand(record, {original["id"]: original})
        self.assertIsNone(error)
        self.assertEqual(validate(reworded)[0]["family"], "asdiv-nluds-0001")


class SvampRangeFilterReadsTheText(unittest.TestCase):   # I-L1
    def test_written_number_above_99_is_dropped(self):
        record = {"id": "svamp-chal-211", "set": "svamp", "equation": "( 78.0 - 31.0 )", "dataset_answer": 47.0,
                  "problem": "Baker made 170 cakes. He sold 78 of them and bought 31 new cakes. "
                             "How many more cakes did baker sell than those he bought?"}
        self.assertEqual(range_check(record), "number above 99 in the text")
        record["problem"] = record["problem"].replace("170", "90")
        self.assertIsNone(range_check(record))


class PossessiveNamesAreNormalised(unittest.TestCase):   # I-L2
    def test_possessive_names_match(self):
        self.assertEqual(normalise("Mia's dog ate 5 bones.", {"Mia"}), ["NAME's", "dog", "ate", "N", "bones"])
        self.assertEqual(normalise("Mia's dog ate 5 bones.", {"Mia"}), normalise("Tom's dog ate 7 bones.", {"Tom"}))
        self.assertEqual(normalise("Mia and Tom", {"Mia", "Tom"}), ["NAME", "and", "NAME"])


class FollowUpsWriteTheirOwnAnswer(unittest.TestCase):   # I-L5
    STATE = {"left": 17, "answer": 17}

    def test_follow_up_without_answer_is_refused(self):
        with self.assertRaisesRegex(NotAllowed, "no 'answer'"):
            execute("def solve(state):\n    now = state['left'] + 8\n    return {**state, 'left': now}", self.STATE)

    def test_state_after_answer_is_refused(self):
        with self.assertRaisesRegex(NotAllowed, "before 'answer'"):
            execute("def solve(state):\n    now = state['left'] + 8\n    return {'answer': now, **state}", self.STATE)

    def test_state_before_answer_is_fine(self):
        execution = execute("def solve(state):\n    now = state['left'] + 8\n    return {**state, 'answer': now}", self.STATE)
        self.assertEqual(execution.result["answer"], 25)

    def test_recursion_error_becomes_not_allowed(self):
        source = "def solve():\n    return {'answer': " + " + ".join(["0"] * 20000) + "}"
        with self.assertRaises(NotAllowed):
            execute(source)


class ValidatorCases(unittest.TestCase):
    def test_every_broken_item_is_caught(self):
        for name, item, expected in validator_cases.CASES:
            with self.subTest(name):
                validated, reason = validate(item)
                if expected is None:
                    self.assertIsNone(reason)
                else:
                    self.assertIn(expected, reason or "")


if __name__ == "__main__":
    unittest.main()
