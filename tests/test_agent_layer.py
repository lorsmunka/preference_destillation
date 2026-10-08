"""The math-tutor agent-layer fixes from the code review (A-H1 ... A-L5, D-L6), one test class
per fix, plus a fake-model run of the tutor harness. CPU only, no model weights; the one test
that reads the real sample labels is skipped when the data files are missing.

    python -m unittest discover -s tests -p "test_agent_layer.py"
"""

import json
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from library.agent.question_templates import Focus, Message, Template
from library.domain.math_tutor import annotations, threads
from library.domain.math_tutor.agentic import MathTutorAgenticDomain, outside_vocabulary_report, vocabulary_texts
from library.domain.math_tutor.annotations import turn_executions
from library.domain.math_tutor.checks import arithmetic_true, check_against_trace, check_explanation, check_lines, parse
from library.domain.math_tutor.conversation import Conversation, labelled_turns, solve_turns, thread_turns, usable
from library.domain.math_tutor.harness import TemplatePupil, TutorSession, run_sessions
from library.domain.math_tutor.items import validate
from library.domain.math_tutor.labels import (by_position, computed_values, gold_calculations, number_mapping,
                                              solve_context, transfer, transfer_turn)
from library.domain.math_tutor.questions import generator as question_generator, solution_foci, what_if_foci

MATH_TUTOR = Path(__file__).resolve().parents[1] / "library" / "domain" / "math_tutor"


def make_item(item_id: str, slice_name: str, problem: str, solve: str, follow_ups=(), renumbered_from=None) -> dict:
    item = {"id": item_id, "slice": slice_name, "unknown": "result", "grade": 3, "distractor": "none",
            "context": "test", "source": "hand", "problem": problem, "solve": solve,
            "follow_ups": [{"question": question, "solve": code} for question, code in follow_ups]}
    validated, reason = validate(item)
    assert reason is None, f"{item_id}: {reason}"
    validated["split"] = "train"
    if renumbered_from:
        validated["renumbered_from"] = renumbered_from
    return validated


def block(*lines: str) -> str:
    return "\n".join(["<solution>", *lines, "</solution>"])


RIBBON = make_item("T-ribbon", "S07", "A ribbon is 50 cm long. Kim cuts it into pieces 6 cm long. How many cm of ribbon are left over?",
                   "def solve():\n    length = 50\n    piece = 6\n    left = length % piece\n"
                   "    return {'length': length, 'piece': piece, 'answer': left}")
APPLES = make_item("T-apples", "S05", "Mia has 4 bags with 6 apples in each bag. How many apples does she have?",
                   "def solve():\n    bags = 4\n    per_bag = 6\n    apples = bags * per_bag\n    return {'apples': apples, 'answer': apples}",
                   [("She buys 6 more apples. How many apples does she have now?",
                     "def solve(state):\n    now = state['apples'] + 6\n    return {**state, 'apples': now, 'answer': now}")])
APPLES_LABELS = {0: block("bags = 4", "apples in each bag = 6", "apples = 4 * 6 = 24", "Answer: 24"),
                 1: block("apples she had = 24", "apples she buys = 6", "apples now = 24 + 6 = 30", "Answer: 30")}


class FactsLabels:
    """Puts verified facts-first labels in place for a test (annotations reads them lazily)."""

    def __init__(self, labels_by_item: dict):
        self.labels = {(item_id, turn): text for item_id, by_turn in labels_by_item.items() for turn, text in by_turn.items()}

    def __enter__(self):
        annotations._FACTS_LABELS.clear()
        annotations._FACTS_LABELS.update(self.labels)

    def __exit__(self, *exception):
        annotations._FACTS_LABELS.clear()


class RemainderOfLines(unittest.TestCase):   # A-H2
    def test_the_left_over_form_passes_the_line_check(self):
        self.assertIsNone(check_lines(block("marbles = 23", "bag size = 5", "marbles left over = remainder of 23 / 5 = 3", "Answer: 3")))

    def test_a_wrong_left_over_fails(self):
        problem = check_lines(block("marbles = 23", "bag size = 5", "marbles left over = remainder of 23 / 5 = 2", "Answer: 2"))
        self.assertTrue(problem.startswith("arithmetic does not hold"))

    def test_its_value_is_what_is_left_over(self):
        line = parse("left = remainder of 23 / 5 = 3")[0]
        self.assertEqual((line.kind, line.number), ("calculation", 3))

    def test_a_percent_step_can_get_a_facts_first_label(self):
        runs = turn_executions(RIBBON)
        gold = gold_calculations(RIBBON, 0, runs)
        self.assertEqual(gold, ["remainder of 50 / 6 = 2"])
        label = block("ribbon length = 50", "length of a piece = 6", f"ribbon left over = {gold[0]}", "Answer: 2")
        context = Conversation().solve_context(RIBBON["problem"])
        self.assertIsNone(check_against_trace(label, computed_values(runs, 0), 2, context))


class WholeNumberArithmeticOnly(unittest.TestCase):   # A-L4 (arithmetic_true)
    def calculation(self, text: str):
        return parse(f"x = {text}")[0]

    def test_floats_powers_xor_and_percent_fail(self):
        for text in ("1.5 * 4 = 6", "2 ** 3 = 8", "9 ^ 3 = 10", "23 % 5 = 3"):
            self.assertFalse(arithmetic_true(self.calculation(text)), text)

    def test_exact_division_and_the_remainder_forms_pass(self):
        for text in ("63 / 7 = 9", "(16 + 2) / 3 = 6", "50 / 6 = 8 remainder 2", "16 / 3 = 5 remainder 1, rounded up = 6"):
            self.assertTrue(arithmetic_true(self.calculation(text)), text)

    def test_inexact_division_fails(self):
        self.assertFalse(arithmetic_true(self.calculation("64 / 7 = 9")))


class ConcessionCheck(unittest.TestCase):   # A-L4 (CONCESSION)
    def test_stating_the_result_is_not_a_concession(self):
        self.assertIsNone(check_explanation("If we split 63 into groups of 7 we should have 9 groups.", "pushback"))

    def test_taking_the_solution_back_still_fails(self):
        for text in ("We should have added the 5 apples.", "I should have multiplied.", "Sorry, I was wrong."):
            self.assertIsNotNone(check_explanation(text, "why"), text)


class FactsMustBeTheNumbersUsed(unittest.TestCase):   # A-M3
    def test_a_block_without_a_calculation_fails(self):
        context = "<task>Tom has 4 bags of 6 apples.</task>\n<solution>\ntotal apples = 4 * 6 = 24\nAnswer: 24\n</solution>\n<message>How many?</message>"
        self.assertEqual(check_against_trace(block("apples he has = 24", "Answer: 24"), [24], 24, context), "no calculation")

    def test_a_fact_the_calculation_does_not_use_fails(self):
        context = "<message>A rabbit eats 9 carrots and 5 apple slices each day. How many snacks?</message>"
        label = block("carrots each day = 9", "apple slices each day = 9", "snacks = 5 + 9 = 14", "Answer: 14")
        self.assertTrue(check_against_trace(label, [14], 14, context).startswith("calculation uses 5"))

    def test_implied_constants_and_earlier_results_are_allowed(self):
        context = "<message>The campsite fee is 20 dollars and Gabriela pays half of it. Then she pays 3 more. How much?</message>"
        label = block("campsite fee = 20", "more money = 3", "half of the fee = 20 / 2 = 10", "paid = 10 + 3 = 13", "Answer: 13")
        self.assertIsNone(check_against_trace(label, [10, 13], 13, context))

    def test_the_sample_labels_still_pass(self):
        sample_path = MATH_TUTOR / "questions" / "sample_labels.jsonl"
        item_paths = [MATH_TUTOR / "items" / "items.jsonl", MATH_TUTOR / "items" / "renumbered.jsonl"]
        if not sample_path.exists() or not all(path.exists() for path in item_paths):
            self.skipTest("sample labels or items not on disk")
        labels = {}
        for line in open(sample_path, encoding="utf-8"):
            record = json.loads(line)
            labels[(record["item_id"], record["turn"])] = record["block"]
        wanted, items = {item_id for item_id, _ in labels}, {}
        for path in item_paths:
            for line in open(path, encoding="utf-8"):
                item_id = line[len('{"id": "'):line.index('"', len('{"id": "'))]
                if item_id in wanted:
                    items[item_id] = json.loads(line)
        failing = []
        for (item_id, turn), text in labels.items():
            if "~n" in item_id or item_id not in items:   # renumbered copies are re-drawn by every rebuild
                continue
            item = items[item_id]
            context = solve_context(item, turn, labels)
            if context is None:   # an earlier turn is not in the sample
                continue
            runs = turn_executions(item)
            if check_against_trace(text, computed_values(runs, turn), item["answers"][turn], context):
                failing.append((item_id, turn))
        self.assertEqual(failing, [("S02-mem-b04-0005", 3)])   # uses the earlier answer 32 without listing it


class NumberWordsInTransfer(unittest.TestCase):   # A-M1
    def test_the_mapping_reads_number_words(self):
        self.assertEqual(number_mapping("Fourteen frogs sit on 3 logs.", "Sixteen frogs sit on 5 logs."), {14: 16, 3: 5})

    def test_a_fact_from_a_number_word_gets_the_copys_number(self):
        original = make_item("T-frogs", "S01", "Fourteen frogs sit on a log. 6 more frogs hop on. How many frogs are on the log now?",
                             "def solve():\n    frogs = 14\n    more = 6\n    total = frogs + more\n    return {'frogs': total, 'answer': total}")
        copy = make_item("T-frogs~n1", "S01", "Sixteen frogs sit on a log. 9 more frogs hop on. How many frogs are on the log now?",
                         "def solve():\n    frogs = 16\n    more = 9\n    total = frogs + more\n    return {'frogs': total, 'answer': total}",
                         renumbered_from="T-frogs")
        label = block("frogs on the log = 14", "frogs that hop on = 6", "frogs now = 14 + 6 = 20", "Answer: 20")
        by_text = number_mapping(original["problem"], copy["problem"])
        new_block, problem = transfer_turn(original, copy, 0, label, turn_executions(original), turn_executions(copy), by_text)
        self.assertIsNone(problem)
        self.assertIn("frogs on the log = 16", new_block)
        context = Conversation().solve_context(copy["problem"])
        self.assertIsNone(check_against_trace(new_block, computed_values(turn_executions(copy), 0), 25, context))


class PerTurnMapping(unittest.TestCase):   # A-M4
    def test_an_ambiguous_number_is_marked(self):
        self.assertEqual(by_position([5, 5, 3], [7, 9, 4]), {5: None, 3: 4})

    def test_a_computed_value_equal_to_a_text_number_no_longer_voids_the_copy(self):
        original = make_item("T-pens", "S10", "Ana has 4 boxes with 3 pens in each box. She gets 12 more pens. How many pens does she have now?",
                             "def solve():\n    boxes = 4\n    per_box = 3\n    in_boxes = boxes * per_box\n    total = in_boxes + 12\n"
                             "    return {'pens': total, 'answer': total}")
        copy = make_item("T-pens~n1", "S10", "Ana has 5 boxes with 4 pens in each box. She gets 14 more pens. How many pens does she have now?",
                         "def solve():\n    boxes = 5\n    per_box = 4\n    in_boxes = boxes * per_box\n    total = in_boxes + 14\n"
                         "    return {'pens': total, 'answer': total}", renumbered_from="T-pens")
        label = block("boxes = 4", "pens in each box = 3", "pens she gets = 12", "pens in the boxes = 4 * 3 = 12",
                      "pens now = 12 + 12 = 24", "Answer: 24")
        by_text = number_mapping(original["problem"], copy["problem"])
        new_block, problem = transfer_turn(original, copy, 0, label, turn_executions(original), turn_executions(copy), by_text)
        self.assertIsNone(problem)
        self.assertEqual([line.number for line in parse(new_block) if line.kind == "fact"], [5, 4, 14])
        context = Conversation().solve_context(copy["problem"])
        self.assertIsNone(check_against_trace(new_block, computed_values(turn_executions(copy), 0), 34, context))


class SplitOrJoinedLabelsAndStepWords(unittest.TestCase):   # A-L5
    def test_a_split_label_says_why_it_cannot_transfer(self):
        label = block("bags = 4", "apples in each bag = 6", "apples in two bags = 2 * 6 = 12", "apples = 12 * 2 = 24", "Answer: 24")
        new_block, problem = transfer(label, ["5 * 7 = 35"], 35, fact_numbers=[{4: 5, 6: 7}], label_numbers=[{4: 5, 6: 7}])
        self.assertEqual(new_block, "")
        self.assertTrue(problem.startswith("split or joined steps"))

    def step_slots(self, label: str) -> dict:
        item = make_item("T-red-green", "S03", "Tom has 5 red apples and 5 green apples. How many apples does he have?",
                         "def solve():\n    red = 5\n    green = 5\n    total = red + green\n    return {'answer': total}")
        steps = [focus for focus in solution_foci(item, 0, label, random.Random(0)) if focus.type == "step"]
        return steps[0].slots

    def test_equal_numbers_get_their_two_labels(self):
        slots = self.step_slots(block("red apples = 5", "green apples = 5", "apples = 5 + 5 = 10", "Answer: 10"))
        self.assertEqual((slots["a_words"], slots["b_words"]), ("red apples", "green apples"))

    def test_one_label_for_both_numbers_gives_no_words(self):
        slots = self.step_slots(block("apples of each colour = 5", "apples = 5 + 5 = 10", "Answer: 10"))
        self.assertNotIn("a_words", slots)


class WhatIfCandidates(unittest.TestCase):   # A-M2
    def test_a_number_two_facts_show_is_not_offered(self):
        item = make_item("T-rabbit", "S10", "A rabbit eats 5 carrots and 5 apple slices each day. How many snacks does it eat in 3 days?",
                         "def solve():\n    carrots = 5\n    slices = 5\n    days = 3\n    per_day = carrots + slices\n"
                         "    total = per_day * days\n    return {'answer': total}")
        label = block("carrots each day = 5", "apple slices each day = 5", "days = 3", "snacks each day = 5 + 5 = 10",
                      "snacks in 3 days = 10 * 3 = 30", "Answer: 30")
        foci = what_if_foci(item, 0, label, random.Random(0), limit=5)
        self.assertEqual({focus.slots["old"] for focus in foci}, {3})
        gold = foci[0].data["gold"]
        new = foci[0].slots["new"]
        self.assertIn(f"days = {new}", gold)
        self.assertIn("snacks each day = 5 + 5 = 10", gold)

    def test_a_number_an_earlier_turn_used_is_not_offered(self):
        self.assertEqual(what_if_foci(APPLES, 1, APPLES_LABELS[1], random.Random(0), limit=5), [])


class ExplanationCutOff(unittest.TestCase):   # A-L2
    MESSAGE = {"question": "Why did you divide?", "kind": "why", "route": "explanation"}

    def test_no_closing_tag_fails_the_thread_check(self):
        entry = threads.reply_entry(self.MESSAGE, "<explanation>Because each bag holds 7 bottles, so we count how many")
        self.assertEqual(entry["check"], "cut off: no </explanation>")

    def test_a_closed_explanation_passes(self):
        entry = threads.reply_entry(self.MESSAGE, "<explanation>Each bag holds 7 bottles, so 63 / 7 = 9 bags.</explanation>")
        self.assertIsNone(entry["check"])
        self.assertEqual(entry["explanation"], "Each bag holds 7 bottles, so 63 / 7 = 9 bags.")


def stub_message(kind: str, route: str) -> Message:
    template = Template(kind=kind, route=route, focus="none", text=f"a {kind} message")
    return Message(template.text, kind, route, Focus("none", key="none"), template)


class StubGenerator:
    """Draws the given kinds in order and records the depth of each draw."""

    def __init__(self, kinds):
        self.kinds, self.depths = list(kinds), []

    def draw(self, candidates, depth, random_generator, kind_weights=None):
        self.depths.append(depth)
        kind = self.kinds[(len(self.depths) - 1) % len(self.kinds)]
        return stub_message(kind, {"off_topic": "out_of_scope", "what_if": "solution"}.get(kind, "explanation"))


class DepthAfterOffTopic(unittest.TestCase):   # A-L3
    def test_the_message_after_an_opening_off_topic_one_is_drawn_at_depth_zero(self):
        generator = StubGenerator(["off_topic", "why", "follow"])
        messages = threads.plan_messages(APPLES, 1, APPLES_LABELS[1], 3, generator, random.Random(0))
        self.assertEqual([message["kind"] for message in messages], ["off_topic", "why", "follow"])
        self.assertEqual(generator.depths, [0, 0, 1])


class FlakyHandler:
    """Replies with a closed explanation, except: the first plan's first reply is cut off (if
    cut_off_first), and the call number `fail_on` raises (an interrupted pass)."""

    def __init__(self, fail_on=None, cut_off_first=True):
        self.calls, self.fail_on, self.cut_off_first = 0, fail_on, cut_off_first

    def generate_texts(self, prompts, **settings):
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("interrupted")
        replies = ["<explanation>Because 4 bags of 6 make 24.</explanation>"] * len(prompts)
        if self.calls == 1 and self.cut_off_first:
            replies[0] = "<explanation>Because 4 bags of"
        return replies


class ThreadPassAppendsAsItGoes(unittest.TestCase):   # A-M5
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.items = [dict(APPLES, id=f"T-apples-{index}") for index in range(4)]
        self.labels = {(item["id"], turn): text for item in self.items for turn, text in APPLES_LABELS.items()}
        self.saved_depths = threads.DEPTHS
        threads.DEPTHS = [2]

    def tearDown(self):
        threads.DEPTHS = self.saved_depths
        shutil.rmtree(self.folder, ignore_errors=True)

    def run_pass(self, handler):
        return threads.run_threads(handler, MathTutorAgenticDomain("facts"), items=self.items, generator=StubGenerator(["why"]),
                                   labels=self.labels, threads_path=self.folder / "threads.jsonl")

    def written(self):
        path = self.folder / "threads.jsonl"
        if not path.exists():
            return []
        with open(path, encoding="utf-8") as lines:
            return [json.loads(line) for line in lines]

    def test_a_stopped_thread_is_saved_before_an_interruption(self):
        with self.assertRaises(RuntimeError):
            self.run_pass(FlakyHandler(fail_on=2))
        saved = self.written()
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["thread"][0]["check"], "cut off: no </explanation>")

    def test_the_next_run_finishes_the_rest_without_repeating(self):
        with self.assertRaises(RuntimeError):
            self.run_pass(FlakyHandler(fail_on=2))
        self.assertEqual(self.run_pass(FlakyHandler(cut_off_first=False)), 3)
        saved = self.written()
        self.assertEqual(sorted(record["item_id"] for record in saved), sorted(item["id"] for item in self.items))
        self.assertTrue(all(len(record["thread"]) == 2 for record in saved[1:]))


class PartlyLabelledItems(unittest.TestCase):   # A-L1
    def test_solve_records_up_to_the_first_unlabelled_turn(self):
        with FactsLabels({APPLES["id"]: {0: APPLES_LABELS[0]}}):
            self.assertEqual(labelled_turns(APPLES, "facts"), 1)
            self.assertTrue(usable(APPLES, "facts"))
            records = list(solve_turns(APPLES, "facts"))
        self.assertEqual([record["turn"] for record in records], [0])
        self.assertIn("apples = 4 * 6 = 24", records[0]["target"])

    def test_no_label_no_records(self):
        with FactsLabels({"someone-else": {0: APPLES_LABELS[0]}}):
            self.assertFalse(usable(APPLES, "facts"))
            self.assertEqual(list(solve_turns(APPLES, "facts")), [])

    def test_a_thread_on_an_unlabelled_turn_gives_no_records(self):   # D-L6 / thread_turns
        thread_record = {"item_id": APPLES["id"], "turn": 1, "thread": []}
        with FactsLabels({APPLES["id"]: {0: APPLES_LABELS[0]}}):
            self.assertEqual(list(thread_turns(APPLES, "facts", thread_record)), [])


EXPLANATION = "Yes, because 4 bags with 6 apples in each bag make 24."


class OutputVocabulary(unittest.TestCase):   # A-H1
    def test_explanation_targets_are_in_the_vocabulary_texts(self):
        entry = {"question": "Why did you multiply?", "kind": "why", "route": "explanation", "routed": "explanation",
                 "explanation": EXPLANATION, "check": None}
        thread_record = {"item_id": APPLES["id"], "turn": 0, "thread": [entry]}
        with FactsLabels({APPLES["id"]: APPLES_LABELS}):
            with_threads = vocabulary_texts([APPLES], [thread_record])
            without_threads = vocabulary_texts([APPLES], [])
        self.assertIn(f"<explanation>{EXPLANATION}</explanation>", with_threads)
        self.assertEqual(with_threads.count(f"<explanation>{EXPLANATION}</explanation>"), 1)
        self.assertNotIn(f"<explanation>{EXPLANATION}</explanation>", without_threads)

    def test_the_report_counts_dropped_targets_per_turn_type(self):
        class SpaceTokenizer:
            def tokenize(self, text):
                return text.split()

        records = [{"turn_type": "solve", "target": "Answer: 24"}, {"turn_type": "explain", "target": "Yes because"},
                   {"turn_type": "explain", "target": "because"}, {"turn_type": "route", "target": "<solution>"}]
        report = outside_vocabulary_report(records, SpaceTokenizer(), {"Answer:", "24", "because", "<solution>"})
        self.assertEqual(report["explain"], {"targets": 2, "dropped": 1, "commonest_outside_tokens": {"Yes": 1}})
        self.assertEqual(report["solve"]["dropped"], 0)


class AdversarialTutor:
    """A fake model that misbehaves on purpose: untagged routes, tasks routed out of scope or to
    an explanation, solutions that fail the check, explanations that concede or are cut off."""

    def __init__(self, seed: int):
        self.random_generator = random.Random(seed)

    def generate(self, requests):
        replies = []
        for request in requests:
            roll = self.random_generator.random()
            if request.kind == "route":
                if not request.prefix and roll < 0.15:
                    replies.append("Sure! Let me think about that.")
                elif roll < 0.3:
                    replies.append(("" if request.prefix else "<") + "out_of_scope>")
                elif roll < 0.6:
                    replies.append(("" if request.prefix else "<") + "explanation>Because")
                else:
                    replies.append(("" if request.prefix else "<") + "solution>\nx = 1")
            elif request.kind == "solution":
                replies.append(self.random_generator.choice(["bags = 4\napples = 4 * 6 = 24\nAnswer: 24\n</solution>",
                                                             "a = 2\nAnswer: 2\n</solution>", "1. bad\n</solution>"]))
            else:
                replies.append(self.random_generator.choice(["I missed it, sorry.</explanation>", "Because 4 bags of",
                                                             "Because 2 + 2 = 4.</explanation>"]))
        return replies


class HarnessWithAFakeModel(unittest.TestCase):
    def test_the_state_machine_holds_up(self):
        questions = question_generator()
        items = [APPLES, RIBBON]
        depths_of_follow_messages = []
        sessions = []
        for seed in range(30):
            pupil = TemplatePupil(items[seed % len(items)], questions, question_chance=1.0, seed=seed)
            original_message = pupil.message

            def recording_message(event, block_text, depth=None, original_message=original_message):
                message = original_message(event, block_text, depth)
                if message is not None and message.kind == "follow":
                    depths_of_follow_messages.append(depth)
                return message

            pupil.message = recording_message
            sessions.append(TutorSession(pupil))
        run_sessions(AdversarialTutor(seed=1), sessions)
        for session in sessions:
            self.assertTrue(session.done)
            tasks = [event for event in session.pupil.events if event["type"] == "task"]
            self.assertEqual(sum(turn["event"] == "task" for turn in session.transcript), len(tasks))
            history = session.conversation.history(with_thread=True)
            for turn in session.transcript:
                answered_by_model = any(other["message"] == turn["message"] and other.get("answered_by") != "harness"
                                        for other in session.transcript)
                if turn.get("answered_by") == "harness" and not answered_by_model:   # off-topic never enters the context
                    self.assertNotIn(f"<task>{turn['message']}</task>", history)
                    self.assertNotIn(f"<question>{turn['message']}</question>", history)
                if turn.get("route") == "explanation" and "output" in turn and turn.get("check") is None:
                    self.assertTrue(turn["output"].endswith("</explanation>"))   # a cut-off one never passes
        self.assertTrue(depths_of_follow_messages)
        self.assertTrue(all(depth > 0 for depth in depths_of_follow_messages))   # only after an explanation


if __name__ == "__main__":
    unittest.main()
