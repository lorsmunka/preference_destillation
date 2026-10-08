"""The answer-only paraphrase test (decided 2026-10-08): test rewordings and their copies get no
labels, threads or teacher data; they are scored on their final answers only."""

import unittest

from library.domain.math_tutor.items import answer_only_test


class AnswerOnlyTest(unittest.TestCase):
    def test_test_rewordings_and_their_copies_are_answer_only(self):
        self.assertTrue(answer_only_test({"id": "S01-mem-b01-0001~w2", "split": "test"}))
        self.assertTrue(answer_only_test({"id": "S01-mem-b01-0001~w2~n1", "split": "test"}))

    def test_originals_copies_and_train_rewordings_keep_full_data(self):
        self.assertFalse(answer_only_test({"id": "S01-mem-b01-0001", "split": "test"}))
        self.assertFalse(answer_only_test({"id": "S01-mem-b01-0001~n1", "split": "test"}))
        self.assertFalse(answer_only_test({"id": "S01-mem-b01-0001~w2", "split": "train"}))
        self.assertFalse(answer_only_test({"id": "S01-mem-b01-0001~w2", "split": "heldout"}))


if __name__ == "__main__":
    unittest.main()
