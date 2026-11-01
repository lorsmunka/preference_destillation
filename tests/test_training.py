"""The training / model fixes from the code review (T-H1/X-H1, T-H2, T-M1, T-M2, T-M3,
T-M4, T-M5, T-L1, T-L4, X-C1, X-M2), one test class per area. CPU only, tiny models,
synthetic batches in a temporary folder (nothing is read from or written to runs/ or
batches/). Needs the Gemma tokenizer in the local Hugging Face cache (no network).

    python -m unittest discover -s tests -p "test_training.py"
"""

import contextlib
import json
import math
import os
import shutil
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

import library.model.student_vocabularies as student_vocabularies_module
import library.training.trainer as trainer_module
import library.training.training_runner as training_runner_module
from library import TrainingRun
from library.domain import REDDIT_SENTIMENT, RedditSentimentDomain
from library.model.custom_student import CustomStudent
from library.model.decoding import SimpleDecoder
from library.model.student import StudentModel
from library.model.transformer import IncrementalDecoder, Transformer
from library.shared import DEFAULT_TEACHER_MODEL
from library.shared.batch_files import vocabulary_fingerprint, write_batch
from library.shared.vocabulary import build_vocabulary
from library.training.batch_handler import BatchHandler, select_records, split_batch_files
from library.training.evaluator import Evaluator, majority_baseline
from library.training.example_encoder import ExampleEncoder
from library.training.losses import DistillationLoss
from library.training.schedules import Schedules

torch.set_num_threads(2)


class CachedAutoTokenizer:
    """Loads the Gemma tokenizer once for the whole test run instead of once per model."""
    tokenizers = {}

    @classmethod
    def from_pretrained(cls, name):
        if name not in cls.tokenizers:
            cls.tokenizers[name] = AutoTokenizer.from_pretrained(name)
        return cls.tokenizers[name]


student_vocabularies_module.AutoTokenizer = CachedAutoTokenizer
TOKENIZER = CachedAutoTokenizer.from_pretrained(DEFAULT_TEACHER_MODEL)
VOCABULARY = build_vocabulary(TOKENIZER, REDDIT_SENTIMENT)
TOKEN_LIST = VOCABULARY["token_list"]
RESPONSE = '{"tone": "neutral", "sentiment": "positive", "safety": "safe", "toxicity": "respectful"}'
SENTENCES = ["the movie was fine i guess", "thanks a lot for the help", "this is the worst take ever",
             "lovely weather today", "what a pointless thread", "great answer, very clear",
             "nobody asked for this", "i really enjoyed reading it"]


def synthetic_record(seed: int, split=None, step_count: int = 6) -> dict:
    """One teacher example: random logits over the reduced vocabulary, the argmax as the token."""
    generator = np.random.default_rng(seed)
    steps = []
    for _ in range(step_count):
        logits = (generator.standard_normal(len(TOKEN_LIST)) * 3).astype(np.float32)
        index = int(np.argmax(logits))
        steps.append({"token": TOKEN_LIST[index], "logits": logits, "predicted_token_index": index})
    return {"sentence": SENTENCES[seed % len(SENTENCES)], "model_response": RESPONSE, "steps": steps, "split": split}


def write_batches(directory: Path, file_count: int, records_per_file: int = 4, splits=None,
                  vocabulary_sha1: str = None) -> None:
    """batch_1..batch_N.npz; `splits` gives each file's per-record labels (default: none)."""
    directory.mkdir(parents=True, exist_ok=True)
    for file_index in range(file_count):
        labels = splits[file_index] if splits is not None else [None] * records_per_file
        records = [synthetic_record(file_index * 100 + record_index, split)
                   for record_index, split in enumerate(labels)]
        write_batch(directory / f"batch_{file_index + 1}.npz", records, vocabulary_sha1=vocabulary_sha1)


def tiny_input_vocabulary(with_unknown_row: bool = True) -> dict:
    """An input vocabulary covering the synthetic sentences and the output tokens."""
    token_ids = set(VOCABULARY["token_to_id"][token] for token in TOKEN_LIST)
    for sentence in SENTENCES:
        token_ids.update(TOKENIZER.encode(sentence + "\n\n", add_special_tokens=False))
    sorted_ids = sorted(token_ids)
    vocabulary = {"compact_vocab_size": len(sorted_ids),
                  "gemma_id_to_compact_id": {str(gemma_id): compact_id for compact_id, gemma_id in enumerate(sorted_ids)}}
    if with_unknown_row:
        vocabulary["compact_vocab_size"] += 1
        vocabulary["unknown_compact_id"] = len(sorted_ids)
    return vocabulary


def tiny_model(input_vocabulary=None, domain=REDDIT_SENTIMENT, seed: int = 0) -> Transformer:
    torch.manual_seed(seed)
    return Transformer(domain=domain, hidden_dim=8, num_layers=1, num_heads=1, dropout=0.0,
                       input_vocabulary=input_vocabulary or tiny_input_vocabulary())


class LooksAheadStudent(torch.nn.Module):
    """Breaks the custom-model contract: every position also sees the mean of the whole input."""

    def __init__(self, input_vocab_size, output_vocab_size):
        super().__init__()
        self.embedding = torch.nn.Embedding(input_vocab_size, 4)
        self.output = torch.nn.Linear(4, output_vocab_size)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        return self.output(embedded + embedded.mean(dim=1, keepdim=True))


class WrongShapeStudent(LooksAheadStudent):
    def forward(self, input_ids):
        return self.output(self.embedding(input_ids))[:, -1]   # the last position only


class TinyLstmStudent(torch.nn.Module):
    """Keeps the custom-model contract: an LSTM reads left to right, so it never looks ahead."""

    def __init__(self, input_vocab_size, output_vocab_size, hidden=8, layers=1):
        super().__init__()
        self.embedding = torch.nn.Embedding(input_vocab_size, hidden)
        self.lstm = torch.nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True)
        self.output = torch.nn.Linear(hidden, output_vocab_size)

    def forward(self, input_ids):
        hidden_states, _ = self.lstm(self.embedding(input_ids))
        return self.output(hidden_states)


class ScriptedExit:
    """Requests a stop at the Nth check (the trainer checks once per training batch)."""

    def __init__(self, stop_at_check: int = None):
        self.checks = 0
        self.stop_at_check = stop_at_check

    def check_exit(self) -> bool:
        self.checks += 1
        return self.stop_at_check is not None and self.checks >= self.stop_at_check

    def stop(self) -> None:
        pass


# ── T-H1 / X-H1 and the split-label contract ─────────────────────────

class SplitTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.folder, ignore_errors=True)

    def test_test_set_is_a_fixed_tail_whatever_the_training_cap(self):
        write_batches(self.folder, 20, records_per_file=1)
        test_sets = []
        for max_training_examples in (None, 4, 16, 40, 1000):
            training_files, test_files = split_batch_files(
                self.folder, 0.8, max_training_examples, batch_size=4, max_test_examples=8)
            test_sets.append([path.name for path in test_files])
            expected_training_count = 16 if max_training_examples is None else min(16, max_training_examples // 4)
            self.assertEqual(len(training_files), expected_training_count)   # only training is capped
            self.assertFalse(set(training_files) & set(test_files))
        self.assertTrue(all(test_set == ["batch_19.npz", "batch_20.npz"] for test_set in test_sets))
        _, whole_region = split_batch_files(self.folder, 0.8, 8, batch_size=4, max_test_examples=None)
        self.assertEqual([path.name for path in whole_region], [f"batch_{index}.npz" for index in range(17, 21)])

    def test_per_record_split_labels_decide_instead_of_the_ratio(self):
        labels = [["train"] * 4, ["train"] * 4, ["train", "train", "test", "test"], ["test"] * 4, ["test"] * 4]
        write_batches(self.folder, 5, splits=labels)
        training_files, test_files = split_batch_files(self.folder, 0.2, None, batch_size=4, max_test_examples=None)
        self.assertEqual([path.name for path in training_files], ["batch_1.npz", "batch_2.npz", "batch_3.npz"])
        self.assertEqual([path.name for path in test_files], ["batch_3.npz", "batch_4.npz", "batch_5.npz"])

        handler = BatchHandler(self.folder, 0.2, None, batch_size=4, max_test_examples=None)
        boundary_for_training = handler.get_training_batch(epoch=0, position=2)
        boundary_for_test = handler.get_test_batch(0)
        self.assertEqual([record["split"] for record in boundary_for_training], ["train", "train"])
        self.assertEqual([record["split"] for record in boundary_for_test], ["test", "test"])

        mixed = [{"split": "train"}, {"split": None}, {"split": "test"}]
        self.assertEqual(select_records(mixed, "train"), [{"split": "train"}])   # the unlabelled one is ambiguous
        unlabelled = [{"split": None}, {"split": None}]
        self.assertEqual(select_records(unlabelled, "test"), unlabelled)

    def test_training_groups_select_training_records_and_leave_the_test_set_alone(self):
        groups = [["original"] * 4, ["original", "original", "new", "new"], ["copy"] * 4, ["copy"] * 4]
        self.folder.mkdir(parents=True, exist_ok=True)
        for file_index, file_groups in enumerate(groups):
            split = "test" if file_index == 3 else "train"
            records = [{**synthetic_record(file_index * 100 + record_index, split), "group": group}
                       for record_index, group in enumerate(file_groups)]
            write_batch(self.folder / f"batch_{file_index + 1}.npz", records)
        everything, test_files = split_batch_files(self.folder, 0.5, None, batch_size=4, max_test_examples=None)
        self.assertEqual([path.name for path in everything], ["batch_1.npz", "batch_2.npz", "batch_3.npz"])

        handler = BatchHandler(self.folder, 0.5, None, batch_size=4, max_test_examples=None,
                               training_groups=("original", "new"))
        self.assertEqual([path.name for path in handler.training_files], ["batch_1.npz", "batch_2.npz"])
        self.assertEqual([record["group"] for record in handler.get_training_batch(0, 1)],
                         ["original", "original", "new", "new"])
        only_copies = BatchHandler(self.folder, 0.5, None, batch_size=4, max_test_examples=None,
                                   training_groups=("copy",))
        self.assertEqual([path.name for path in only_copies.training_files], ["batch_3.npz"])
        self.assertEqual(only_copies.test_files, handler.test_files)
        self.assertEqual(handler.test_files, test_files)   # the test set never depends on the groups

        only_originals = BatchHandler(self.folder, 0.5, None, batch_size=4, max_test_examples=None,
                                      training_groups=("original",))
        self.assertEqual([record["group"] for record in only_originals.get_training_batch(0, 1)], ["original", "original"])

    def test_shuffling_is_off_by_default_and_fixed_per_seed_and_epoch(self):
        write_batches(self.folder, 10, records_per_file=1)
        in_file_order = BatchHandler(self.folder, 1.0, None, batch_size=1)
        self.assertEqual(in_file_order.training_order(0), list(range(10)))
        shuffled = BatchHandler(self.folder, 1.0, None, batch_size=1, shuffle_seed=3)
        again = BatchHandler(self.folder, 1.0, None, batch_size=1, shuffle_seed=3)
        self.assertEqual(sorted(shuffled.training_order(0)), list(range(10)))
        self.assertEqual(shuffled.training_order(1), again.training_order(1))   # a resumed epoch replays it
        self.assertNotEqual(shuffled.training_order(0), shuffled.training_order(1))
        self.assertNotEqual(shuffled.training_order(0), list(range(10)))

    def test_batches_from_another_vocabulary_are_refused(self):
        write_batches(self.folder / "other", 2, vocabulary_sha1=vocabulary_fingerprint(["a", "b"]))
        handler = BatchHandler(self.folder / "other", 0.5, None, batch_size=4, teacher_token_list=TOKEN_LIST)
        with self.assertRaises(ValueError):
            handler.get_training_batch(0, 0)

        write_batches(self.folder / "same", 2, vocabulary_sha1=vocabulary_fingerprint(TOKEN_LIST))
        write_batches(self.folder / "unknown", 2)   # batches made before the fingerprint existed
        for name in ("same", "unknown"):
            handler = BatchHandler(self.folder / name, 0.5, None, batch_size=4, teacher_token_list=TOKEN_LIST)
            self.assertEqual(len(handler.get_training_batch(0, 0)), 4)


# ── T-M4 / T-L1 ───────────────────────────────────────────────────────

def float64_divergence(divergence, student_logits, teacher_logits, student_temperature, teacher_temperature):
    p = F.softmax(teacher_logits.double() / teacher_temperature, -1)
    q = F.softmax(student_logits.double() / student_temperature, -1)
    if divergence == "forward_kl":
        return (p * (p.log() - q.log())).sum(-1)
    if divergence == "reverse_kl":
        return (q * (q.log() - p.log())).sum(-1)
    if divergence == "js":
        m = 0.5 * (p + q)
        return 0.5 * (p * (p.log() - m.log())).sum(-1) + 0.5 * (q * (q.log() - m.log())).sum(-1)
    return 0.5 * (p - q).abs().sum(-1)


class LossTests(unittest.TestCase):
    def test_reverse_kl_matches_float64_on_a_sharp_teacher(self):
        teacher = torch.tensor([[40.0, 0.0, -10.0, -30.0]])   # tail far below the old 1e-12 clamp
        student = torch.zeros(1, 4)
        divergence, _, _ = DistillationLoss("reverse_kl").compute(
            student, teacher, torch.tensor([0]), 1.0, 1.0, kl_ratio=1.0, reduction="sum")
        reference = float64_divergence("reverse_kl", student, teacher, 1.0, 1.0).sum().item()
        self.assertAlmostEqual(reference, 38.6137, places=3)
        self.assertLess(abs(divergence.item() - reference) / reference, 1e-5)

    def test_every_divergence_matches_float64(self):
        torch.manual_seed(0)
        student = torch.randn(5, 7)
        teacher = torch.randn(5, 7) * 3
        for divergence in ("forward_kl", "reverse_kl", "js", "tvd"):
            for student_temperature, teacher_temperature in ((1.0, 1.0), (2.0, 2.0), (1.0, 3.0)):
                value, _, _ = DistillationLoss(divergence).compute(
                    student, teacher, teacher.argmax(-1), student_temperature, teacher_temperature, 0.3, "sum")
                reference = (float64_divergence(divergence, student, teacher, student_temperature, teacher_temperature)
                             * student_temperature ** 2).sum().item()
                self.assertAlmostEqual(value.item(), reference, places=4, msg=f"{divergence} T={student_temperature}")

    def test_cross_entropy_is_taken_at_temperature_one(self):
        torch.manual_seed(1)
        student = torch.randn(6, 9)
        teacher = torch.randn(6, 9)
        targets = teacher.argmax(-1)
        divergence, cross_entropy, combined = DistillationLoss().compute(
            student, teacher, targets, student_temperature=2.0, teacher_temperature=2.0, kl_ratio=0.25, reduction="sum")
        plain = F.cross_entropy(student, targets, reduction="sum")
        tempered = F.cross_entropy(student / 2.0, targets, reduction="sum") * 4.0
        self.assertAlmostEqual(cross_entropy.item(), plain.item(), places=5)
        self.assertNotAlmostEqual(cross_entropy.item(), tempered.item(), places=2)
        self.assertAlmostEqual(combined.item(), (0.25 * divergence + 0.75 * cross_entropy).item(), places=4)


# ── T-L4, T-M3 (decoder), T-M5 ───────────────────────────────────────

class ModelTests(unittest.TestCase):
    def test_unknown_token_maps_to_the_unknown_row(self):
        vocabulary = tiny_input_vocabulary()
        model = tiny_model(vocabulary)
        unseen_token_id = max(int(gemma_id) for gemma_id in vocabulary["gemma_id_to_compact_id"]) + 1
        known_token_id = int(next(iter(vocabulary["gemma_id_to_compact_id"])))
        remapped = model.remap_input_tokens([known_token_id, unseen_token_id])
        self.assertEqual(remapped, [vocabulary["gemma_id_to_compact_id"][str(known_token_id)], vocabulary["unknown_compact_id"]])
        self.assertTrue(torch.all(model.input_embedding.weight[vocabulary["unknown_compact_id"]] == 0))

        old_style = tiny_model(tiny_input_vocabulary(with_unknown_row=False))
        with self.assertRaises(ValueError):
            old_style.remap_input_tokens([unseen_token_id])

    def test_checkpoints_load_strictly_and_old_ones_without_the_unknown_row_still_load(self):
        old_model = tiny_model(tiny_input_vocabulary(with_unknown_row=False), seed=1)
        old_state = old_model.state_dict()
        old_state["rotary_embedding.cos_cached"] = torch.zeros(3, 8)   # older checkpoints stored the rotary tables
        old_state["rotary_embedding.sin_cached"] = torch.zeros(3, 8)
        new_model = tiny_model(tiny_input_vocabulary(), seed=2)
        new_model.load_weights(old_state)
        rows = old_model.input_embedding.weight.shape[0]
        self.assertTrue(torch.equal(new_model.input_embedding.weight[:rows], old_model.input_embedding.weight))
        self.assertTrue(torch.all(new_model.input_embedding.weight[rows] == 0))

        renamed = dict(old_model.state_dict())
        renamed["output_head.weight"] = renamed.pop("output_projection.weight")
        with self.assertRaises(RuntimeError):
            new_model.load_weights(renamed)

    def test_incremental_decoder_matches_full_recomputation(self):
        model = tiny_model().eval()
        torch.manual_seed(3)
        prompts = [torch.randint(0, 50, (length,)).tolist() for length in (3, 11, 6)]
        sequences = [list(prompt) for prompt in prompts]
        with torch.no_grad():
            decoder = IncrementalDecoder(model, prompts)
            for _ in range(8):
                for row, sequence in enumerate(sequences):
                    reference = model(torch.tensor([sequence]))[0, -1]
                    self.assertLess((reference - decoder.next_logits()[row]).abs().max().item(), 1e-5)
                next_tokens = (decoder.next_logits().argmax(-1) % 50).tolist()
                for sequence, token in zip(sequences, next_tokens):
                    sequence.append(token)
                decoder.append(next_tokens)

    def test_fused_attention_equals_the_explicit_masked_attention(self):
        torch.manual_seed(5)
        model = Transformer(domain=REDDIT_SENTIMENT, hidden_dim=16, num_layers=2, num_heads=2, dropout=0.0,
                            input_vocabulary=tiny_input_vocabulary()).eval()
        input_ids = torch.randint(0, 50, (1, 13))
        with torch.no_grad():
            fused = model(input_ids)   # no mask: the fused kernel
            explicit = model(input_ids, attention_mask=model.create_causal_mask(13, input_ids.device))
        self.assertLess((fused - explicit).abs().max().item(), 1e-5)

        model.train()
        model.set_attention_noise(0.5)   # noise needs the explicit scores: still trains, and differs from no noise
        noisy = model(input_ids)
        noisy.sum().backward()
        self.assertGreater((noisy.detach() - explicit).abs().max().item(), 1e-4)

    def test_batched_rollouts_equal_one_at_a_time(self):
        model = tiny_model(seed=4).eval()
        encoder = ExampleEncoder(model, "cpu")
        evaluator = Evaluator(model, REDDIT_SENTIMENT, encoder, None, "cpu", loss_function=None,
                              eval_top_k=20, eval_cap_multiple=2.0)
        encoded_examples = [encoder.encode(synthetic_record(seed, step_count=3 + seed)) for seed in range(5)]
        with torch.no_grad():
            together = evaluator.rollouts(encoded_examples)
            alone = [evaluator.rollouts([encoded])[0] for encoded in encoded_examples]
        self.assertEqual(together, alone)
        self.assertEqual([len(rollout.tokens) for rollout in together if not rollout.terminated],
                         [2 * encoded.num_steps for encoded, rollout in zip(encoded_examples, together) if not rollout.terminated])

    def test_student_generate_stops_on_the_domain_stop_rule(self):
        class StopsAfterTwoTokens(RedditSentimentDomain):
            stop_token = "☃"   # never generated: only is_stop can end the rollout

            def is_stop(self, last_token_decoded, generated_text):
                self.calls = getattr(self, "calls", 0) + 1
                return self.calls >= 2

        domain = StopsAfterTwoTokens()
        model = tiny_model(domain=domain).eval()
        result = StudentModel(model, domain, "cpu").generate("lovely weather today", max_new_tokens=10)
        self.assertTrue(result.terminated)
        self.assertEqual(result.steps, 2)


# ── custom student networks (TrainingRun(model=...)) ──────────────────

class CustomModelTests(unittest.TestCase):
    @staticmethod
    def custom_student(model_class, model_settings=None) -> CustomStudent:
        torch.manual_seed(0)
        return CustomStudent(model_class, model_settings, REDDIT_SENTIMENT, DEFAULT_TEACHER_MODEL,
                             input_vocabulary=tiny_input_vocabulary())

    def test_a_model_that_looks_ahead_or_has_the_wrong_shape_is_refused(self):
        with self.assertRaisesRegex(ValueError, "looks ahead"):
            self.custom_student(LooksAheadStudent)
        with self.assertRaisesRegex(ValueError, "shape"):
            self.custom_student(WrongShapeStudent)
        student = self.custom_student(TinyLstmStudent, {"hidden": 8, "layers": 1})
        self.assertEqual(student.get_model_info()["output_vocab_size"], len(TOKEN_LIST))

    def test_a_custom_run_must_be_rebuildable_later(self):
        class DefinedInAFunction(TinyLstmStudent):
            pass

        script_model = type("ScriptModel", (TinyLstmStudent,), {"__module__": "__main__"})
        with self.assertRaisesRegex(ValueError, "script being run"):
            TrainingRun(domain=REDDIT_SENTIMENT, run_name="check", model=script_model)
        with self.assertRaisesRegex(ValueError, "cannot be imported"):
            TrainingRun(domain=REDDIT_SENTIMENT, run_name="check", model=DefinedInAFunction)
        with self.assertRaisesRegex(ValueError, "pass model= too"):
            TrainingRun(domain=REDDIT_SENTIMENT, run_name="check", model_settings={"hidden": 8})
        with self.assertRaisesRegex(ValueError, "plain values"):
            TrainingRun(domain=REDDIT_SENTIMENT, run_name="check", model=TinyLstmStudent,
                        model_settings={"hidden": torch.tensor(8)})

    def test_built_in_runs_keep_their_info_and_custom_runs_record_their_model(self):
        built_in = TrainingRun(domain=REDDIT_SENTIMENT, run_name="built-in").as_dict()
        self.assertNotIn("model", built_in)   # cohorts of old and new built-in runs still match
        self.assertNotIn("model_settings", built_in)
        self.assertEqual(built_in["hidden_dim"], 48)
        custom = TrainingRun(domain=REDDIT_SENTIMENT, run_name="custom", model=TinyLstmStudent,
                             model_settings={"hidden": 8}).as_dict()
        self.assertEqual(custom["model"], f"{TinyLstmStudent.__module__}.TinyLstmStudent")
        self.assertEqual(custom["model_settings"], {"hidden": 8})
        self.assertEqual([custom[key] for key in ("hidden_dim", "num_layers", "num_heads", "dropout")], [None] * 4)

    def test_simple_decoder_matches_the_cached_one(self):
        model = tiny_model().eval()
        torch.manual_seed(3)
        prompts = [torch.randint(0, 50, (length,)).tolist() for length in (3, 11, 6)]
        with torch.no_grad():
            cached, simple = IncrementalDecoder(model, prompts), SimpleDecoder(model, prompts)
            for step in range(6):
                self.assertLess((cached.next_logits() - simple.next_logits()).abs().max().item(), 1e-5)
                next_tokens = (cached.next_logits().argmax(-1) % 50).tolist()
                cached.append(next_tokens)
                simple.append(next_tokens)
                if step == 2:
                    cached.keep_rows([0, 2])
                    simple.keep_rows([0, 2])


# ── X-C1 ──────────────────────────────────────────────────────────────

class MajorityBaselineTests(unittest.TestCase):
    def test_majority_baseline_from_confusion_matrix_row_sums(self):
        confusion_matrices = {
            "tone": {"neutral": {"neutral": 5, "rude": 1, "<invalid>": 0}, "rude": {"neutral": 2, "rude": 0, "<invalid>": 0}},
            "safety": {"safe": {"safe": 9, "harmful": 0, "<invalid>": 0}, "harmful": {"safe": 1, "harmful": 0, "<invalid>": 0}},
        }
        self.assertAlmostEqual(majority_baseline(confusion_matrices), (6 + 9) / (8 + 10))
        self.assertIsNone(majority_baseline({}))


# ── one Stop for students training side by side (2026-10-27) ─────────

class StopFileTests(unittest.TestCase):
    def test_one_stop_file_reaches_every_process_and_the_next_start_clears_it(self):
        from library.shared.exit_listener import STOP_FILE, ExitListener
        previous_directory = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            os.chdir(directory)
            try:
                with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet):
                    first, second = ExitListener(), ExitListener()
                    Path(STOP_FILE).touch()
                    self.assertTrue(first.check_exit())
                    self.assertTrue(second.check_exit())   # the first one saw it, the file is still there
                    ExitListener()
                self.assertFalse(os.path.exists(STOP_FILE))
            finally:
                os.chdir(previous_directory)


# ── T-M1, T-L2, T-H2, T-M2, X-M2: whole runs on synthetic batches ────

TRAINING_FILES = 6   # ratio 0.75 of 8 files; the test set is the last 2 files (max_test_examples 8)


class RunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = Path(tempfile.mkdtemp())
        cls.batches = cls.folder / "batches"
        write_batches(cls.batches, 8, vocabulary_sha1=vocabulary_fingerprint(TOKEN_LIST))
        cls.runs = cls.folder / "runs"
        cls.patched = {
            (training_runner_module, "get_batches_dir"): lambda domain, teacher: str(cls.batches),
            (training_runner_module, "get_training_run_dir"): lambda name: str(cls.runs / name),
            (training_runner_module, "load_input_vocabulary"): lambda domain, teacher: tiny_input_vocabulary(),
            (training_runner_module, "write_queue"): lambda *arguments, **keywords: None,
            (training_runner_module, "plot_training_logs"): lambda *arguments, **keywords: None,
            (trainer_module, "plot_training_logs"): lambda *arguments, **keywords: None,
        }
        cls.originals = {key: getattr(*key) for key in cls.patched}
        for (module, name), replacement in cls.patched.items():
            setattr(module, name, replacement)
        cls.straight_weights = cls.train("straight")

    @classmethod
    def tearDownClass(cls):
        for (module, name), original in cls.originals.items():
            setattr(module, name, original)
        shutil.rmtree(cls.folder, ignore_errors=True)

    @staticmethod
    def spec(name: str) -> TrainingRun:
        return TrainingRun(domain=REDDIT_SENTIMENT, run_name=name, hidden_dim=8, num_layers=1, num_heads=1,
                           dropout=0.0, epoch_count=2, batch_size=4, learning_rate=0.003, lr_warmup_ratio=0.25,
                           max_training_examples=None, training_test_ratio=0.75, max_test_examples=8,
                           kl_ratio_start=0.9, kl_ratio_end=0.1)

    @staticmethod
    def custom_spec(name: str) -> TrainingRun:
        return replace(RunTests.spec(name), model=TinyLstmStudent, model_settings={"hidden": 8, "layers": 1})

    @classmethod
    def train(cls, name: str, stop_at_check: int = None, custom: bool = False):
        """One session of a run (seeded the same way every time). Returns the final
        weights, or None when the session stopped."""
        torch.manual_seed(7)
        spec = cls.custom_spec(name) if custom else cls.spec(name)
        completed = training_runner_module.TrainingRunner(spec, ScriptedExit(stop_at_check)).run()
        return cls.final_weights(name) if completed else None

    @classmethod
    def final_weights(cls, name: str) -> dict:
        path = cls.runs / name / "checkpoints" / "checkpoint_epoch_1.pt"
        return torch.load(path, weights_only=True)["model_state_dict"]

    def records(self, name: str, record_type: str) -> list:
        with open(self.runs / name / "logs" / "training.jsonl", encoding="utf-8") as file:
            return [record for record in map(json.loads, file) if record["type"] == record_type]

    def assertSameWeights(self, first: dict, second: dict):
        self.assertEqual(sorted(first), sorted(second))
        for key in first:
            self.assertTrue(torch.equal(first[key], second[key]), key)

    def test_warmup_applies_to_the_first_batch_and_the_used_rate_is_logged(self):
        used_rates = []
        original_step = torch.optim.AdamW.step

        def recording_step(optimizer, *arguments, **keywords):
            used_rates.append(optimizer.param_groups[0]["lr"])
            return original_step(optimizer, *arguments, **keywords)

        torch.optim.AdamW.step = recording_step
        try:
            self.train("warmup")
        finally:
            torch.optim.AdamW.step = original_step
        schedules = Schedules(self.spec("x").as_dict(), TRAINING_FILES, 2 * TRAINING_FILES)
        self.assertEqual(schedules.warmup_steps, 3)
        self.assertEqual(len(used_rates), 2 * TRAINING_FILES)   # one optimizer step per batch, not per example
        for step, rate in enumerate(used_rates):
            self.assertEqual(rate, schedules.learning_rate(step), step)
        self.assertEqual(used_rates[0], 0.0)   # batch 1 is inside the warm-up, not at the peak
        logged = [record["learning_rate"] for record in self.records("warmup", "train_batch")]
        self.assertEqual(logged, used_rates)
        logged_kl = [record["kl_ratio"] for record in self.records("warmup", "train_batch")]
        self.assertAlmostEqual(logged_kl[0], 0.9)   # the dials of the batch itself, not of the next one

    def test_completed_run_logs_baseline_and_leaves_no_resume_checkpoint(self):
        run_dir = self.runs / "straight"
        self.assertFalse((run_dir / "checkpoints" / "temp_checkpoint.pt").exists())
        self.assertTrue((run_dir / "input_vocabulary.json").exists())
        info = json.loads((run_dir / "info.json").read_text(encoding="utf-8"))
        self.assertEqual((info["status"], info["training_batch_count"], info["test_batch_count"]), ("completed", 6, 2))
        self.assertEqual(info["optimizer_updates"], "per_batch")   # tells these runs apart from the per-example ones
        evaluations = self.records("straight", "eval_epoch")
        self.assertEqual([record["epoch"] for record in evaluations], [0, 1])
        for record in evaluations:
            self.assertAlmostEqual(record["majority_baseline"], 1.0)   # every synthetic label is the same
            self.assertAlmostEqual(record["task_accuracy_excess"], round(record["task_accuracy"] - 1.0, 4))

    def test_stop_on_an_epochs_last_batch_still_writes_its_checkpoint(self):
        self.assertIsNone(self.train("edge", stop_at_check=TRAINING_FILES))
        self.assertTrue((self.runs / "edge" / "checkpoints" / "checkpoint_epoch_0.pt").exists())
        self.assertEqual([record["epoch"] for record in self.records("edge", "train_epoch")], [0])
        self.assertSameWeights(self.train("edge"), self.straight_weights)
        self.assertEqual([record["epoch"] for record in self.records("edge", "eval_epoch")], [0, 1])

    def test_mid_epoch_stop_resumes_exactly(self):
        self.assertIsNone(self.train("middle", stop_at_check=2))
        self.assertSameWeights(self.train("middle"), self.straight_weights)

    def test_custom_model_trains_resumes_and_loads_back(self):
        straight = self.train("lstm-straight", custom=True)
        self.assertIsNone(self.train("lstm-middle", stop_at_check=2, custom=True))
        self.assertSameWeights(self.train("lstm-middle", custom=True), straight)

        run_dir = self.runs / "lstm-straight"
        info = json.loads((run_dir / "info.json").read_text(encoding="utf-8"))
        self.assertEqual((info["status"], info["model"], info["hidden_dim"]),
                         ("completed", f"{TinyLstmStudent.__module__}.TinyLstmStudent", None))
        self.assertEqual(info["model_info"]["model"], "TinyLstmStudent")
        self.assertEqual([record["epoch"] for record in self.records("lstm-straight", "eval_epoch")], [0, 1])

        run = SimpleNamespace(name="lstm-straight", info=info, run_dir=run_dir)
        student = StudentModel.from_run(run, run_dir / "checkpoints" / "checkpoint_epoch_1.pt", device="cpu")
        self.assertIsInstance(student.model, CustomStudent)
        self.assertSameWeights(student.model.state_dict(), straight)
        self.assertGreater(student.generate("lovely weather today", max_new_tokens=5).steps, 0)

    def test_hard_crash_resumes_from_the_last_epoch_end(self):
        self.crash_in_second_epoch("crash")
        self.assertSameWeights(self.train("crash"), self.straight_weights)

    def test_stale_or_missing_resume_checkpoint_is_refused(self):
        checkpoint_folder = self.runs / "stale" / "checkpoints"
        self.assertIsNone(self.train("stale", stop_at_check=1))   # graceful stop after batch 1 of epoch 0
        stale_copy = self.folder / "stale_resume_checkpoint.pt"
        shutil.copy(checkpoint_folder / "temp_checkpoint.pt", stale_copy)
        self.crash_in_second_epoch("stale")   # resumes, finishes epoch 0, crashes in epoch 1

        shutil.copy(stale_copy, checkpoint_folder / "temp_checkpoint.pt")   # the old file reappears
        with self.assertRaisesRegex(RuntimeError, "Cannot resume"):
            self.train("stale")
        os.remove(checkpoint_folder / "temp_checkpoint.pt")
        with self.assertRaisesRegex(RuntimeError, "no resume checkpoint"):
            self.train("stale")
        info = json.loads((self.runs / "stale" / "info.json").read_text(encoding="utf-8"))
        self.assertEqual(info["status"], "in_progress")

    def crash_in_second_epoch(self, name: str):
        original_process_batch = trainer_module.Trainer._process_batch

        def crashing_process_batch(trainer, position, epoch):
            if epoch == 1 and position == 2:
                raise KeyboardInterrupt("simulated hard crash")
            return original_process_batch(trainer, position, epoch)

        trainer_module.Trainer._process_batch = crashing_process_batch
        try:
            with self.assertRaises(KeyboardInterrupt):
                self.train(name)
        finally:
            trainer_module.Trainer._process_batch = original_process_batch

    def test_run_training_reports_an_interrupt(self):
        runs = [self.spec("queue-a"), self.spec("queue-b")]
        torch.manual_seed(7)
        self.assertFalse(training_runner_module.run_training(runs, ScriptedExit(stop_at_check=1)))
        self.assertFalse((self.runs / "queue-b").exists())   # the queue stopped at the interrupted run
        self.assertTrue(training_runner_module.run_training(runs, ScriptedExit()))
        self.assertTrue(training_runner_module.run_training(runs, ScriptedExit()))   # both skipped as completed


if __name__ == "__main__":
    unittest.main()
