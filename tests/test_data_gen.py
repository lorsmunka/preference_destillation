"""The data-generation fixes from the code review (D-H1, D-H2, D-H3, D-M1 ... D-M4,
D-L1 ... D-L4), one test class per fix. CPU only and fast: a fake teacher, a tiny word-level
tokenizer and temporary folders, no model weights. The BOS check also runs on the real
Gemma tokenizers and the legacy check on real batch files when they are on disk (skipped
otherwise).

    python -m unittest discover -s tests -p "test_data_gen.py"
"""

import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ["CUDA_VISIBLE_DEVICES"] = ""   # CPU only, even on a machine with a GPU
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
from transformers import AutoTokenizer, PreTrainedTokenizerFast

import generate_data
import library.data_gen.generator as generator_module
import library.data_gen.model_handler as model_handler_module
import library.shared.batch_files as batch_files
from library import GenerationJob
from library.data_gen.generator import DistillationDataGenerator
from library.data_gen.model_handler import ModelHandler
from library.domain.base import Domain
from library.shared.batch_files import batch_paths, read_batch, read_batch_meta, vocabulary_fingerprint, write_batch
from library.shared.vocabulary import extract_logits_as_vector

REPO_ROOT = Path(__file__).resolve().parents[1]
METADATA_KEYS = ("split", "item_id", "turn_type", "teacher_mode")

# "abcdef": StopStringCriteria probes the vocabulary with that word, so it must be a token.
TINY_VOCABULARY = ["<pad>", "<eos>", "<bos>", "<unk>", "<turn|>", "abcdef", "Answer", ":", "1", "2", ";", "');", "</s>"]
REDUCED_TOKENS = ["Answer", ":", "1", "2", ";"]   # the logit columns


class TinyDomain(Domain):
    """A test domain: answers end with ";" and the corpus lives in a temporary folder."""
    name = "tiny_test_domain"
    stop_token = ";"
    max_steps = 6

    def __init__(self, corpus_path: str = ""):
        self.test_corpus_path = corpus_path

    @property
    def corpus_path(self) -> str:
        return self.test_corpus_path

    def teacher_prompt(self, text: str) -> str:
        return f"Answer : {text}"

    def task_metric(self):
        return None

    def answer_matches(self, response: str, ground_truth) -> bool:
        return response.rstrip(";") == str(ground_truth)


def tiny_tokenizer(adds_bos: bool) -> PreTrainedTokenizerFast:
    """A word-level tokenizer over TINY_VOCABULARY (split on spaces, decoded without them).
    adds_bos=True puts a BOS in front by itself, like Gemma 3; False does not, like Gemma 4.
    Its chat template writes <bos> into the text, as both Gemma templates do."""
    backend = Tokenizer(models.WordLevel({token: index for index, token in enumerate(TINY_VOCABULARY)},
                                         unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.decoder = decoders.Fuse()
    if adds_bos:
        backend.post_processor = processors.TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", 2)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="<bos>", eos_token="<eos>",
                                        pad_token="<pad>", unk_token="<unk>")
    tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}{{ message['content'] }}{% endfor %} <turn|>"
    return tokenizer


class ScriptedModel:
    """Stands in for the teacher's generate(): every row gets the same scripted tokens, the
    logits processors see every step as in the real loop, and each call's arguments are kept."""

    def __init__(self, script_tokens, tokenizer):
        self.script_ids = tokenizer.convert_tokens_to_ids(script_tokens)
        self.vocabulary_size = len(tokenizer.get_vocab())
        self.calls = []

    def generate(self, input_ids, logits_processor=(), **arguments):
        self.calls.append({"input_ids": input_ids, **arguments})
        sequence = input_ids
        for token_id in self.script_ids:
            scores = torch.zeros(sequence.shape[0], self.vocabulary_size)
            scores[:, token_id] = 1.0
            for processor in logits_processor:
                processor(sequence, scores)
            sequence = torch.cat([sequence, torch.full((sequence.shape[0], 1), token_id)], dim=1)
        return sequence


def tiny_handler(tokenizer, chat_template: bool = False) -> ModelHandler:
    """A real ModelHandler around `tokenizer` and the scripted model (no weights loaded)."""
    def load(handler):
        handler.tokenizer = tokenizer
        handler.model = ScriptedModel(["1", ";"], tokenizer)
        handler.end_token_ids = {tokenizer.eos_token_id}
        handler.vocabulary = {"token_list": REDUCED_TOKENS, "vocab_size": len(REDUCED_TOKENS),
                              "token_to_id": {token: tokenizer.convert_tokens_to_ids(token) for token in REDUCED_TOKENS}}
        handler.response_tokens = set(REDUCED_TOKENS)

    with mock.patch.object(ModelHandler, "load_tokenizer_and_model", load):
        return ModelHandler("tiny", TinyDomain(), chat_template=chat_template)


def example(sentence: str, response: str = "1;", **fields) -> dict:
    steps = [{"token": "1", "logits": [0.5, 1.0, -2.0], "predicted_token_index": 1},
             {"token": ";", "logits": [0.0, 0.25, 3.0], "predicted_token_index": 2}]
    return {"sentence": sentence, "model_response": response, "steps": steps, **fields}


class FakeTeacher:
    """Stands in for ModelHandler inside the generator: a generated answer is "1;", a forced
    one is its gold target; texts in `rejected` are skipped as unexpected_tokens; call number
    `crash_on_call` raises like a CUDA out-of-memory error."""

    def __init__(self, crash_on_call=None, rejected=(), token_list=REDUCED_TOKENS):
        self.crash_on_call, self.rejected, self.token_list = crash_on_call, set(rejected), token_list
        self.calls = 0

    def get_vocabulary(self):
        return {"token_list": list(self.token_list)}

    def outcomes(self, texts, responses):
        self.calls += 1
        if self.calls == self.crash_on_call:
            raise RuntimeError("CUDA out of memory (simulated)")
        return [(None, "unexpected_tokens") if text in self.rejected else (example(text, response), None)
                for text, response in zip(texts, responses)]

    def generate_training_examples(self, texts):
        return self.outcomes(texts, ["1;"] * len(texts))

    def generate_training_example(self, text):
        return self.outcomes([text], ["1;"])[0]

    def score_training_examples(self, texts, targets):
        return self.outcomes(texts, targets)


class ExitListenerStub:
    """Answers check_exit like ExitListener: True once `stop_after_checks` checks were made
    (never, by default) or once `stopped` is set."""

    def __init__(self, stop_after_checks=None):
        self.stop_after_checks, self.checks, self.stopped = stop_after_checks, 0, False

    def check_exit(self):
        self.checks += 1
        if self.stop_after_checks is not None and self.checks >= self.stop_after_checks:
            self.stopped = True
        return self.stopped


def corpus(count: int = 40) -> list:
    """Records shaped like the math_tutor corpus: train turns first, then test turns; every
    third record is a teacher-forced explanation, the rest are generated solve turns."""
    records = []
    for index in range(count):
        record = {"text": f"text {index}", "answer": 1, "split": "train" if index < 30 else "test",
                  "item_id": f"item-{index // 2}", "turn_type": "explain" if index % 3 == 0 else "solve"}
        if index % 3 == 0:
            record.update(teacher="forced", target="1;")
        records.append(record)
    return records


def write_corpus(directory: str, records: list) -> None:
    with open(os.path.join(directory, "corpus.jsonl"), "w", encoding="utf-8") as file:   # a fresh temporary file
        for record in records:
            file.write(json.dumps(record) + "\n")


def run_generation(directory: str, teacher: FakeTeacher, exit_listener=None, max_examples: int = 1000) -> Path:
    """One generation job over directory/corpus.jsonl into directory/batches, with the fake teacher."""
    job = GenerationJob(domain=TinyDomain(os.path.join(directory, "corpus.jsonl")), teacher_model="fake/teacher",
                        max_examples=max_examples, batch_size=4, generation_batch_size=2,
                        load_in_4bit=True, chat_template=True, description="test job")
    output = Path(directory) / "batches"
    with mock.patch.object(generator_module, "ModelHandler", return_value=teacher), \
            mock.patch.object(generator_module, "get_output_dir", return_value=str(output)), \
            contextlib.redirect_stdout(io.StringIO()):
        DistillationDataGenerator(job, exit_listener or ExitListenerStub()).run()
    return output


def read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def saved_records(output: Path) -> list:
    return [record for path in batch_paths(output) for record in read_batch(path)]


def setUpModule():
    patcher = mock.patch.object(model_handler_module, "get_device", return_value="cpu")
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)


class StopCriterionBuiltOnce(unittest.TestCase):   # D-H1
    def test_one_criterion_per_stop_set_reused_across_calls(self):
        handler = tiny_handler(tiny_tokenizer(adds_bos=False))
        built = []
        real_criterion = model_handler_module.StopStringCriteria

        def counting_criterion(tokenizer, stop_strings):
            built.append(tuple(stop_strings))
            return real_criterion(tokenizer, stop_strings)

        with mock.patch.object(model_handler_module, "StopStringCriteria", side_effect=counting_criterion), \
                contextlib.redirect_stdout(io.StringIO()):
            for _ in range(3):
                results = handler.generate_training_examples(["1", "2"])
                texts = handler.generate_texts(["Answer"], stop_strings=["</s>"])
                handler.generate_texts(["Answer"])
        self.assertEqual(built, [(";",), ("</s>",)])   # each built on first use only, over three rounds
        calls = handler.model.calls
        self.assertTrue(all("stop_strings" not in call for call in calls))   # nothing for generate() to rebuild
        domain_criteria = [call["stopping_criteria"][0] for call in calls[0::3]]
        self.assertTrue(all(criterion is domain_criteria[0] for criterion in domain_criteria))
        self.assertIsNone(calls[2]["stopping_criteria"])
        self.assertEqual([example["model_response"] for example, _ in results], ["1;", "1;"])
        self.assertEqual(texts, ["1;"])

    def test_criterion_fires_on_tokens_that_contain_the_stop_string(self):
        # Why one-token stops are not turned into eos ids: an eos id would miss "');".
        tokenizer = tiny_tokenizer(adds_bos=False)
        criterion = tiny_handler(tokenizer).stopping_criteria([";"])
        for tokens, expected in ((["Answer", ":", "1", ";"], True), (["1", "');"], True), (["Answer", ":", "1"], False)):
            token_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])
            self.assertEqual(bool(criterion(token_ids, None)[0]), expected, tokens)


class CrashResumesAfterSavedBatches(unittest.TestCase):   # D-H2 (and D-L3: started_at kept)
    def test_crash_keeps_earlier_batches_and_resumes_after_them(self):
        with tempfile.TemporaryDirectory() as directory:
            write_corpus(directory, corpus(40))
            with self.assertRaises(RuntimeError):
                run_generation(directory, FakeTeacher(crash_on_call=12))
            output = Path(directory) / "batches"
            state = read_json(output / "state.json")
            saved = [record["sentence"] for record in saved_records(output)]
            self.assertGreaterEqual(len(saved), 8)
            self.assertEqual(state["successful_sentence_count"], len(saved))   # the state counts exactly what is on disk
            self.assertEqual(saved, [f"text {index}" for index in range(len(saved))])
            first_started_at = read_json(output / "info.json")["started_at"]

            run_generation(directory, FakeTeacher())
            sentences = [record["sentence"] for record in saved_records(output)]
            self.assertEqual(sentences, [f"text {index}" for index in range(40)])   # nothing redone twice, nothing lost
            self.assertEqual(len(batch_paths(output)), 10)
            info = read_json(output / "info.json")
            self.assertEqual(info["status"], "completed")
            self.assertEqual(info["started_at"], first_started_at)
            self.assertEqual(info["examples_generated"], 40)
            self.assertEqual(sum(info["kept_by_turn_type"].values()), 40)
            self.assertEqual(list(output.glob("*.partial")), [])

    def test_crash_while_writing_leaves_no_batch_file(self):
        def crashing_savez(file, **arrays):
            file.write(b"half a batch")
            raise RuntimeError("killed mid-write (simulated)")

        with tempfile.TemporaryDirectory() as directory:
            write_batch(Path(directory) / "batch_1.npz", [example("a")])
            with mock.patch.object(batch_files.np, "savez", side_effect=crashing_savez):
                with self.assertRaises(RuntimeError):
                    write_batch(Path(directory) / "batch_2.npz", [example("b")])
            self.assertEqual([path.name for path in batch_paths(directory)], ["batch_1.npz"])
            self.assertEqual(read_batch(Path(directory) / "batch_1.npz")[0]["sentence"], "a")

    def test_records_added_after_a_partial_batch_overwrite_nothing(self):
        # 10 records in batches of 4: the last file holds 2. Records added to the corpus later go to
        # new files after it; numbering by the example count would have reused numbers 3 and 4.
        for added in (1, 5):
            with tempfile.TemporaryDirectory() as directory:
                write_corpus(directory, corpus(10))
                output = run_generation(directory, FakeTeacher())
                old_files = {path.name: path.read_bytes() for path in batch_paths(output)}
                self.assertEqual(sorted(old_files), ["batch_1.npz", "batch_2.npz", "batch_3.npz"])
                write_corpus(directory, corpus(10 + added))   # the same 10 records first, then the new ones
                run_generation(directory, FakeTeacher())
                self.assertTrue(all((output / name).read_bytes() == data for name, data in old_files.items()))
                self.assertEqual([record["sentence"] for record in saved_records(output)],
                                 [f"text {index}" for index in range(10 + added)])
                self.assertEqual(len(batch_paths(output)), 3 + -(-added // 4))
                self.assertEqual(read_json(output / "info.json")["batches_written"], 3 + -(-added // 4))

    def test_graceful_stop_then_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            write_corpus(directory, corpus(40))
            output = run_generation(directory, FakeTeacher(), exit_listener=ExitListenerStub(stop_after_checks=2))
            self.assertEqual(read_json(output / "info.json")["status"], "incomplete")
            self.assertEqual(len(saved_records(output)), 8)
            run_generation(directory, FakeTeacher())
            info = read_json(output / "info.json")
            self.assertEqual([record["sentence"] for record in saved_records(output)], [f"text {index}" for index in range(40)])
            self.assertEqual((info["status"], sum(info["kept_by_turn_type"].values())), ("completed", 40))


class VocabularyFingerprint(unittest.TestCase):   # D-H3
    def test_fingerprint_is_sha1_of_the_ordered_token_list(self):
        self.assertEqual(vocabulary_fingerprint(["a", "b"]), hashlib.sha1(b"a\nb").hexdigest())
        self.assertNotEqual(vocabulary_fingerprint(["a", "b"]), vocabulary_fingerprint(["b", "a"]))

    def test_written_and_read_back(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_batch(Path(directory) / "batch_1.npz", [example("a")], vocabulary_sha1="abc123")
            self.assertEqual(read_batch_meta(path)["vocabulary_sha1"], "abc123")
            self.assertEqual(read_batch(path)[0]["sentence"], "a")

    def test_generator_stores_it_in_every_batch_and_in_info(self):
        with tempfile.TemporaryDirectory() as directory:
            write_corpus(directory, corpus(10))
            output = run_generation(directory, FakeTeacher())
            expected = vocabulary_fingerprint(REDUCED_TOKENS)
            self.assertEqual([read_batch_meta(path)["vocabulary_sha1"] for path in batch_paths(output)], [expected] * 3)
            info = read_json(output / "info.json")
            self.assertEqual((info["vocabulary_sha1"], info["vocabulary_size"]), (expected, len(REDUCED_TOKENS)))

    def test_resume_with_a_changed_vocabulary_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            write_corpus(directory, corpus(40))
            run_generation(directory, FakeTeacher(), exit_listener=ExitListenerStub(stop_after_checks=1))
            with self.assertRaises(ValueError):
                run_generation(directory, FakeTeacher(token_list=REDUCED_TOKENS + ["extra"]))


class ExactlyOneBos(unittest.TestCase):   # D-M2
    def assert_one_bos(self, token_ids, bos_token_id):
        self.assertEqual(token_ids[0], bos_token_id)
        self.assertEqual(list(token_ids).count(bos_token_id), 1)

    def test_tokenizer_that_adds_bos_or_not_with_and_without_chat_template(self):
        for adds_bos in (False, True):
            for chat_template in (False, True):
                tokenizer = tiny_tokenizer(adds_bos)
                handler = tiny_handler(tokenizer, chat_template=chat_template)
                self.assert_one_bos(handler.prompt_token_ids(handler.build_prompt("1")), tokenizer.bos_token_id)
                with contextlib.redirect_stdout(io.StringIO()):
                    handler.generate_training_examples(["1", "2 2 2"])   # different lengths: left padding
                call = handler.model.calls[-1]
                for row, mask in zip(call["input_ids"].tolist(), call["attention_mask"].tolist()):
                    self.assert_one_bos([token for token, keep in zip(row, mask) if keep], tokenizer.bos_token_id)

    def test_real_gemma_tokenizers(self):
        sources = [str(REPO_ROOT / "models" / "gemma-4-12B-it"), "google/gemma-3-4b-it"]
        checked = 0
        for source in sources:
            try:
                tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
            except (OSError, ValueError):
                continue
            for chat_template in (False, True):
                handler = tiny_handler(tokenizer, chat_template=chat_template)
                self.assert_one_bos(handler.prompt_token_ids(handler.build_prompt("What is 2 + 3?")),
                                    tokenizer.bos_token_id)
            checked += 1
        if not checked:
            self.skipTest("no Gemma tokenizer on disk")


class StopEndsTheJobQueue(unittest.TestCase):   # D-M1
    def run_queue(self, exit_listener, stop_during_first_job: bool) -> list:
        started = []

        class FakeGenerator:
            def __init__(self, job, listener):
                started.append(job.domain.name)

            def run(self):
                if stop_during_first_job:
                    exit_listener.stopped = True   # Ctrl+C while this job runs

        jobs = [SimpleNamespace(domain=SimpleNamespace(name=f"tiny_queue_{letter}"), teacher_model="fake/teacher")
                for letter in "ab"]
        with mock.patch.object(generate_data, "DistillationDataGenerator", FakeGenerator), \
                contextlib.redirect_stdout(io.StringIO()):
            generate_data.run_jobs(jobs, exit_listener)
        return started

    def test_stop_during_a_job_does_not_start_the_next(self):
        self.assertEqual(self.run_queue(ExitListenerStub(), stop_during_first_job=True), ["tiny_queue_a"])

    def test_without_a_stop_every_job_runs(self):
        self.assertEqual(self.run_queue(ExitListenerStub(), stop_during_first_job=False), ["tiny_queue_a", "tiny_queue_b"])


class PerRecordMetadata(unittest.TestCase):   # D-M3, D-L2
    def test_round_trip_through_a_batch_file(self):
        with tempfile.TemporaryDirectory() as directory:
            fields = {"split": "test", "item_id": "S01-0001", "turn_type": "explain", "teacher_mode": "forced"}
            path = write_batch(Path(directory) / "batch_1.npz", [example("a", **fields), example("b")])
            first, second = read_batch(path)
            self.assertEqual({key: first[key] for key in METADATA_KEYS}, fields)
            self.assertEqual({key: second[key] for key in METADATA_KEYS}, dict.fromkeys(METADATA_KEYS))

    def test_generator_carries_corpus_fields_into_the_examples(self):
        with tempfile.TemporaryDirectory() as directory:
            records = corpus(40)
            write_corpus(directory, records)
            examples = saved_records(run_generation(directory, FakeTeacher()))
            self.assertEqual(len(examples), 40)
            for record, saved in zip(records, examples):
                forced = record.get("teacher") == "forced"
                self.assertEqual((saved["split"], saved["item_id"], saved["turn_type"]),
                                 (record["split"], record["item_id"], record["turn_type"]))
                self.assertEqual(saved["teacher_mode"], "forced" if forced else "generate")
                # a forced response is the gold target itself: no claim about the teacher (D-L2)
                self.assertEqual(saved["answer_correct"], None if forced else True)
            self.assertEqual(sum(saved["split"] == "test" for saved in examples), 10)


class JobCompletesWhenEveryRecordWasProcessed(unittest.TestCase):   # D-M4, D-L3, skips per turn type
    def test_skips_do_not_keep_a_whole_corpus_job_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            write_corpus(directory, corpus(40))
            output = run_generation(directory, FakeTeacher(rejected={"text 3", "text 9"}), max_examples=1_000_000)
            info = read_json(output / "info.json")
            self.assertEqual(info["status"], "completed")
            self.assertIsNotNone(info["completed_at"])
            self.assertEqual((info["examples_processed"], info["examples_generated"]), (40, 38))
            self.assertEqual(info["skipped_by_turn_type"], {"explain": {"unexpected_tokens": 2}})
            self.assertEqual(info["kept_by_turn_type"], {"explain": 12, "solve": 26})

    def test_info_records_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            write_corpus(directory, corpus(10))
            info = read_json(run_generation(directory, FakeTeacher()) / "info.json")
            self.assertEqual({key: info[key] for key in ("load_in_4bit", "chat_template", "thinking",
                                                        "generation_batch_size", "teacher_mode")},
                             {"load_in_4bit": True, "chat_template": True, "thinking": False,
                              "generation_batch_size": 2, "teacher_mode": "generate"})
            self.assertEqual(info["teacher_mode_counts"], {"forced": 4, "generate": 6})


class ValidatesStoredStepTokens(unittest.TestCase):   # D-L1
    def steps(self, tokens):
        return [{"token": token, "logits": [0.0] * len(REDUCED_TOKENS), "predicted_token_index": 0} for token in tokens]

    def test_stored_token_outside_the_vocabulary_is_rejected(self):
        handler = tiny_handler(tiny_tokenizer(adds_bos=False))
        with mock.patch.object(handler.tokenizer, "tokenize", return_value=["1", ";"]), \
                contextlib.redirect_stdout(io.StringIO()):   # a clean re-tokenisation must not hide the stored token
            self.assertEqual(handler.finish_example("text", "1');", self.steps(["1", "');"]), ""), (None, "unexpected_tokens"))
            kept, reason = handler.finish_example("text", "1;", self.steps(["1", ";"]), "")
        self.assertIsNone(reason)
        self.assertEqual([step["token"] for step in kept["steps"]], ["1", ";"])

    def test_too_long_counts_the_stored_steps(self):
        handler = tiny_handler(tiny_tokenizer(adds_bos=False))
        with contextlib.redirect_stdout(io.StringIO()):
            _, reason = handler.finish_example("text", "1" * 7, self.steps(["1"] * (TinyDomain.max_steps + 1)), "")
        self.assertEqual(reason, "too_long")


class ReducedLogitsInOneCopy(unittest.TestCase):   # D-L4
    def test_same_values_as_one_item_per_entry(self):
        vocabulary = {"token_list": ["c", "a", "b"], "token_to_id": {"a": 0, "b": 3, "c": 7}}
        for dtype in (torch.float32, torch.bfloat16):
            logits = torch.randn(10).to(dtype)
            expected = [logits[vocabulary["token_to_id"][token]].item() for token in vocabulary["token_list"]]
            self.assertEqual(extract_logits_as_vector(logits, vocabulary), expected)


class LegacyBatchesStillLoad(unittest.TestCase):
    def test_legacy_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "batch_1.jsonl"
            with open(path, "w", encoding="utf-8") as file:   # a fresh temporary file
                file.write(json.dumps(example("a")) + "\n")
            record = read_batch(path)[0]
            self.assertEqual(record["steps"][1]["logits"], [0.0, 0.25, 3.0])
            self.assertEqual({key: record[key] for key in METADATA_KEYS}, dict.fromkeys(METADATA_KEYS))
            self.assertEqual(read_batch_meta(path), {})

    def test_npz_written_before_these_fixes(self):
        records = [example("a"), example("b", "2;")]
        meta = {"sentences": ["a", "b"], "model_responses": ["1;", "2;"],
                "tokens": [["1", ";"], ["1", ";"]], "thinking": ["", ""]}
        logits = np.asarray([step["logits"] for record in records for step in record["steps"]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "batch_1.npz"
            np.savez(path, logits_bf16=batch_files.float32_to_bfloat16_bits(logits),
                     step_offsets=np.array([0, 2, 4], dtype=np.int64),
                     predicted_token_index=np.array([1, 2, 1, 2], dtype=np.int32), meta=np.array(json.dumps(meta)))
            loaded = read_batch(path)
            self.assertNotIn("vocabulary_sha1", read_batch_meta(path))
        self.assertEqual([record["model_response"] for record in loaded], ["1;", "2;"])
        self.assertEqual(loaded[1]["steps"][1]["logits"].tolist(), [0.0, 0.25, 3.0])
        self.assertTrue(all(record[key] is None for record in loaded for key in METADATA_KEYS))

    def test_real_batches_on_disk(self):
        directory = REPO_ROOT / "batches" / "math_word_problem" / "google_gemma-3-4b-it"
        paths = [path for path in (directory / "batch_1.jsonl", directory / "batch_1.npz") if path.exists()]
        if not paths:
            self.skipTest("no existing batches on disk")
        loaded = [read_batch(path) for path in paths]
        for records in loaded:
            self.assertGreater(len(records), 0)
            self.assertTrue(all(record[key] is None for record in records for key in METADATA_KEYS))
        if len(loaded) == 2:   # the same batch in both formats reads the same
            self.assertEqual([record["sentence"] for record in loaded[0]], [record["sentence"] for record in loaded[1]])


if __name__ == "__main__":
    unittest.main()
