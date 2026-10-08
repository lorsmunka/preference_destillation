"""DistillationDataGenerator — generates + saves teacher distillation data for one job.

Renamed from the old QueueRunner; driven by a plain Python config dict (no queue JSON) and
logs via library.shared.logging.RunLogger. Prompt/stop dispatch lives in library.domain
via ModelHandler.
"""

from collections import Counter
from time import sleep, time

from library.data_gen.input_handler import InputHandler
from library.data_gen.model_handler import ModelHandler
from library.data_gen.saving_handler import SavingHandler
from library.shared.batch_files import vocabulary_fingerprint
from library.shared.logging import RunLogger
from library.shared import ExitListener, get_output_dir
from library.domain import get_domain


class DistillationDataGenerator:
    def __init__(self, config, exit_listener: ExitListener):
        if isinstance(config, dict):  # legacy dict — resolve the domain name to its object
            self.config = config
            self.domain = get_domain(config["domain"])
        else:  # a GenerationJob spec — keep its Domain object, flatten the rest for info.json
            self.config = config.as_dict()
            self.domain = config.domain
        self.exit_listener = exit_listener

        self.model_name = self.config["model_name"]
        self.max_examples = self.config["max_examples"]
        self.batch_size = self.config["batch_size"]
        self.generation_batch_size = self.config.get("generation_batch_size", 1)
        # the domain's teacher mode; a record may name its own (an agentic corpus mixes them)
        self.default_teacher_mode = getattr(self.domain, "teacher", "generate")

        input_path = self.domain.corpus_path
        output_dir = get_output_dir(self.domain.name, self.model_name)

        self.input_handler = InputHandler(input_path)
        self.teacher_mode_counts = dict(Counter(self.teacher_mode(index) for index in range(self.input_handler.input_count)))
        self.model_handler = ModelHandler(
            self.model_name, self.domain,
            load_in_4bit=self.config.get("load_in_4bit", False),
            chat_template=self.config.get("chat_template", False),
            thinking=self.config.get("thinking", False))
        self.token_list = self.model_handler.get_vocabulary()["token_list"]   # the logit columns, in order
        self.vocabulary_sha1 = vocabulary_fingerprint(self.token_list)
        self.logger = RunLogger(output_dir, batch_size=self.batch_size)
        self.saving_handler = SavingHandler(self.logger, output_dir, vocabulary_sha1=self.vocabulary_sha1)

        # Kept / skipped counts per turn type, continued from earlier sessions of this job.
        previous_info = self.saving_handler.read_info()
        self.kept_by_turn_type = previous_info.get("kept_by_turn_type", {})
        self.skipped_by_turn_type = previous_info.get("skipped_by_turn_type", {})
        self.previous_vocabulary_sha1 = previous_info.get("vocabulary_sha1")

    def teacher_mode(self, index) -> str:
        return self.input_handler.get_teacher_mode(index) or self.default_teacher_mode

    def info_fields(self) -> dict:
        """What info.json records: the job's settings and provenance (NF4 or bf16, chat
        template, teacher mode, vocabulary), how far it got, and per turn type how many
        examples were kept and why the others were skipped."""
        return {
            "domain": self.config["domain"],
            "model_name": self.model_name,
            "description": self.config.get("description", ""),
            "max_examples": self.max_examples,
            "batch_size": self.batch_size,
            "generation_batch_size": self.generation_batch_size,
            "load_in_4bit": self.config.get("load_in_4bit", False),
            "chat_template": self.config.get("chat_template", False) or self.config.get("thinking", False),
            "thinking": self.config.get("thinking", False),
            "teacher_mode": self.default_teacher_mode,
            "teacher_mode_counts": self.teacher_mode_counts,   # records per mode in the corpus
            "vocabulary_sha1": self.vocabulary_sha1,
            "vocabulary_size": len(self.token_list),
            "input_count": self.input_handler.input_count,
            "examples_processed": self.logger.processed_sentence_count,
            "examples_generated": self.logger.successful_sentence_count,
            "batches_written": -(-self.logger.successful_sentence_count // self.batch_size),
            "kept_by_turn_type": self.kept_by_turn_type,
            "skipped_by_turn_type": self.skipped_by_turn_type,
        }

    def run(self):
        if self.logger.successful_sentence_count and self.previous_vocabulary_sha1 not in (None, self.vocabulary_sha1):
            raise ValueError(
                f"{self.saving_handler.output_dir}: the reduced vocabulary changed since the earlier batches "
                f"({self.previous_vocabulary_sha1} -> {self.vocabulary_sha1}); their logit columns would mean "
                f"other tokens. Generate into a fresh directory instead.")
        self.saving_handler.write_info(self.info_fields())

        batch_examples = []
        batch_start_time = time()
        batch_processed = 0
        batch_skip_reasons = {}

        index = self.logger.processed_sentence_count
        while index < self.input_handler.input_count:
            if self.logger.successful_sentence_count >= self.max_examples:
                print(f"Reached max_examples limit ({self.max_examples})")
                break

            # One input at a time (the original loop), or `generation_batch_size` together
            group = list(range(index, min(index + self.generation_batch_size, self.input_handler.input_count)))
            texts = [self.input_handler.get_input(i) for i in group]
            modes = [self.teacher_mode(i) for i in group]
            if len(set(modes)) > 1:
                outcomes = [None] * len(group)
                for mode in set(modes):
                    positions = [position for position, record_mode in enumerate(modes) if record_mode == mode]
                    subset_texts = [texts[position] for position in positions]
                    if mode == "forced":
                        subset = self.model_handler.score_training_examples(
                            subset_texts, [self.input_handler.get_target(group[position]) for position in positions])
                    else:
                        subset = self.model_handler.generate_training_examples(subset_texts)
                    for position, outcome in zip(positions, subset):
                        outcomes[position] = outcome
            elif modes[0] == "forced":   # gold targets, teacher soft labels
                targets = [self.input_handler.get_target(i) for i in group]
                outcomes = self.model_handler.score_training_examples(texts, targets)
            elif self.generation_batch_size == 1:
                outcomes = [self.model_handler.generate_training_example(texts[0])]
            else:
                outcomes = self.model_handler.generate_training_examples(texts)
            index = group[-1] + 1

            stop_requested = False
            for i, mode, (training_example, skip_reason) in zip(group, modes, outcomes):
                self.logger.processed_sentence_count += 1
                batch_processed += 1
                metadata = self.input_handler.get_metadata(i)
                turn_type = metadata["turn_type"] or "none"

                if training_example is not None:
                    training_example.update(metadata)
                    training_example["teacher_mode"] = mode
                    ground_truth = self.input_handler.get_ground_truth(i)
                    if ground_truth is not None:   # kept either way; the flag is for analysis
                        training_example["ground_truth"] = ground_truth
                        # a forced response is the gold target itself, so it says nothing about the teacher
                        training_example["answer_correct"] = None if mode == "forced" else self.domain.answer_matches(
                            training_example["model_response"], ground_truth)
                    batch_examples.append(training_example)
                    self.logger.successful_sentence_count += 1
                    self.kept_by_turn_type[turn_type] = self.kept_by_turn_type.get(turn_type, 0) + 1
                else:
                    batch_skip_reasons[skip_reason] = batch_skip_reasons.get(skip_reason, 0) + 1
                    reasons = self.skipped_by_turn_type.setdefault(turn_type, {})
                    reasons[skip_reason] = reasons.get(skip_reason, 0) + 1

                if len(batch_examples) == self.batch_size:
                    batch_examples, batch_processed, batch_skip_reasons, batch_start_time = self.save_batch(
                        batch_examples, batch_processed, batch_skip_reasons, batch_start_time)
                    if self.exit_listener.check_exit():
                        stop_requested = True
                        break
            if stop_requested:   # right after a saved batch and its state; the rest of this group is redone on resume
                break

            if self.generation_batch_size == 1:
                sleep(0.1)

        if batch_examples:
            self.save_batch(batch_examples, batch_processed, batch_skip_reasons, batch_start_time)
            print(f"Saved final partial batch with {len(batch_examples)} examples")

        # Done when the example limit is reached or every input record has been through the
        # teacher (a job over the whole corpus never reaches max_examples once anything is skipped).
        finished = (self.logger.successful_sentence_count >= self.max_examples
                    or self.logger.processed_sentence_count >= self.input_handler.input_count)
        self.logger.save()
        self.saving_handler.write_info(self.info_fields(), status="completed" if finished else "incomplete")
        print(f"Kept per turn type: {self.kept_by_turn_type}; skipped per turn type: {self.skipped_by_turn_type}")

    def save_batch(self, batch_examples, batch_processed, batch_skip_reasons, batch_start_time):
        batch_time = time() - batch_start_time
        target_examples = min(self.max_examples, self.input_handler.input_count)
        progress_percent = (self.logger.successful_sentence_count / target_examples) * 100
        batch_index = -(-self.logger.successful_sentence_count // self.batch_size)   # the number in the file name

        print(f"Saving batch {batch_index}, progress: {progress_percent:.2f}% "
              f"({self.logger.successful_sentence_count}/{target_examples}), "
              f"batch time: {batch_time:.2f}s, avg per sentence: {batch_time / batch_processed:.2f}s")

        self.saving_handler.save_batch(batch_examples)

        self.logger.log_generation_batch(
            batch_index=batch_index,
            processed=batch_processed,
            successful=len(batch_examples),
            skipped=batch_processed - len(batch_examples),
            time_seconds=batch_time,
            skip_reasons=batch_skip_reasons if batch_skip_reasons else None,
        )
        # The batch file is on disk before the state that counts it, so a crash resumes right
        # after this batch instead of starting the job again (and overwriting its batches).
        self.logger.save()
        self.saving_handler.write_info(self.info_fields())

        return [], 0, {}, time()
