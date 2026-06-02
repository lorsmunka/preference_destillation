"""DistillationDataGenerator — generates + saves teacher distillation data for one job.

Renamed from the old QueueRunner; driven by a plain Python config dict (no queue JSON) and
logs via library.shared.logging.RunLogger. Prompt/stop dispatch lives in library.domain
via ModelHandler.
"""

from time import sleep, time
from datetime import datetime, timezone

from library.data_gen.input_handler import InputHandler
from library.data_gen.model_handler import ModelHandler
from library.data_gen.saving_handler import SavingHandler
from library.shared.logging import RunLogger
from library.shared import ExitListener, get_output_dir
from library.domain import get_domain


class DistillationDataGenerator:
    def __init__(self, config: dict, exit_listener: ExitListener):
        self.config = config
        self.exit_listener = exit_listener

        self.domain = config["domain"]
        self.model_name = config["model_name"]
        self.max_examples = config["max_examples"]
        self.batch_size = config["batch_size"]

        self.config["started_at"] = datetime.now(timezone.utc).isoformat()

        input_path = get_domain(self.domain).corpus_path
        output_dir = get_output_dir(self.domain, self.model_name)

        self.input_handler = InputHandler(input_path)
        self.model_handler = ModelHandler(self.model_name, self.domain)
        self.logger = RunLogger(output_dir, batch_size=self.batch_size)
        self.saving_handler = SavingHandler(self.logger, output_dir)

    def run(self):
        self.saving_handler.write_info(self.config)

        batch_examples = []
        batch_start_time = time()
        batch_processed = 0
        batch_skip_reasons = {}

        for i in range(self.logger.processed_sentence_count, self.input_handler.input_count):
            if self.logger.successful_sentence_count >= self.max_examples:
                print(f"Reached max_examples limit ({self.max_examples})")
                break

            text = self.input_handler.get_input(i)
            training_example, skip_reason = self.model_handler.generate_training_example(text)

            self.logger.processed_sentence_count += 1
            batch_processed += 1

            if training_example is not None:
                batch_examples.append(training_example)
                self.logger.successful_sentence_count += 1
            else:
                batch_skip_reasons[skip_reason] = batch_skip_reasons.get(skip_reason, 0) + 1

            if len(batch_examples) == self.batch_size:
                batch_examples, batch_processed, batch_skip_reasons, batch_start_time = self.save_batch(
                    batch_examples, batch_processed, batch_skip_reasons, batch_start_time)

                if self.exit_listener.check_exit():
                    self.logger.save()
                    break

            sleep(0.1)

        if batch_examples:
            self.saving_handler.save_batch(batch_examples)
            print(f"Saved final partial batch with {len(batch_examples)} examples")

        status = "completed" if self.logger.successful_sentence_count >= self.max_examples else "incomplete"

        self.saving_handler.write_info(
            self.config,
            status=status,
            examples_generated=self.logger.successful_sentence_count,
            batches_written=self.logger.batch_count,
        )

        self.logger.save()

    def save_batch(self, batch_examples, batch_processed, batch_skip_reasons, batch_start_time):
        batch_time = time() - batch_start_time
        target_examples = min(self.max_examples, self.input_handler.input_count)
        progress_percent = (self.logger.successful_sentence_count / target_examples) * 100

        print(f"Saving batch {self.logger.batch_count}, progress: {progress_percent:.2f}% "
              f"({self.logger.successful_sentence_count}/{target_examples}), "
              f"batch time: {batch_time:.2f}s, avg per sentence: {batch_time / batch_processed:.2f}s")

        self.saving_handler.save_batch(batch_examples)

        self.logger.log_generation_batch(
            batch_index=self.logger.batch_count,
            processed=batch_processed,
            successful=len(batch_examples),
            skipped=batch_processed - len(batch_examples),
            time_seconds=batch_time,
            skip_reasons=batch_skip_reasons if batch_skip_reasons else None,
        )

        return [], 0, {}, time()
