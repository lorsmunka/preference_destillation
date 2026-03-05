from time import sleep, time
from datetime import datetime, timezone

from input_handler import InputHandler
from model_handler import ModelHandler
from saving_handler import SavingHandler
from shared import ExitListener, Logger, get_input_path, get_output_dir


class QueueRunner:
    def __init__(self, queue_element: dict, exit_listener: ExitListener):
        self.queue_element = queue_element
        self.exit_listener = exit_listener

        self.domain = queue_element["domain"]
        self.model_name = queue_element["model_name"]
        self.max_examples = queue_element["max_examples"]
        self.batch_size = queue_element["batch_size"]

        self.queue_element["started_at"] = datetime.now(timezone.utc).isoformat()

        input_path = get_input_path(self.domain)
        output_dir = get_output_dir(self.domain, self.model_name)

        self.input_handler = InputHandler(input_path)
        self.model_handler = ModelHandler(self.model_name, self.domain)
        self.logger = Logger()
        self.saving_handler = SavingHandler(self.logger, output_dir)

    def run(self):
        self.saving_handler.write_info(self.queue_element)

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
                batch_skip_reasons[skip_reason] = batch_skip_reasons.get(
                    skip_reason, 0) + 1

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

        if self.logger.successful_sentence_count >= self.max_examples:
            status = "completed"
        else:
            status = "incomplete"

        self.saving_handler.write_info(
            self.queue_element,
            status=status,
            examples_generated=self.logger.successful_sentence_count,
            batches_written=self.logger.batch_count,
        )

        self.logger.save()

    def save_batch(self, batch_examples, batch_processed, batch_skip_reasons, batch_start_time):
        batch_time = time() - batch_start_time
        target_examples = min(self.max_examples, self.input_handler.input_count)
        progress_percent = (
            self.logger.successful_sentence_count / target_examples) * 100

        print(f"Saving batch {self.logger.batch_count}, progress: {progress_percent:.2f}% ({self.logger.successful_sentence_count}/{target_examples}), "
              f"batch time: {batch_time:.2f}s, avg per sentence: {batch_time / batch_processed:.2f}s")

        self.saving_handler.save_batch(batch_examples)

        self.logger.log_generation_batch(
            batch_index=self.logger.batch_count,
            processed=batch_processed,
            successful=self.batch_size,
            skipped=batch_processed - self.batch_size,
            time_seconds=batch_time,
            skip_reasons=batch_skip_reasons if batch_skip_reasons else None
        )

        return [], 0, {}, time()
