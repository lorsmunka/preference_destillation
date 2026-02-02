from time import sleep, time

from sentence_handler import SentenceHandler
from model_handler import ModelHandler
from saving_handler import SavingHandler
from shared import ExitListener, Logger, BATCH_SIZE, MAX_TRAINING_EXAMPLES

exit_listener = ExitListener()
sentence_handler = SentenceHandler()
model_handler = ModelHandler()
logger = Logger()
saving_handler = SavingHandler(logger)

batch_examples = []
batch_start_time = time()
batch_processed = 0
batch_skip_reasons = {}

for i in range(logger.processed_sentence_count, sentence_handler.sentence_count):
    if logger.successful_sentence_count >= MAX_TRAINING_EXAMPLES:
        print(f"Reached MAX_TRAINING_EXAMPLES limit ({MAX_TRAINING_EXAMPLES})")
        break

    sentence = sentence_handler.get_sentence(i)
    training_example, skip_reason = model_handler.generate_training_example(sentence)

    logger.processed_sentence_count += 1
    batch_processed += 1

    if training_example is not None:
        batch_examples.append(training_example)
        logger.successful_sentence_count += 1
    else:
        batch_skip_reasons[skip_reason] = batch_skip_reasons.get(skip_reason, 0) + 1

    if len(batch_examples) == BATCH_SIZE:
        batch_time = time() - batch_start_time
        target_examples = min(MAX_TRAINING_EXAMPLES, sentence_handler.sentence_count)
        progress_percent = (logger.successful_sentence_count / target_examples) * 100

        print(f"Saving batch {logger.batch_count}, progress: {progress_percent:.2f}% ({logger.successful_sentence_count}/{target_examples}), "
              f"batch time: {batch_time:.2f}s, avg per sentence: {batch_time / batch_processed:.2f}s")

        saving_handler.save_batch(batch_examples)

        logger.log_generation_batch(
            batch_index=logger.batch_count,
            processed=batch_processed,
            successful=BATCH_SIZE,
            skipped=batch_processed - BATCH_SIZE,
            time_seconds=batch_time,
            skip_reasons=batch_skip_reasons if batch_skip_reasons else None
        )

        batch_examples = []
        batch_processed = 0
        batch_skip_reasons = {}
        batch_start_time = time()

        if exit_listener.check_exit():
            logger.save()
            break

    sleep(0.1)

if batch_examples:
    saving_handler.save_batch(batch_examples)
    print(f"Saved final partial batch with {len(batch_examples)} examples")

logger.save()
exit_listener.stop()
print("Bye!")
