from time import sleep, time

from sentence_handler import SentenceHandler
from model_handler import ModelHandler
from saving_handler import SavingHandler
from shared import ExitListener, Logger, BATCH_SIZE

exit_listener = ExitListener()
sentence_handler = SentenceHandler()
model_handler = ModelHandler()
logger = Logger()
saving_handler = SavingHandler(logger)

batch_examples = []
batch_start_time = time()
for i in range(logger.processed_sentence_count, sentence_handler.sentence_count):
    sentence = sentence_handler.get_sentence(i)
    training_example = model_handler.generate_training_example(sentence)
    logger.processed_sentence_count += 1

    if training_example is not None:
        batch_examples.append(training_example)
        logger.successful_sentence_count += 1

    if len(batch_examples) == BATCH_SIZE:
        progress_percent = (
            logger.processed_sentence_count / sentence_handler.sentence_count) * 100

        print(
            f"Saving batch, progress: {progress_percent:.2f}%, elapsed time for batch: {time() - batch_start_time:.2f} seconds, average time per sentence: {(time() - batch_start_time) / BATCH_SIZE:.2f} seconds")
        batch_start_time = time()

        saving_handler.save_batch(batch_examples)
        batch_examples = []

        if exit_listener.check_exit():
            logger.save()
            break

    sleep(0.1)

exit_listener.stop()
print("Bye!")
