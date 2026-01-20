from time import sleep, time

from SentenceHandler import SentenceHandler
from ModelHandler import ModelHandler
from SavingHandler import SavingHandler
from shared import ExitListener, TelemetryHandler, BATCH_SIZE

exitListener = ExitListener()
sentenceHandler = SentenceHandler()
modelHandler = ModelHandler()
telemetryHandler = TelemetryHandler()
savingHandler = SavingHandler(telemetryHandler)

batch_examples = []
batch_start_time = time()
for i in range(telemetryHandler.processed_sentence_count, sentenceHandler.sentence_count):
    sentence = sentenceHandler.get_sentence(i)
    training_example = modelHandler.generate_training_example(sentence)
    telemetryHandler.processed_sentence_count += 1

    if training_example is not None:
        batch_examples.append(training_example)
        telemetryHandler.successful_sentence_count += 1

    if len(batch_examples) == BATCH_SIZE:
        progress_percent = (
            telemetryHandler.processed_sentence_count / sentenceHandler.sentence_count) * 100

        print(
            f"Saving batch, progress: {progress_percent:.2f}%, elapsed time for batch: {time() - batch_start_time:.2f} seconds, average time per sentence: {(time() - batch_start_time) / BATCH_SIZE:.2f} seconds")
        batch_start_time = time()

        savingHandler.save_batch(batch_examples)
        batch_examples = []

        if exitListener.check_exit():
            telemetryHandler.save()
            break

    sleep(0.1)

exitListener.stop()
print("Bye!")
