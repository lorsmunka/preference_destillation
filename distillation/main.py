from time import sleep
from SentenceHandler import SentenceHandler
from TelemetryHandler import TelemetryHandler
from ModelHandler import ModelHandler
from SavingHandler import SavingHandler
from ExitListener import ExitListener

exitListener = ExitListener()
sentenceHandler = SentenceHandler()
modelHandler = ModelHandler()
telemetryHandler = TelemetryHandler()
savingHandler = SavingHandler(telemetryHandler)

BATCH_SIZE = 32

batch_examples = []
for i in range(telemetryHandler.processed_sentence_count, sentenceHandler.sentence_count):
    sentence = sentenceHandler.get_sentence(i)
    training_example = modelHandler.generate_training_example(sentence)
    telemetryHandler.processed_sentence_count += 1

    batch_examples.append(training_example)

    if len(batch_examples) == BATCH_SIZE:
        progress_percent = (
            telemetryHandler.processed_sentence_count / sentenceHandler.sentence_count) * 100
        print(f"Saving batch, progress: {progress_percent:.2f}%")

        savingHandler.save_batch(batch_examples)
        batch_examples = []

        if exitListener.check_exit():
            telemetryHandler.save()
            break

    sleep(0.1)

exitListener.stop()
print("Bye!")
