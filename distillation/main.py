from time import sleep
from SentenceHandler import SentenceHandler
from TelemetryHandler import TelemetryHandler
from ModelHandler import ModelHandler
from SavingHandler import SavingHandler

sentenceHandler = SentenceHandler()
telemetryHandler = TelemetryHandler()
modelHandler = ModelHandler()
savingHandler = SavingHandler(telemetryHandler)

BATCH_SIZE = 32

batch_examples = []
for i in range(sentenceHandler.sentence_count):
    sentence = sentenceHandler.get_sentence(i)
    training_example = modelHandler.generate_training_example(sentence)
    telemetryHandler.processed_sentence_count += 1

    batch_examples.append(training_example)

    if len(batch_examples) == BATCH_SIZE:
        savingHandler.save_batch(batch_examples)
        batch_examples = []

    sleep(0.1)
