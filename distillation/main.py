from SentenceHandler import SentenceHandler
from TelemetryHandler import TelemetryHandler
from ModelHandler import ModelHandler

sentenceHandler = SentenceHandler()
telemetryHandler = TelemetryHandler()
modelHandler = ModelHandler()

for i in range(sentenceHandler.sentence_count):
    sentence = sentenceHandler.get_sentence(i)
    training_example = modelHandler.generate_training_example(sentence)
    telemetryHandler.processed_sentence_count += 1
