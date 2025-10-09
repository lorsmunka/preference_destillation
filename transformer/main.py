from transformers import AutoTokenizer
from BatchHandler import BatchHandler
from TransformerModelHandler import TransformerModelHandler
from TrainingHandler import TrainingHandler
from Utilities import Utilities


tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
vocabulary_map = Utilities.build_vocabulary_map(tokenizer)
output_token_ids = list(vocabulary_map.values())

batchHandler = BatchHandler()
transformerModelHandler = TransformerModelHandler(
    input_vocab_size=tokenizer.vocab_size,
    output_vocab_size=tokenizer.vocab_size,
    output_token_ids=output_token_ids,
)
trainingHandler = TrainingHandler(transformerModelHandler, vocabulary_map)

batch_start, batch_end = batchHandler.get_training_batches_radius()
test_start, test_end = batchHandler.get_test_batches_radius()

for epoch in range(trainingHandler.epoch_count()):
    print(f"\nEpoch {epoch + 1}/{trainingHandler.epoch_count()}")

    train_loss = trainingHandler.train_epoch(
        batchHandler, batch_start, batch_end)
    print(f"Train Loss: {train_loss:.4f}")

    eval_loss, eval_accuracy = trainingHandler.eval_epoch(
        batchHandler, test_start, test_end)
    print(f"Eval Loss: {eval_loss:.4f} | Accuracy: {eval_accuracy:.4f}")

    trainingHandler.save_checkpoint(
        f"checkpoint_epoch_{epoch+1}.pt", epoch+1, train_loss)
