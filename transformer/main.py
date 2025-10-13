from transformers import AutoTokenizer
import os
from BatchHandler import BatchHandler
from Transformer import Transformer
from Trainer import Trainer
from TelemetryHandler import TelemetryHandler
from ExitListener import ExitListener
from Utilities import Utilities


tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
vocabulary_map = Utilities.build_vocabulary_map(tokenizer)
output_token_ids = list(vocabulary_map.values())

batchHandler = BatchHandler()
transformerModelHandler = Transformer(
    input_vocab_size=tokenizer.vocab_size,
    output_vocab_size=len(output_token_ids),
    output_token_ids=output_token_ids,
)

trainer = Trainer(
    transformerModelHandler, vocabulary_map, tokenizer)

telemetryHandler = TelemetryHandler()
exitListener = ExitListener()

batch_start, batch_end = batchHandler.get_training_batches_radius()
test_start, test_end = batchHandler.get_test_batches_radius()

start_epoch = telemetryHandler.current_epoch
resume_batch = telemetryHandler.current_batch

if telemetryHandler.should_resume():
    print(f"Resuming from epoch {start_epoch + 1}, batch {resume_batch}")
    temp_checkpoint_path = 'checkpoints/temp_checkpoint.pt'
    if os.path.exists(temp_checkpoint_path):
        trainer.load_checkpoint(temp_checkpoint_path)
        print("Loaded temp checkpoint\n")
    else:
        print("Warning: No temp checkpoint found, starting from scratch\n")
else:
    print("Starting fresh training\n")

for epoch in range(start_epoch, trainer.epoch_count()):
    print(f"\nEpoch {epoch + 1}/{trainer.epoch_count()}")

    resume_from = resume_batch if epoch == start_epoch else 0

    train_loss = trainer.train_epoch(
        batchHandler, batch_start, batch_end, epoch + 1, telemetryHandler, exitListener, resume_from)

    if train_loss is None:
        print("Training interrupted by user.")
        break

    print(f"Train Loss: {train_loss:.4f}")

    telemetryHandler.current_batch = 0

    eval_loss, eval_accuracy = trainer.eval_epoch(
        batchHandler, test_start, test_end, epoch + 1, telemetryHandler)
    print(f"Eval Loss: {eval_loss:.4f} | Accuracy: {eval_accuracy:.4f}")

    trainer.save_checkpoint(epoch + 1, train_loss)
    telemetryHandler.save()


exitListener.stop()
print("Bye!")
