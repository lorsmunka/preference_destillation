from BatchHandler import BatchHandler
from Transformer import Transformer
from Trainer import Trainer
from TelemetryHandler import TelemetryHandler
from ExitListener import ExitListener


transformer = Transformer()
telemetryHandler = TelemetryHandler()
trainer = Trainer(
    transformer, transformer.vocabulary_map, transformer.tokenizer, telemetryHandler)
batchHandler = BatchHandler()
exitListener = ExitListener()

batch_start, batch_end = batchHandler.get_training_batches_radius()
test_start, test_end = batchHandler.get_test_batches_radius()

start_epoch = telemetryHandler.current_epoch
resume_batch = telemetryHandler.current_batch


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
