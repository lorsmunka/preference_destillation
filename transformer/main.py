from BatchHandler import BatchHandler
from Transformer import Transformer
from Trainer import Trainer
from TelemetryHandler import TelemetryHandler
from ExitListener import ExitListener


transformer = Transformer()
telemetryHandler = TelemetryHandler()
exitListener = ExitListener()
trainer = Trainer(
    transformer, transformer.vocabulary_map, transformer.tokenizer, telemetryHandler, exitListener)
batchHandler = BatchHandler()

batch_start, batch_end = batchHandler.get_training_batches_radius()
test_start, test_end = batchHandler.get_test_batches_radius()

start_epoch = telemetryHandler.current_epoch
resume_batch = telemetryHandler.current_batch


for epoch in range(start_epoch, trainer.epoch_count() + 1):
    resume_from = resume_batch if epoch == start_epoch else 0
    print(f"\nEpoch {epoch}/{trainer.epoch_count()}\n")

    should_continue = trainer.train_epoch(
        batchHandler, batch_start, batch_end, epoch, resume_from)

    if not should_continue:
        break

    trainer.eval_epoch(
        batchHandler, test_start, test_end, epoch)

    telemetryHandler.save()


exitListener.stop()
print("Bye!")
