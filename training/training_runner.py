import json
import os
from datetime import datetime

from batch_handler import BatchHandler
from model import Transformer
from trainer import Trainer
from shared import (
    ExitListener,
    Logger,
    get_batches_dir,
    get_training_run_dir,
)


class TrainingRunner:
    def __init__(self, config: dict, exit_listener: ExitListener):
        self.config = config
        self.exit_listener = exit_listener
        self.run_name = config["run_name"]
        self.domain = config["domain"]
        self.teacher_model = config["teacher_model"]
        self.run_dir = get_training_run_dir(self.run_name)
        self.checkpoints_dir = os.path.join(self.run_dir, "checkpoints")
        self.logs_dir = os.path.join(self.run_dir, "logs")

    def run(self) -> bool:
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        trainer_config = dict(self.config)
        trainer_config["checkpoints_dir"] = self.checkpoints_dir
        trainer_config["logs_dir"] = self.logs_dir

        print(f"\n{'=' * 60}")
        print(f"Run: {self.run_name}")
        print(f"  Description: {self.config.get('description', '')}")
        print(f"  Domain: {self.domain}")
        print(f"  Teacher: {self.teacher_model}")
        print(f"  Architecture: {self.config['hidden_dim']}h, {self.config['num_layers']}L, {self.config['num_heads']} heads")
        print(f"  KL annealing: {self.config['kl_ratio_start']} -> {self.config['kl_ratio_end']}")
        print(f"  LR: {self.config['learning_rate']}, Epochs: {self.config['epoch_count']}")
        print(f"  Auxiliary token %: {self.config.get('auxiliary_token_percentage', 1.0)}")
        print(f"  Output: {self.run_dir}")
        print(f"{'=' * 60}\n")

        started_at = datetime.now().isoformat()
        self._write_info("in_progress", started_at=started_at)

        batches_dir = get_batches_dir(self.domain, self.teacher_model)
        batch_handler = BatchHandler(batches_dir, self.config["training_test_ratio"])
        logger = Logger(self.logs_dir)

        transformer = Transformer(
            domain=self.domain,
            teacher_model=self.teacher_model,
            hidden_dim=self.config["hidden_dim"],
            num_layers=self.config["num_layers"],
            num_heads=self.config["num_heads"],
            dropout=self.config["dropout"],
            auxiliary_token_percentage=self.config.get("auxiliary_token_percentage", 1.0),
        )

        trainer = Trainer(transformer, logger, self.exit_listener, batch_handler, trainer_config)

        batch_start, batch_end = batch_handler.get_training_batches_radius()
        test_start, test_end = batch_handler.get_test_batches_radius()
        start_epoch = logger.current_epoch
        resume_batch = logger.current_batch

        for epoch in range(start_epoch, trainer.epoch_count()):
            resume_from = resume_batch if epoch == start_epoch else 0
            print(f"\nEpoch {epoch + 1}/{trainer.epoch_count()}\n")

            should_continue = trainer.train_epoch(
                batch_start, batch_end, epoch, resume_from)

            if not should_continue:
                return False

            trainer.eval_epoch(test_start, test_end, epoch)
            logger.save()

        completed_at = datetime.now().isoformat()
        self._write_info("completed", started_at=started_at, completed_at=completed_at)
        print(f"\nRun '{self.run_name}' completed.\n")
        return True

    def _write_info(self, status, started_at=None, completed_at=None):
        info = dict(self.config)
        info["status"] = status
        info["started_at"] = started_at
        info["completed_at"] = completed_at
        info_path = os.path.join(self.run_dir, "info.json")
        with open(info_path, "w", encoding="utf-8") as file:
            json.dump(info, file, indent=2)
