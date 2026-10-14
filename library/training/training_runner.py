import json
import os
from datetime import datetime

from library.training.batch_handler import BatchHandler
from library.model.transformer import Transformer
from library.training.trainer import Trainer
from library.shared.logging import RunLogger
from library.tooling.render.plots import plot_training_logs
from library.shared import (
    ExitListener,
    build_vocabulary,
    get_batches_dir,
    get_training_run_dir,
    load_input_vocabulary,
)
from library.shared.batch_files import vocabulary_fingerprint
from library.domain import get_domain
from library.shared.queue_file import write_queue
from library.specs import DEFAULT_MAX_TEST_EXAMPLES

INPUT_VOCABULARY_SNAPSHOT = "input_vocabulary.json"   # per-run copy, so a rebuild of the batches' one never changes a run


class TrainingRunner:
    def __init__(self, config, exit_listener: ExitListener):
        if isinstance(config, dict):  # legacy dict — resolve the domain name to its object
            self.config = config
            self.domain = get_domain(config["domain"])
        else:  # a TrainingRun spec — keep its Domain object, flatten the rest for info.json
            self.config = config.as_dict()
            self.domain = config.domain
        self.exit_listener = exit_listener
        self.run_name = self.config["run_name"]
        self.teacher_model = self.config["teacher_model"]
        self.run_dir = get_training_run_dir(self.run_name)
        self.checkpoints_dir = os.path.join(self.run_dir, "checkpoints")
        self.logs_dir = os.path.join(self.run_dir, "logs")

    def run(self) -> bool:
        """Trains the run to completion (True) or until a stop is requested (False)."""
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        trainer_config = dict(self.config)
        trainer_config["checkpoints_dir"] = self.checkpoints_dir
        trainer_config["logs_dir"] = self.logs_dir

        print(f"\n{'=' * 60}")
        print(f"Run: {self.run_name}")
        print(f"  Description: {self.config.get('description', '')}")
        print(f"  Domain: {self.domain.name}")
        print(f"  Teacher: {self.teacher_model}")
        print(f"  Architecture: {self.config['hidden_dim']}h, {self.config['num_layers']}L, {self.config['num_heads']} heads")
        print(f"  KL annealing: {self.config['kl_ratio_start']} -> {self.config['kl_ratio_end']}")
        print(f"  Temperature annealing: {self.config['distillation_temperature_start']} -> {self.config['distillation_temperature_end']}")
        print(f"  LR: {self.config['learning_rate']}, Epochs: {self.config['epoch_count']}")
        print(f"  Auxiliary token %: {self.config.get('auxiliary_token_percentage', 1.0)}")
        print(f"  Output: {self.run_dir}")
        print(f"{'=' * 60}\n")

        started_at = datetime.now().isoformat()

        transformer = Transformer(
            domain=self.domain,
            teacher_model=self.teacher_model,
            hidden_dim=self.config["hidden_dim"],
            num_layers=self.config["num_layers"],
            num_heads=self.config["num_heads"],
            dropout=self.config["dropout"],
            auxiliary_token_percentage=self.config.get("auxiliary_token_percentage", 1.0),
            input_vocabulary=self._input_vocabulary(),
        )
        # The batches' logit columns are the domain's full reduced vocabulary (auxiliary 1.0);
        # the student's output vocabulary is a prefix of it.
        teacher_token_list = build_vocabulary(transformer.tokenizer, self.domain)["token_list"]

        batch_handler = BatchHandler(
            get_batches_dir(self.domain.name, self.teacher_model),
            self.config["training_test_ratio"],
            max_training_examples=self.config.get("max_training_examples"),
            batch_size=self.config["batch_size"],
            max_test_examples=self.config.get("max_test_examples", DEFAULT_MAX_TEST_EXAMPLES),
            teacher_token_list=teacher_token_list,
            shuffle_seed=self.config.get("shuffle_seed"),
        )
        logger = RunLogger(self.logs_dir)

        self.model_info = transformer.get_model_info()
        self.data_info = {
            "split_rule": "fixed_tail",   # test = fixed tail of the batch directory (batch_handler.split_batch_files)
            "training_batch_count": batch_handler.training_batch_count(),
            "test_batch_count": batch_handler.test_batch_count(),
            "vocabulary_sha1": vocabulary_fingerprint(teacher_token_list),
        }
        self._write_info("in_progress", started_at=started_at)

        trainer = Trainer(transformer, logger, self.exit_listener, batch_handler, trainer_config)

        start_epoch = logger.current_epoch
        resume_batch = logger.current_batch

        for epoch in range(start_epoch, trainer.epoch_count()):
            resume_from = resume_batch if epoch == start_epoch else 0
            print(f"\nEpoch {epoch + 1}/{trainer.epoch_count()}\n")

            if not trainer.train_epoch(epoch, resume_from):
                self._write_info("in_progress", started_at=started_at, longest_sequence=trainer.longest_sequence)
                return False

            trainer.eval_epoch(epoch)
            plot_training_logs(self.logs_dir)
            trainer.save_epoch_end(epoch)

        trainer.delete_resume_checkpoint()
        completed_at = datetime.now().isoformat()
        self._write_info("completed", started_at=started_at, completed_at=completed_at,
                         longest_sequence=trainer.longest_sequence)
        print(f"\nRun '{self.run_name}' completed.\n")
        return True

    def _input_vocabulary(self):
        """The run's own snapshot when it has one (a resumed run keeps the vocabulary it
        started with), else the batches' input_vocabulary.json, snapshotted on first use."""
        snapshot_path = os.path.join(self.run_dir, INPUT_VOCABULARY_SNAPSHOT)
        if os.path.exists(snapshot_path):
            with open(snapshot_path, encoding="utf-8") as file:
                return json.load(file)
        input_vocabulary = load_input_vocabulary(self.domain.name, self.teacher_model)
        if input_vocabulary is not None:
            with open(snapshot_path, "w", encoding="utf-8") as file:
                json.dump(input_vocabulary, file)
        return input_vocabulary

    def _write_info(self, status, started_at=None, completed_at=None, longest_sequence=0):
        info = dict(self.config)
        info["optimizer_updates"] = "per_batch"   # runs without this key were updated after every example
        info["status"] = status
        info["started_at"] = started_at
        info["completed_at"] = completed_at
        if hasattr(self, "model_info"):
            info["model_info"] = self.model_info
        if hasattr(self, "data_info"):
            info.update(self.data_info)
        info_path = os.path.join(self.run_dir, "info.json")
        # the longest sequence trained on so far; kept across resumes (each session sees only its part)
        previous = 0
        if os.path.exists(info_path):
            with open(info_path, encoding="utf-8") as file:
                previous = json.load(file).get("longest_training_sequence", 0)
        if max(previous, longest_sequence):
            info["longest_training_sequence"] = max(previous, longest_sequence)
        with open(info_path, "w", encoding="utf-8") as file:
            json.dump(info, file, indent=2)


def is_completed(run_name: str) -> bool:
    """Whether this run already finished — the resumability check every entry script needs."""
    info_path = os.path.join(get_training_run_dir(run_name), "info.json")
    if not os.path.exists(info_path):
        return False
    with open(info_path, "r", encoding="utf-8") as file:
        return json.load(file).get("status") == "completed"


def write_training_queue(runs) -> None:
    """Announce the planned runs to the dashboard (queue/<experiment tag>.json)."""
    items = []
    for run in runs:
        spec = run if isinstance(run, dict) else run.as_dict()
        items.append({"name": spec["run_name"], "description": spec.get("description", ""),
                      "info_path": os.path.join(get_training_run_dir(spec["run_name"]), "info.json")})
    if items:
        first = runs[0] if isinstance(runs[0], dict) else runs[0].as_dict()
        write_queue(first.get("experiment") or "training", items, kind="training")


def run_training(runs, exit_listener: ExitListener = None) -> bool:
    """Train every run in the list, skipping the ones already completed.

    Returns True when every run is completed (trained now or skipped) and False when a stop
    was requested (Stop button / Ctrl+C), so a calling script can stop instead of going on
    to its analysis. The shared body of train.py and every experiments/*.py script — they
    differ only in which runs they define. Pass an `exit_listener` to share one across
    several calls; otherwise this owns and stops its own.
    """
    owns_listener = exit_listener is None
    exit_listener = exit_listener or ExitListener()
    write_training_queue(runs)
    all_completed = True
    for run in runs:
        run_name = run["run_name"] if isinstance(run, dict) else run.run_name
        if is_completed(run_name):
            print(f"Skipping completed run: {run_name}")
            continue
        if not TrainingRunner(run, exit_listener).run():
            all_completed = False
            break
    if owns_listener:
        exit_listener.stop()
    return all_completed
