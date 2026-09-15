"""The training loop. Everything that is not the loop itself lives beside it:
schedules.py (annealed dials), losses.py (the distillation loss), example_encoder.py
(example -> tensors), evaluator.py (eval passes), checkpoints.py (save / load)."""

import os
from time import time
from typing import Dict, Optional, Tuple

import torch
from torch.optim import AdamW

from library.shared import ExitListener, MINI_EVAL_FREQUENCY, MINI_EVAL_BATCH_COUNT
from library.shared.device import get_device
from library.model.transformer import Transformer
from library.shared.logging import RunLogger
from library.tooling.render.plots import plot_training_logs
from library.training.batch_handler import BatchHandler
from library.training.checkpoints import load_checkpoint, save_checkpoint
from library.training.evaluator import Evaluator
from library.training.example_encoder import ExampleEncoder
from library.training.losses import DistillationLoss
from library.training.schedules import Schedules


class Trainer:
    def __init__(self, model: Transformer, logger: RunLogger, exit_listener: ExitListener,
                 batch_handler: BatchHandler, config: dict):
        start_time = time()
        print("Initializing Trainer...")

        self.config = config
        self.domain = model.domain  # the Domain object the model was built from (used directly)
        self.epoch_count_value = config['epoch_count']
        self.checkpoints_dir = config.get('checkpoints_dir', './checkpoints')
        self.logs_dir = config.get('logs_dir', './logs')
        self.temp_checkpoint_path = os.path.join(self.checkpoints_dir, 'temp_checkpoint.pt')

        self.device = get_device()
        self.model = model.to(self.device)
        self.optimizer = AdamW(model.parameters(), lr=config['learning_rate'])

        self.batch_handler = batch_handler
        batch_count = batch_handler.get_training_batches_radius()[1]
        self.total_training_steps = self.epoch_count_value * batch_count
        self.current_step = 0

        self.schedules = Schedules(config, steps_per_epoch=batch_count,
                                   total_training_steps=self.total_training_steps)
        self.loss = DistillationLoss(
            divergence=config.get('divergence', 'forward_kl'),
            entropy_kl_weighting=config.get('entropy_kl_weighting', 0.0))
        self.encoder = ExampleEncoder(self.model, self.device)
        self.evaluator = Evaluator(
            self.model, self.domain, self.encoder, batch_handler, self.device,
            loss_function=self._compute_loss,
            eval_top_k=config.get('eval_top_k', 20),
            eval_cap_multiple=config.get('eval_cap_multiple', 2.0))

        self.mini_eval_frequency = config.get('mini_eval_frequency', MINI_EVAL_FREQUENCY)
        self.mini_eval_batch_count = config.get('mini_eval_batch_count', MINI_EVAL_BATCH_COUNT)
        self.eval_top_k = self.evaluator.eval_top_k

        self.logger = logger
        self.exit_listener = exit_listener

        elapsed_time = time() - start_time
        print(
            f"Trainer initialized on {self.device} -> took {elapsed_time:.2f} seconds (T_max={self.total_training_steps})")
        print(f"LR Warmup Steps: {self.schedules.warmup_steps}\n")

        if self.logger.should_resume():
            if os.path.exists(self.temp_checkpoint_path):
                self.load_checkpoint(self.temp_checkpoint_path)
                print("Loaded temp checkpoint\n")
            else:
                print("Warning: No temp checkpoint found, starting from scratch\n")

    def epoch_count(self):
        return self.epoch_count_value

    # ── training ──────────────────────────────────────────────────────

    def train_epoch(self, batch_start: int, batch_end: int, epoch: int, resume_from_batch: int = 0) -> Optional[bool]:
        self.model.train()
        total_loss = 0.0
        total_kl_loss = 0.0
        total_ce_loss = 0.0
        total_steps = 0
        start_batch = max(batch_start, resume_from_batch)

        print(
            f"Training Epoch {epoch} with starting learning rate: {self._get_current_learning_rate():.6f}, current KL ratio: {self._get_current_kl_ratio():.6f}, current temperature: {self._get_current_temperature():.6f}\n ")

        for batch_idx in range(start_batch, batch_end):
            batch_result = self._process_batch(batch_idx, epoch)

            if batch_result is None:
                return None

            batch_loss, batch_kl_loss, batch_ce_loss, batch_steps = batch_result
            total_loss += batch_loss
            total_kl_loss += batch_kl_loss
            total_ce_loss += batch_ce_loss
            total_steps += batch_steps

        if total_steps > 0:
            avg_epoch_loss = total_loss / total_steps
            avg_epoch_kl_loss = total_kl_loss / total_steps
            avg_epoch_ce_loss = total_ce_loss / total_steps
            self.logger.log_train_epoch(
                epoch, avg_epoch_loss, total_steps, avg_epoch_kl_loss, avg_epoch_ce_loss)
            self.save_checkpoint(epoch, avg_epoch_loss)
        else:
            print(
                f"No training steps completed in epoch {epoch} (exit requested)")
            self.logger.current_batch = 0

        return True

    def _process_batch(self, batch_idx: int, epoch: int):
        batch_start_time = time()
        batch_data = self.batch_handler.get_batch(batch_idx)

        batch_loss = 0.0
        batch_kl_loss = 0.0
        batch_ce_loss = 0.0
        batch_steps = 0
        batch_correct = 0

        for example in batch_data:
            example_loss, example_kl_loss, example_ce_loss, example_steps, example_correct = self.train_single_example(example)
            batch_loss += example_loss
            batch_kl_loss += example_kl_loss
            batch_ce_loss += example_ce_loss
            batch_steps += example_steps
            batch_correct += example_correct

        self.current_step += 1
        self._apply_learning_rate()
        print(
            f"Batch {batch_idx + 1} -> LR: {self._get_current_learning_rate():.8f}, KL ratio: {self._get_current_kl_ratio():.3f}, Temp: {self._get_current_temperature():.3f}")

        self._log_batch_completion(
            batch_idx, batch_steps, batch_loss, batch_kl_loss, batch_ce_loss, batch_correct, batch_start_time, epoch)

        if self.exit_listener.check_exit():
            avg = batch_loss / batch_steps if batch_steps > 0 else 0.0
            return self._handle_exit_request(epoch, avg)

        return batch_loss, batch_kl_loss, batch_ce_loss, batch_steps

    def train_single_example(self, example: Dict) -> Tuple[float, float, float, int, int]:
        if len(example['steps']) == 0:
            return 0.0, 0.0, 0.0, 0, 0

        encoded = self.encoder.encode(example)
        num_steps = encoded.num_steps

        self.optimizer.zero_grad(set_to_none=True)
        self.model.set_attention_noise(self._get_current_attention_noise())
        model_logits = self.model(encoded.input_tensor)

        sentence_length = len(encoded.sentence_tokens)
        prediction_logits = model_logits[0, sentence_length - 1:sentence_length - 1 + num_steps, :]

        kl_loss, ce_loss, total_loss = self._compute_loss(
            prediction_logits, encoded.target_logits_tensor, encoded.target_indices_tensor, reduction='sum')

        total_loss.backward()
        self.optimizer.step()

        correct_predictions = (torch.argmax(prediction_logits, dim=-1) == encoded.target_indices_tensor).sum().item()

        return total_loss.item(), kl_loss.item(), ce_loss.item(), num_steps, correct_predictions

    def _log_batch_completion(self, batch_idx: int, batch_steps: int, batch_loss: float, batch_kl_loss: float, batch_ce_loss: float, batch_correct: int, batch_start_time: float, epoch: int):
        batch_elapsed = time() - batch_start_time
        avg_batch_loss = batch_loss / batch_steps if batch_steps > 0 else 0.0
        avg_batch_kl_loss = batch_kl_loss / batch_steps if batch_steps > 0 else 0.0
        avg_batch_ce_loss = batch_ce_loss / batch_steps if batch_steps > 0 else 0.0
        batch_accuracy = batch_correct / batch_steps if batch_steps > 0 else 0.0

        print(f"Batch {batch_idx + 1} (of Epoch {epoch}) processed: {batch_steps} total steps, loss={avg_batch_loss:.4f}, kl={avg_batch_kl_loss:.4f}, ce={avg_batch_ce_loss:.4f}, accuracy={batch_accuracy:.4f} -> took {batch_elapsed:.2f}s\n")

        self.logger.log_training_batch(
            epoch=epoch,
            batch=batch_idx + 1,
            steps=batch_steps,
            loss=avg_batch_loss,
            kl_loss=avg_batch_kl_loss,
            ce_loss=avg_batch_ce_loss,
            accuracy=batch_accuracy,
            learning_rate=self._get_current_learning_rate(),
            kl_ratio=self._get_current_kl_ratio(),
            temperature=self._get_current_temperature(),
            time_seconds=batch_elapsed,
            teacher_temperature=self._get_current_teacher_temperature(),
            attention_noise=self._get_current_attention_noise(),
        )

        if (batch_idx + 1) % 100 == 0:
            plot_training_logs(self.logs_dir)
        else:
            until_update = 100 - ((batch_idx + 1) % 100)
            mini_eval_part = ""
            if self.mini_eval_frequency > 0:
                until_mini_eval = self.mini_eval_frequency - ((batch_idx + 1) % self.mini_eval_frequency)
                mini_eval_part = f", {until_mini_eval} until mini-eval"
            print(f"{until_update} until next update{mini_eval_part}\n")

        if self.mini_eval_frequency > 0 and (batch_idx + 1) % self.mini_eval_frequency == 0:
            test_start, test_end = self.batch_handler.get_test_batches_radius()
            total_test_batches = test_end - test_start
            mini_eval_count = min(self.mini_eval_batch_count, total_test_batches)
            self._run_mini_eval(test_start, test_start + mini_eval_count, epoch, batch_idx + 1)
            plot_training_logs(self.logs_dir)

        self.logger.update_progress(epoch, batch_idx + 1)

    def _handle_exit_request(self, epoch: int, avg_batch_loss: float):
        print("Exit requested. Saving progress...")
        self.save_checkpoint(epoch, avg_batch_loss, filepath=self.temp_checkpoint_path, include_optimizer=True)
        self.logger.save()
        return None

    # ── evaluation ────────────────────────────────────────────────────

    def eval_epoch(self, batch_start: int, batch_end: int, epoch: int) -> Tuple[float, float, float, float]:
        self.model.eval()
        results = self.evaluator.evaluate_batches(batch_start, batch_end, verbose=True)

        self.logger.log_eval_epoch(
            epoch, results['avg_loss'], results['teacher_forced_accuracy'],
            results['student_accuracy'], results['task_accuracy'],
            results['confusion_matrices'], results['total_steps'],
            results['avg_kl_loss'], results['avg_ce_loss'],
            distribution=results['distribution'])

        dist = results['distribution']
        print(
            f"Eval Loss: {results['avg_loss']:.4f} | KL Loss: {results['avg_kl_loss']:.4f} | CE Loss: {results['avg_ce_loss']:.4f} | TF Accuracy: {results['teacher_forced_accuracy']:.4f} | Student Accuracy: {results['student_accuracy']:.4f} | Task Accuracy: {results['task_accuracy']:.4f}")
        print(
            f"  top{self.eval_top_k}={dist['topk_accuracy']:.4f} overlap={dist['teacher_student_overlap']:.4f} mean_rank={dist['mean_target_rank']:.1f} teacher_H={dist['teacher_entropy']:.3f} student_H={dist['student_entropy']:.3f} ppl={dist['perplexity']:.3f} termination={dist['termination_rate']:.4f}")
        return results['avg_loss'], results['teacher_forced_accuracy'], results['student_accuracy'], results['task_accuracy']

    def _run_mini_eval(self, batch_start: int, batch_end: int, epoch: int, current_batch: int):
        print(f"\n--- Mini-eval on test batches {batch_start + 1}-{batch_end} ---")
        self.model.eval()
        results = self.evaluator.evaluate_batches(batch_start, batch_end)

        self.logger.log_mini_eval(
            epoch, current_batch, results['teacher_forced_accuracy'],
            results['student_accuracy'], results['task_accuracy'],
            results['total_steps'], distribution=results['distribution'])

        print(f"Mini-eval: TF={results['teacher_forced_accuracy']:.4f}, Student={results['student_accuracy']:.4f}, Task={results['task_accuracy']:.4f} ({results['total_steps']} steps)")
        print(f"--- Mini-eval done ---\n")

        self.model.train()

    # ── loss and schedules at the current step ────────────────────────

    def _compute_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                      target_indices: torch.Tensor, reduction: str = 'mean'):
        return self.loss.compute(
            student_logits, teacher_logits, target_indices,
            student_temperature=self._get_current_temperature(),
            teacher_temperature=self._get_current_teacher_temperature(),
            kl_ratio=self._get_current_kl_ratio(),
            reduction=reduction)

    def _get_current_temperature(self) -> float:
        return self.schedules.student_temperature(self.current_step)

    def _get_current_teacher_temperature(self) -> float:
        return self.schedules.teacher_temperature(self.current_step)

    def _get_current_kl_ratio(self) -> float:
        return self.schedules.kl_ratio(self.current_step)

    def _get_current_attention_noise(self) -> float:
        return self.schedules.attention_noise(self.current_step)

    def _get_current_learning_rate(self) -> float:
        return self.schedules.learning_rate(self.current_step)

    def _apply_learning_rate(self):
        current_lr = self._get_current_learning_rate()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr

    # ── checkpoints ───────────────────────────────────────────────────

    def save_checkpoint(self, epoch: int, train_loss: float, filepath: str = None,
                        include_optimizer: bool = False):
        if filepath is None:
            filepath = os.path.join(self.checkpoints_dir, f'checkpoint_epoch_{epoch}.pt')
        save_checkpoint(filepath, self.model, self.optimizer, epoch, self.current_step,
                        train_loss, include_optimizer=include_optimizer)

    def load_checkpoint(self, filepath: str) -> int:
        start_time = time()
        checkpoint = load_checkpoint(filepath, self.model, self.device)

        self.optimizer = AdamW(self.model.parameters(), lr=self.schedules.learning_rate_peak)
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint['epoch']
        if 'current_step' in checkpoint:
            self.current_step = checkpoint['current_step']
        else:
            self.current_step = epoch * self.schedules.steps_per_epoch

        self._apply_learning_rate()
        print(
            f"Checkpoint loaded: {filepath} (epoch {epoch}, step {self.current_step}) -> took {time() - start_time:.2f}s\n")
        return epoch
