import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Dict, List, Optional, Tuple
from time import time

import math

from shared import (
    ExitListener,
    Logger,
    ClassificationAccuracyCalculator,
    MathAccuracyCalculator,
    PostGenerationAccuracyCalculator,
    get_device,
    PROMPT_DELIMITER,
    MINI_EVAL_FREQUENCY,
    MINI_EVAL_BATCH_COUNT,
)
from model import Transformer
from batch_handler import BatchHandler
from analysis.visualize_logs import update_training_plot


class Trainer:
    def __init__(self, model: Transformer, logger: Logger, exit_listener: ExitListener,
                 batch_handler: BatchHandler, config: dict):
        start_time = time()
        print("Initializing Trainer...")

        self.config = config
        self.domain = config['domain']
        self.learning_rate = config['learning_rate']
        self.epoch_count_value = config['epoch_count']
        self.kl_ratio_start = config['kl_ratio_start']
        self.kl_ratio_end = config['kl_ratio_end']
        self.distillation_temperature_start = config['distillation_temperature_start']
        self.distillation_temperature_end = config['distillation_temperature_end']
        self.lr_warmup_ratio = config['lr_warmup_ratio']
        self.checkpoints_dir = config.get('checkpoints_dir', './checkpoints')
        self.logs_dir = config.get('logs_dir', './logs')
        self.temp_checkpoint_path = os.path.join(self.checkpoints_dir, 'temp_checkpoint.pt')

        self.device = get_device()
        self.model = model.to(self.device)
        self.vocabulary = model.vocabulary
        self.vocab_size = model.vocabulary['vocab_size']
        self.output_token_to_index = {
            token: index
            for index, token in enumerate(self.vocabulary['token_list'])
        }
        self.tokenizer = model.tokenizer
        self.optimizer = AdamW(model.parameters(), lr=self.learning_rate)

        self.batch_handler = batch_handler
        batch_count = batch_handler.get_training_batches_radius()[1]
        self.total_training_steps = self.epoch_count_value * batch_count
        self.current_step = 0
        self.warmup_steps = int(self.total_training_steps * self.lr_warmup_ratio)
        self.lr_min = 1e-5

        self.mini_eval_frequency = config.get('mini_eval_frequency', MINI_EVAL_FREQUENCY)
        self.mini_eval_batch_count = config.get('mini_eval_batch_count', MINI_EVAL_BATCH_COUNT)

        self.logger = logger
        self.exit_listener = exit_listener

        elapsed_time = time() - start_time
        print(
            f"Trainer initialized on {self.device} -> took {elapsed_time:.2f} seconds (T_max={self.total_training_steps})")
        print(f"LR Warmup Steps: {self.warmup_steps}\n")

        if self.logger.should_resume():
            if os.path.exists(self.temp_checkpoint_path):
                self.load_checkpoint(self.temp_checkpoint_path)
                print("Loaded temp checkpoint\n")
            else:
                print("Warning: No temp checkpoint found, starting from scratch\n")

    def epoch_count(self):
        return self.epoch_count_value

    def train_epoch(self, batch_start: int, batch_end: int, epoch: int, resume_from_batch: int = 0) -> Optional[bool]:
        self.model.train()
        total_loss = 0.0
        total_kl_loss = 0.0
        total_ce_loss = 0.0
        total_steps = 0
        start_batch = max(batch_start, resume_from_batch)

        current_lr = self._get_current_learning_rate()
        print(
            f"Training Epoch {epoch} with starting learning rate: {current_lr:.6f}, current KL ratio: {self._get_current_kl_ratio():.6f}, current temperature: {self._get_current_temperature():.6f}\n ")

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

        for example_idx, example in enumerate(batch_data):
            example_loss, example_kl_loss, example_ce_loss, example_steps, example_correct = self._process_example(
                example, example_idx, len(batch_data), epoch, batch_idx)
            batch_loss += example_loss
            batch_kl_loss += example_kl_loss
            batch_ce_loss += example_ce_loss
            batch_steps += example_steps
            batch_correct += example_correct

        self.current_step += 1
        self._apply_learning_rate()
        current_lr = self._get_current_learning_rate()
        current_kl_ratio = self._get_current_kl_ratio()
        current_temperature = self._get_current_temperature()
        print(
            f"Batch {batch_idx + 1} -> LR: {current_lr:.8f}, KL ratio: {current_kl_ratio:.3f}, Temp: {current_temperature:.3f}")

        self._log_batch_completion(
            batch_idx, batch_steps, batch_loss, batch_kl_loss, batch_ce_loss, batch_correct, batch_start_time, epoch)

        if self.exit_listener.check_exit():
            avg = batch_loss / batch_steps if batch_steps > 0 else 0.0
            return self._handle_exit_request(epoch, avg)

        return batch_loss, batch_kl_loss, batch_ce_loss, batch_steps

    def _process_example(self, example, example_idx: int, total_examples: int, epoch: int, batch_idx: int):
        example_start_time = time()
        loss_sum, kl_loss_sum, ce_loss_sum, num_steps, correct = self.train_single_example(
            example)

        avg_loss = loss_sum / num_steps if num_steps > 0 else 0.0
        avg_kl_loss = kl_loss_sum / num_steps if num_steps > 0 else 0.0
        avg_ce_loss = ce_loss_sum / num_steps if num_steps > 0 else 0.0
        accuracy = correct / num_steps if num_steps > 0 else 0.0
        example_elapsed = time() - example_start_time
        # print(f"\tExample {example_idx + 1}/{total_examples}: {num_steps} steps, loss={avg_loss:.4f}, kl={avg_kl_loss:.4f}, ce={avg_ce_loss:.4f}, acc={accuracy:.4f} -> took {example_elapsed:.2f}s")

        return loss_sum, kl_loss_sum, ce_loss_sum, num_steps, correct

    def _log_batch_completion(self, batch_idx: int, batch_steps: int, batch_loss: float, batch_kl_loss: float, batch_ce_loss: float, batch_correct: int, batch_start_time: float, epoch: int):
        batch_elapsed = time() - batch_start_time
        avg_batch_loss = batch_loss / batch_steps if batch_steps > 0 else 0.0
        avg_batch_kl_loss = batch_kl_loss / batch_steps if batch_steps > 0 else 0.0
        avg_batch_ce_loss = batch_ce_loss / batch_steps if batch_steps > 0 else 0.0
        batch_accuracy = batch_correct / batch_steps if batch_steps > 0 else 0.0
        current_lr = self._get_current_learning_rate()
        current_kl_ratio = self._get_current_kl_ratio()
        current_temperature = self._get_current_temperature()

        print(f"Batch {batch_idx + 1} (of Epoch {epoch}) processed: {batch_steps} total steps, loss={avg_batch_loss:.4f}, kl={avg_batch_kl_loss:.4f}, ce={avg_batch_ce_loss:.4f}, accuracy={batch_accuracy:.4f} -> took {batch_elapsed:.2f}s\n")

        self.logger.log_training_batch(
            epoch=epoch,
            batch=batch_idx + 1,
            steps=batch_steps,
            loss=avg_batch_loss,
            kl_loss=avg_batch_kl_loss,
            ce_loss=avg_batch_ce_loss,
            accuracy=batch_accuracy,
            learning_rate=current_lr,
            kl_ratio=current_kl_ratio,
            temperature=current_temperature,
            time_seconds=batch_elapsed
        )

        if (batch_idx + 1) % 100 == 0:
            update_training_plot(self.logs_dir)
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
            update_training_plot(self.logs_dir)

        self.logger.update_progress(epoch, batch_idx + 1)

    def _run_mini_eval(self, batch_start: int, batch_end: int, epoch: int, current_batch: int):
        print(f"\n--- Mini-eval on test batches {batch_start + 1}-{batch_end} ---")
        self.model.eval()
        results = self._evaluate_batches(batch_start, batch_end)

        self.logger.log_mini_eval(
            epoch, current_batch, results['teacher_forced_accuracy'],
            results['student_accuracy'], results['classification_accuracy'],
            results['total_steps'])

        print(f"Mini-eval: TF={results['teacher_forced_accuracy']:.4f}, Student={results['student_accuracy']:.4f}, Classification={results['classification_accuracy']:.4f} ({results['total_steps']} steps)")
        print(f"--- Mini-eval done ---\n")

        self.model.train()

    def _handle_exit_request(self, epoch: int, avg_batch_loss: float):
        print("Exit requested. Saving progress...")
        self.save_checkpoint(epoch, avg_batch_loss, filepath=self.temp_checkpoint_path)
        self.logger.save()
        return None

    def train_single_example(self, example: Dict) -> Tuple[float, float, float, int, int]:
        sentence_tokens = self._get_sentence_tokens(example)
        steps = example['steps']
        num_steps = len(steps)

        if num_steps == 0:
            return 0.0, 0.0, 0.0, 0, 0

        all_token_ids = []
        all_target_logits = []
        all_target_indices = []
        for step in steps:
            token_id, target_logits, target_index = self._prepare_step_data(
                step)
            all_token_ids.append(token_id)
            all_target_logits.append(target_logits)
            all_target_indices.append(target_index)

        full_input_ids = sentence_tokens + all_token_ids[:-1]
        full_input_ids = self.model.remap_input_tokens(full_input_ids)
        input_tensor = torch.tensor(
            [full_input_ids], dtype=torch.long, device=self.device)
        target_logits_tensor = torch.tensor(
            all_target_logits, dtype=torch.float32, device=self.device)
        target_indices_tensor = torch.tensor(
            all_target_indices, dtype=torch.long, device=self.device)

        self.optimizer.zero_grad(set_to_none=True)
        model_logits = self.model(input_tensor)

        sentence_length = len(sentence_tokens)
        prediction_logits = model_logits[0, sentence_length -
                                         1:sentence_length - 1 + num_steps, :]

        kl_loss, ce_loss, total_loss = self._compute_loss(
            prediction_logits, target_logits_tensor, target_indices_tensor, reduction='sum')

        total_loss.backward()
        self.optimizer.step()

        correct_predictions = (torch.argmax(
            prediction_logits, dim=-1) == target_indices_tensor).sum().item()

        return total_loss.item(), kl_loss.item(), ce_loss.item(), num_steps, correct_predictions

    def _evaluate_batches(self, batch_start: int, batch_end: int, verbose: bool = False):
        if self.domain == "math_word_problem":
            task_accuracy_calculator = MathAccuracyCalculator()
        elif self.domain == "post_generation":
            task_accuracy_calculator = PostGenerationAccuracyCalculator()
        else:
            task_accuracy_calculator = ClassificationAccuracyCalculator()

        total_loss = 0.0
        total_kl_loss = 0.0
        total_ce_loss = 0.0
        total_teacher_forced_correct = 0
        total_student_correct = 0
        total_steps = 0

        with torch.no_grad():
            for batch_idx in range(batch_start, batch_end):
                batch_start_time = time()
                batch_data = self.batch_handler.get_batch(batch_idx)

                batch_loss = 0.0
                batch_kl_loss = 0.0
                batch_ce_loss = 0.0
                batch_teacher_forced_correct = 0
                batch_student_correct = 0
                batch_steps = 0

                for example in batch_data:
                    loss_sum, kl_loss_sum, ce_loss_sum, teacher_forced_correct, student_correct, num_steps, student_tokens = self._eval_single_example(example)
                    batch_loss += loss_sum
                    batch_kl_loss += kl_loss_sum
                    batch_ce_loss += ce_loss_sum
                    batch_teacher_forced_correct += teacher_forced_correct
                    batch_student_correct += student_correct
                    batch_steps += num_steps
                    total_loss += loss_sum
                    total_kl_loss += kl_loss_sum
                    total_ce_loss += ce_loss_sum
                    total_teacher_forced_correct += teacher_forced_correct
                    total_student_correct += student_correct
                    total_steps += num_steps

                    ground_truth_response = example.get('model_response', '')
                    task_accuracy_calculator.update(student_tokens, ground_truth_response)

                if verbose:
                    batch_elapsed = time() - batch_start_time
                    avg_batch_loss = batch_loss / batch_steps if batch_steps > 0 else 0.0
                    avg_batch_kl_loss = batch_kl_loss / batch_steps if batch_steps > 0 else 0.0
                    avg_batch_ce_loss = batch_ce_loss / batch_steps if batch_steps > 0 else 0.0
                    batch_tf_accuracy = batch_teacher_forced_correct / batch_steps if batch_steps > 0 else 0.0
                    batch_student_accuracy = batch_student_correct / batch_steps if batch_steps > 0 else 0.0
                    running_classification_accuracy = task_accuracy_calculator.get_accuracy()
                    print(f"Eval Batch {batch_idx + 1}: {batch_steps} steps, loss={avg_batch_loss:.4f}, kl={avg_batch_kl_loss:.4f}, ce={avg_batch_ce_loss:.4f}, tf_acc={batch_tf_accuracy:.4f}, student_acc={batch_student_accuracy:.4f}, class_acc={running_classification_accuracy:.4f} -> took {batch_elapsed:.2f}s")

        avg_loss = total_loss / total_steps if total_steps > 0 else 0.0
        avg_kl_loss = total_kl_loss / total_steps if total_steps > 0 else 0.0
        avg_ce_loss = total_ce_loss / total_steps if total_steps > 0 else 0.0
        teacher_forced_accuracy = total_teacher_forced_correct / total_steps if total_steps > 0 else 0.0
        student_accuracy = total_student_correct / total_steps if total_steps > 0 else 0.0
        classification_accuracy = task_accuracy_calculator.get_accuracy()
        confusion_matrices = task_accuracy_calculator.get_confusion_matrices()

        return {
            'avg_loss': avg_loss,
            'avg_kl_loss': avg_kl_loss,
            'avg_ce_loss': avg_ce_loss,
            'teacher_forced_accuracy': teacher_forced_accuracy,
            'student_accuracy': student_accuracy,
            'classification_accuracy': classification_accuracy,
            'confusion_matrices': confusion_matrices,
            'total_steps': total_steps,
        }

    def eval_epoch(self, batch_start: int, batch_end: int, epoch: int) -> Tuple[float, float, float, float]:
        self.model.eval()
        results = self._evaluate_batches(batch_start, batch_end, verbose=True)

        self.logger.log_eval_epoch(
            epoch, results['avg_loss'], results['teacher_forced_accuracy'],
            results['student_accuracy'], results['classification_accuracy'],
            results['confusion_matrices'], results['total_steps'],
            results['avg_kl_loss'], results['avg_ce_loss'])

        print(
            f"Eval Loss: {results['avg_loss']:.4f} | KL Loss: {results['avg_kl_loss']:.4f} | CE Loss: {results['avg_ce_loss']:.4f} | TF Accuracy: {results['teacher_forced_accuracy']:.4f} | Student Accuracy: {results['student_accuracy']:.4f} | Classification Accuracy: {results['classification_accuracy']:.4f}")
        return results['avg_loss'], results['teacher_forced_accuracy'], results['student_accuracy'], results['classification_accuracy']

    def _eval_single_example(self, example: Dict) -> Tuple[float, float, float, int, int, int, List[str]]:
        sentence_tokens = self._get_sentence_tokens(example)
        steps = example['steps']
        num_steps = len(steps)

        if num_steps == 0:
            return 0.0, 0.0, 0.0, 0, 0, 0, []

        all_token_ids = []
        all_target_logits = []
        all_target_indices = []
        for step in steps:
            token_id, target_logits, target_index = self._prepare_step_data(step)
            all_token_ids.append(token_id)
            all_target_logits.append(target_logits)
            all_target_indices.append(target_index)

        # Teacher-forced: single forward pass (all ground truth tokens as input)
        full_input_ids = sentence_tokens + all_token_ids[:-1]
        full_input_ids = self.model.remap_input_tokens(full_input_ids)
        input_tensor = torch.tensor([full_input_ids], dtype=torch.long, device=self.device)
        target_logits_tensor = torch.tensor(all_target_logits, dtype=torch.float32, device=self.device)
        target_indices_tensor = torch.tensor(all_target_indices, dtype=torch.long, device=self.device)

        model_logits = self.model(input_tensor)
        sentence_length = len(sentence_tokens)
        prediction_logits = model_logits[0, sentence_length - 1:sentence_length - 1 + num_steps, :]

        kl_loss, ce_loss, total_loss = self._compute_loss(
            prediction_logits, target_logits_tensor, target_indices_tensor, reduction='sum')

        teacher_forced_correct = (torch.argmax(prediction_logits, dim=-1) == target_indices_tensor).sum().item()

        # Student: sequential forward passes (own predictions as input)
        student_correct = 0
        remapped_sentence_tokens = self.model.remap_input_tokens(sentence_tokens)
        student_token_ids = []
        student_tokens = []

        for step_index in range(num_steps):
            student_input_tensor = torch.tensor(
                [remapped_sentence_tokens + student_token_ids], dtype=torch.long, device=self.device)
            student_logits = self.model(student_input_tensor)[:, -1, :]
            student_predicted_index = torch.argmax(student_logits[0]).item()
            student_predicted_token_id = self.model.output_token_ids[student_predicted_index]
            student_predicted_token = self.vocabulary['token_list'][student_predicted_index]
            if student_predicted_index == all_target_indices[step_index]:
                student_correct += 1
            student_token_ids.append(self.model.remap_input_tokens([student_predicted_token_id])[0])
            student_tokens.append(student_predicted_token)

        return total_loss.item(), kl_loss.item(), ce_loss.item(), teacher_forced_correct, student_correct, num_steps, student_tokens

    def _get_current_temperature(self) -> float:
        if self.total_training_steps <= 1:
            return self.distillation_temperature_start
        progress = self.current_step / self.total_training_steps
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return self.distillation_temperature_end + (self.distillation_temperature_start - self.distillation_temperature_end) * cosine_decay

    def _get_current_kl_ratio(self) -> float:
        if self.total_training_steps <= 1:
            return self.kl_ratio_start
        progress = self.current_step / self.total_training_steps
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return self.kl_ratio_end + (self.kl_ratio_start - self.kl_ratio_end) * cosine_decay

    def _get_current_learning_rate(self) -> float:
        if self.warmup_steps > 0 and self.current_step <= self.warmup_steps:
            return self.learning_rate * (self.current_step / self.warmup_steps)
        if self.total_training_steps <= self.warmup_steps:
            return self.learning_rate
        progress = (self.current_step - self.warmup_steps) / (self.total_training_steps - self.warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return self.lr_min + (self.learning_rate - self.lr_min) * cosine_decay

    def _apply_learning_rate(self):
        current_lr = self._get_current_learning_rate()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr

    def _get_sentence_tokens(self, example: Dict) -> List[int]:
        text = example['sentence'] + PROMPT_DELIMITER
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _prepare_step_data(self, step: Dict) -> Tuple[int, List[float], int]:
        token = step['token']
        logit_vector = step['logits'][:self.vocab_size]
        target_index = self.output_token_to_index.get(
            token, step['predicted_token_index'])

        token_ids = self.tokenizer.encode(token, add_special_tokens=False)
        token_id = token_ids[0] if token_ids else self.tokenizer.unk_token_id

        return token_id, logit_vector, target_index

    def _create_tensors(self, input_ids: List[int], target_logits: List[float]) -> Tuple[torch.Tensor, torch.Tensor]:
        input_tensor = torch.tensor(
            [input_ids], dtype=torch.long, device=self.device)
        target_tensor = torch.tensor(
            [target_logits], dtype=torch.float32, device=self.device)
        return input_tensor, target_tensor

    def _compute_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                       target_indices: torch.Tensor, reduction: str = 'mean'):
        temperature = self._get_current_temperature()
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

        kl_reduction = 'batchmean' if reduction == 'mean' else reduction
        kl_loss = F.kl_div(student_log_probs, teacher_probs,
                           reduction=kl_reduction) * (temperature ** 2)
        ce_loss = F.cross_entropy(
            student_logits / temperature, target_indices, reduction=reduction) * (temperature ** 2)

        kl_ratio = self._get_current_kl_ratio()
        combined_loss = kl_ratio * kl_loss + (1 - kl_ratio) * ce_loss

        return kl_loss, ce_loss, combined_loss

    def save_checkpoint(self, epoch: int, train_loss: float, filepath: str = None):
        start_time = time()

        if filepath is None:
            filepath = os.path.join(self.checkpoints_dir, f'checkpoint_epoch_{epoch}.pt')

        checkpoint = {
            'epoch': epoch,
            'current_step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
        }

        os.makedirs(self.checkpoints_dir, exist_ok=True)

        torch.save(checkpoint, filepath)

        elapsed_time = time() - start_time
        print(f"Checkpoint saved: {filepath} -> took {elapsed_time:.2f}s\n")

    def load_checkpoint(self, filepath: str) -> int:
        start_time = time()
        checkpoint = torch.load(
            filepath, map_location=self.device, weights_only=True)

        model_state = checkpoint['model_state_dict']
        model_state.pop('rotary_embedding.cos_cached', None)
        model_state.pop('rotary_embedding.sin_cached', None)
        self.model.load_state_dict(model_state, strict=False)

        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        batch_count = self.batch_handler.get_training_batches_radius()[1]
        epoch = checkpoint['epoch']

        if 'current_step' in checkpoint:
            self.current_step = checkpoint['current_step']
        else:
            self.current_step = epoch * batch_count

        self._apply_learning_rate()

        elapsed_time = time() - start_time
        print(
            f"Checkpoint loaded: {filepath} (epoch {epoch}, step {self.current_step}) -> took {elapsed_time:.2f}s\n")

        return epoch
