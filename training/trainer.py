import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, List, Optional, Tuple
from time import time

import math

from shared import (
    ExitListener,
    Logger,
    get_device,
    EPOCH_COUNT,
    LEARNING_RATE,
    DISTILLATION_TEMPERATURE,
    KL_RATIO_START,
    KL_RATIO_END,
    TEMP_CHECKPOINT_PATH,
)
from model import Transformer
from batch_handler import BatchHandler


class Trainer:
    def __init__(self, model: Transformer, vocabulary: dict, tokenizer, logger: Logger, exit_listener: ExitListener, batch_handler: BatchHandler):
        start_time = time()
        print("Initializing Trainer...")

        self.device = get_device()
        self.model = model.to(self.device)
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

        self.batch_handler = batch_handler
        batch_count = batch_handler.get_training_batches_radius()[1]
        self.total_training_steps = EPOCH_COUNT * batch_count
        self.current_step = 0

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.total_training_steps,
            eta_min=1e-5
        )

        self.vocab_size = vocabulary['vocab_size']
        self.token_to_index = vocabulary['token_to_index']

        self.logger = logger
        self.exit_listener = exit_listener

        elapsed_time = time() - start_time
        print(
            f"Trainer initialized on {self.device} -> took {elapsed_time:.2f} seconds (T_max={self.total_training_steps})\n")

        if self.logger.should_resume():
            if os.path.exists(TEMP_CHECKPOINT_PATH):
                self.load_checkpoint(TEMP_CHECKPOINT_PATH)
                print("Loaded temp checkpoint\n")
            else:
                print("Warning: No temp checkpoint found, starting from scratch\n")

    def epoch_count(self):
        return EPOCH_COUNT

    def train_epoch(self, batch_start: int, batch_end: int, epoch: int, resume_from_batch: int = 0) -> Optional[bool]:
        self.model.train()
        total_loss = 0.0
        total_kl_loss = 0.0
        total_ce_loss = 0.0
        total_steps = 0
        start_batch = max(batch_start, resume_from_batch)

        current_lr = self.optimizer.param_groups[0]['lr']
        print(
            f"Training Epoch {epoch} with starting learning rate: {current_lr:.6f}\n")

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

        self.scheduler.step()
        self.current_step += 1
        current_lr = self.optimizer.param_groups[0]['lr']
        current_kl_ratio = self._get_current_kl_ratio()
        print(f"Batch {batch_idx + 1} -> LR: {current_lr:.8f}, KL ratio: {current_kl_ratio:.3f}")

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
        print(f"\tExample {example_idx + 1}/{total_examples}: {num_steps} steps, loss={avg_loss:.4f}, kl={avg_kl_loss:.4f}, ce={avg_ce_loss:.4f}, acc={accuracy:.4f} -> took {example_elapsed:.2f}s")

        return loss_sum, kl_loss_sum, ce_loss_sum, num_steps, correct

    def _log_batch_completion(self, batch_idx: int, batch_steps: int, batch_loss: float, batch_kl_loss: float, batch_ce_loss: float, batch_correct: int, batch_start_time: float, epoch: int):
        batch_elapsed = time() - batch_start_time
        avg_batch_loss = batch_loss / batch_steps if batch_steps > 0 else 0.0
        avg_batch_kl_loss = batch_kl_loss / batch_steps if batch_steps > 0 else 0.0
        avg_batch_ce_loss = batch_ce_loss / batch_steps if batch_steps > 0 else 0.0
        batch_accuracy = batch_correct / batch_steps if batch_steps > 0 else 0.0
        current_lr = self.optimizer.param_groups[0]['lr']

        print(f"Batch {batch_idx + 1} processed: {batch_steps} total steps, loss={avg_batch_loss:.4f}, kl={avg_batch_kl_loss:.4f}, ce={avg_batch_ce_loss:.4f}, accuracy={batch_accuracy:.4f} -> took {batch_elapsed:.2f}s\n")

        self.logger.log_training_batch(
            epoch=epoch,
            batch=batch_idx + 1,
            steps=batch_steps,
            loss=avg_batch_loss,
            kl_loss=avg_batch_kl_loss,
            ce_loss=avg_batch_ce_loss,
            accuracy=batch_accuracy,
            learning_rate=current_lr,
            time_seconds=batch_elapsed
        )
        self.logger.update_progress(epoch, batch_idx + 1)

    def _handle_exit_request(self, epoch: int, avg_batch_loss: float):
        print("Exit requested. Saving progress...")
        self.save_temp_checkpoint(epoch, avg_batch_loss)
        self.logger.save()
        return None

    def train_single_example(self, example: Dict) -> Tuple[float, float, float, int, int]:
        sentence_tokens = self._get_sentence_tokens(example)
        steps = example['steps']

        total_loss = 0.0
        total_kl_loss = 0.0
        total_ce_loss = 0.0
        valid_steps = 0
        correct_predictions = 0
        generated_token_ids = []

        self.optimizer.zero_grad(set_to_none=True)

        for step in steps:
            token_id, target_logits, target_index = self._prepare_step_data(
                step)

            input_tensor, target_tensor = self._create_tensors(
                sentence_tokens + generated_token_ids,
                target_logits
            )

            loss, kl_loss, ce_loss, is_correct = self._compute_step_loss(
                input_tensor, target_tensor, target_index)

            loss.backward()

            total_loss += loss.item()
            total_kl_loss += kl_loss.item()
            total_ce_loss += ce_loss.item()
            valid_steps += 1
            if is_correct:
                correct_predictions += 1
            generated_token_ids.append(token_id)

        self.optimizer.step()

        return total_loss, total_kl_loss, total_ce_loss, valid_steps, correct_predictions

    def eval_epoch(self, batch_start: int, batch_end: int, epoch: int) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_kl_loss = 0.0
        total_ce_loss = 0.0
        total_correct = 0
        total_steps = 0

        with torch.no_grad():
            for batch_idx in range(batch_start, batch_end):
                batch_start_time = time()
                batch_data = self.batch_handler.get_batch(batch_idx)

                batch_loss = 0.0
                batch_kl_loss = 0.0
                batch_ce_loss = 0.0
                batch_correct = 0
                batch_steps = 0

                for example_idx, example in enumerate(batch_data):
                    loss_sum, kl_loss_sum, ce_loss_sum, correct, num_steps = self._eval_single_example(
                        example)
                    batch_loss += loss_sum
                    batch_kl_loss += kl_loss_sum
                    batch_ce_loss += ce_loss_sum
                    batch_correct += correct
                    batch_steps += num_steps
                    total_loss += loss_sum
                    total_kl_loss += kl_loss_sum
                    total_ce_loss += ce_loss_sum
                    total_correct += correct
                    total_steps += num_steps

                batch_elapsed = time() - batch_start_time
                avg_batch_loss = batch_loss / batch_steps if batch_steps > 0 else 0.0
                avg_batch_kl_loss = batch_kl_loss / batch_steps if batch_steps > 0 else 0.0
                avg_batch_ce_loss = batch_ce_loss / batch_steps if batch_steps > 0 else 0.0
                batch_accuracy = batch_correct / batch_steps if batch_steps > 0 else 0.0
                print(
                    f"Eval Batch {batch_idx + 1}: {batch_steps} steps, loss={avg_batch_loss:.4f}, kl={avg_batch_kl_loss:.4f}, ce={avg_batch_ce_loss:.4f}, acc={batch_accuracy:.4f} -> took {batch_elapsed:.2f}s")

        avg_loss = total_loss / total_steps if total_steps > 0 else 0.0
        avg_kl_loss = total_kl_loss / total_steps if total_steps > 0 else 0.0
        avg_ce_loss = total_ce_loss / total_steps if total_steps > 0 else 0.0
        accuracy = total_correct / total_steps if total_steps > 0 else 0.0

        self.logger.log_eval_epoch(
            epoch, avg_loss, accuracy, total_steps, avg_kl_loss, avg_ce_loss)

        print(
            f"Eval Loss: {avg_loss:.4f} | KL Loss: {avg_kl_loss:.4f} | CE Loss: {avg_ce_loss:.4f} | Accuracy: {accuracy:.4f}")
        return avg_loss, accuracy

    def _eval_single_example(self, example: Dict) -> Tuple[float, float, float, int, int]:
        sentence_tokens = self._get_sentence_tokens(example)
        steps = example['steps']

        total_loss = 0.0
        total_kl_loss = 0.0
        total_ce_loss = 0.0
        correct_predictions = 0
        valid_steps = 0
        generated_token_ids = []

        for step in steps:
            token_id, target_logits, target_index = self._prepare_step_data(
                step)

            input_tensor, target_tensor = self._create_tensors(
                sentence_tokens + generated_token_ids,
                target_logits
            )

            loss, kl_loss, ce_loss, is_correct = self._compute_step_loss(
                input_tensor, target_tensor, target_index)
            total_loss += loss.item()
            total_kl_loss += kl_loss.item()
            total_ce_loss += ce_loss.item()

            if is_correct:
                correct_predictions += 1

            valid_steps += 1
            generated_token_ids.append(token_id)

        return total_loss, total_kl_loss, total_ce_loss, correct_predictions, valid_steps

    def _compute_step_loss(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor, target_index: int):
        model_logits = self.model(input_tensor)
        last_token_logits = model_logits[:, -1, :]

        kl_loss = self._compute_Kullback_Leibler_Divergence_Loss(
            last_token_logits, target_tensor)

        ce_loss = self._compute_Cross_Entropy_Loss(
            last_token_logits, target_index)

        kl_ratio = self._get_current_kl_ratio()
        combined_loss = kl_ratio * kl_loss + (1 - kl_ratio) * ce_loss

        predicted_idx = torch.argmax(last_token_logits[0]).item()
        is_correct = predicted_idx == target_index

        return combined_loss, kl_loss, ce_loss, is_correct

    def _get_current_kl_ratio(self) -> float:
        if self.total_training_steps <= 1:
            return KL_RATIO_START
        progress = self.current_step / self.total_training_steps
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return KL_RATIO_END + (KL_RATIO_START - KL_RATIO_END) * cosine_decay

    def _get_sentence_tokens(self, example: Dict) -> List[int]:
        text = example['sentence'] + '\n\n'
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _prepare_step_data(self, step: Dict) -> Tuple[int, List[float], int]:
        token = step['token']
        logit_vector = step['logits']
        target_index = step['predicted_token_index']

        token_ids = self.tokenizer.convert_tokens_to_ids([token])
        token_id = token_ids[0]

        return token_id, logit_vector, target_index

    def _create_tensors(self, input_ids: List[int], target_logits: List[float]) -> Tuple[torch.Tensor, torch.Tensor]:
        input_tensor = torch.tensor(
            [input_ids], dtype=torch.long, device=self.device)
        target_tensor = torch.tensor(
            [target_logits], dtype=torch.float32, device=self.device)
        return input_tensor, target_tensor

    def _compute_Kullback_Leibler_Divergence_Loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        temperature = DISTILLATION_TEMPERATURE
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

        kl_loss = F.kl_div(student_log_probs, teacher_probs,
                           reduction='batchmean') * (temperature ** 2)

        return kl_loss

    def _compute_Cross_Entropy_Loss(self, student_logits: torch.Tensor, target_index: int) -> torch.Tensor:
        temperature = DISTILLATION_TEMPERATURE
        target_tensor = torch.tensor(
            [target_index], device=student_logits.device)
        ce_loss = F.cross_entropy(
            student_logits / temperature, target_tensor) * (temperature ** 2)

        return ce_loss

    def save_checkpoint(self, epoch: int, train_loss: float):
        start_time = time()

        checkpoint = {
            'epoch': epoch,
            'current_step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
        }

        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')

        filepath = os.path.join('checkpoints', f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, filepath)

        elapsed_time = time() - start_time
        print(f"Checkpoint saved: {filepath} -> took {elapsed_time:.2f}s\n")

    def save_temp_checkpoint(self, epoch: int, train_loss: float):
        start_time = time()

        checkpoint = {
            'epoch': epoch,
            'current_step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
        }

        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')

        torch.save(checkpoint, TEMP_CHECKPOINT_PATH)

        elapsed_time = time() - start_time
        print(
            f"Temp checkpoint saved: {TEMP_CHECKPOINT_PATH} -> took {elapsed_time:.2f}s\n")

    def load_checkpoint(self, filepath: str) -> int:
        start_time = time()
        checkpoint = torch.load(
            filepath, map_location=self.device, weights_only=True)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer = AdamW(self.model.parameters(), lr=LEARNING_RATE)

        batch_count = self.batch_handler.get_training_batches_radius()[1]
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.total_training_steps, eta_min=1e-5)

        epoch = checkpoint['epoch']

        if 'current_step' in checkpoint:
            self.current_step = checkpoint['current_step']
        else:
            self.current_step = epoch * batch_count

        for _ in range(self.current_step):
            self.scheduler.step()

        elapsed_time = time() - start_time
        print(
            f"Checkpoint loaded: {filepath} (epoch {epoch}, step {self.current_step}) -> took {elapsed_time:.2f}s\n")

        return epoch
