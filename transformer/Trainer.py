import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Dict, List, Tuple
from time import time

from TelemetryHandler import TelemetryHandler
from ExitListener import ExitListener
from Transformer import Transformer

EPOCH_COUNT = 10
LEARNING_RATE = 3e-4
TEMPERATURE = 1.0
TEMP_CHECKPOINT_PATH = 'checkpoints/temp_checkpoint.pt'


class Trainer:
    def __init__(self, model: Transformer, token_to_id: Dict[str, int], tokenizer, telemetry_handler: TelemetryHandler, exit_listener: ExitListener):
        start_time = time()
        print("Initializing Trainer...")

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"

        self.model = model.to(self.device)
        self.token_to_id = token_to_id
        self.tokenizer = tokenizer
        self.optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

        self.output_token_ids = model.output_token_ids
        self.token_id_to_output_idx = {
            token_id: idx for idx, token_id in enumerate(self.output_token_ids)
        }

        self.telemetry_handler = telemetry_handler
        self.exit_listener = exit_listener

        elapsed_time = time() - start_time
        print(
            f"Trainer initialized on {self.device} -> took {elapsed_time:.2f} seconds \n")

        if self.telemetry_handler.should_resume():
            if os.path.exists(TEMP_CHECKPOINT_PATH):
                self.load_checkpoint(TEMP_CHECKPOINT_PATH)
                print("Loaded temp checkpoint\n")
            else:
                print("Warning: No temp checkpoint found, starting from scratch\n")

    def epoch_count(self):
        return EPOCH_COUNT

    def train_epoch(self, batch_handler, batch_start: int, batch_end: int, epoch: int, resume_from_batch: int = 0) -> float:
        self.model.train()
        total_loss = 0.0
        total_steps = 0
        start_batch = max(batch_start, resume_from_batch)

        for batch_idx in range(start_batch, batch_end):
            batch_result = self._process_batch(batch_handler, batch_idx, epoch)

            if batch_result is None:
                break

            batch_loss, batch_steps = batch_result
            total_loss += batch_loss
            total_steps += batch_steps

        avg_epoch_loss = total_loss / max(total_steps, 1)
        self.telemetry_handler.log_train_epoch(
            epoch, avg_epoch_loss, total_steps)

        self.telemetry_handler.current_batch = 0

    def _process_batch(self, batch_handler, batch_idx: int, epoch: int):
        batch_start_time = time()
        batch_data = batch_handler.get_batch(batch_idx)

        batch_loss = 0.0
        batch_steps = 0

        for example_idx, example in enumerate(batch_data):
            example_loss, example_steps = self._process_example(
                example, example_idx, len(batch_data), epoch, batch_idx)
            batch_loss += example_loss
            batch_steps += example_steps

        self._log_batch_completion(
            batch_idx, batch_steps, batch_loss, batch_start_time, epoch)

        if self.exit_listener.check_exit():
            return self._handle_exit_request(epoch, batch_loss / max(batch_steps, 1))

        return batch_loss, batch_steps

    def _process_example(self, example, example_idx: int, total_examples: int, epoch: int, batch_idx: int):
        example_start_time = time()
        loss, num_steps = self.train_single_example(example)

        weighted_loss = 0.0
        if num_steps > 0:
            weighted_loss = loss * num_steps

        example_elapsed = time() - example_start_time
        print(
            f"\tExample {example_idx + 1}/{total_examples}: {num_steps} steps, loss={loss:.4f} -> took {example_elapsed:.2f}s")

        self.telemetry_handler.log_training_example(
            epoch, batch_idx + 1, example_idx + 1, num_steps, loss, example_elapsed)

        return weighted_loss, num_steps

    def _log_batch_completion(self, batch_idx: int, batch_steps: int, batch_loss: float, batch_start_time: float, epoch: int):
        batch_elapsed = time() - batch_start_time
        avg_batch_loss = batch_loss / max(batch_steps, 1)
        print(f"Batch {batch_idx + 1} processed: {batch_steps} total steps, avg_loss={avg_batch_loss:.4f} -> took {batch_elapsed:.2f}s\n")
        self.telemetry_handler.update_progress(epoch, batch_idx + 1)

    def _handle_exit_request(self, epoch: int, avg_batch_loss: float):
        print("Exit requested. Saving progress...")
        self.save_temp_checkpoint(epoch, avg_batch_loss)
        self.telemetry_handler.save()
        return None

    def train_single_example(self, example: Dict) -> Tuple[float, int]:
        sentence_tokens = self._get_sentence_tokens(example)
        steps = sorted(example['steps'],
                       key=lambda s: example['steps'].index(s))

        total_loss = 0.0
        valid_steps = 0
        generated_token_ids = []

        for step in steps:
            token_id, target_logits = self._prepare_step_data(step)
            if token_id is None:
                continue

            input_tensor, target_tensor = self._create_tensors(
                sentence_tokens + generated_token_ids,
                target_logits
            )

            self.optimizer.zero_grad()
            model_logits = self.model(input_tensor)
            last_token_logits = model_logits[:, -1, :]
            loss = self._compute_Kullback_Leibler_Divergence_Loss(
                last_token_logits, target_tensor)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            valid_steps += 1
            generated_token_ids.append(token_id)

        return total_loss, valid_steps

    def eval_epoch(self, batch_handler, batch_start: int, batch_end: int, epoch: int) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_steps = 0

        with torch.no_grad():
            for batch_idx in range(batch_start, batch_end):
                batch_start_time = time()
                batch_data = batch_handler.get_batch(batch_idx)

                batch_loss = 0.0
                batch_correct = 0
                batch_steps = 0

                for example_idx, example in enumerate(batch_data):
                    loss, correct, num_steps = self._eval_single_example(
                        example)

                    if num_steps > 0:
                        batch_loss += loss * num_steps
                        batch_correct += correct
                        batch_steps += num_steps
                        total_loss += loss * num_steps
                        total_correct += correct
                        total_steps += num_steps

                batch_elapsed = time() - batch_start_time
                avg_batch_loss = batch_loss / max(batch_steps, 1)
                batch_accuracy = batch_correct / max(batch_steps, 1)
                print(
                    f"Eval Batch {batch_idx + 1}: {batch_steps} steps, loss={avg_batch_loss:.4f}, acc={batch_accuracy:.4f} -> took {batch_elapsed:.2f}s")

        avg_loss = total_loss / max(total_steps, 1)
        accuracy = total_correct / max(total_steps, 1)

        self.telemetry_handler.log_eval_epoch(
            epoch, avg_loss, accuracy, total_steps)

        print(f"Eval Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")
        return avg_loss, accuracy

    def _eval_single_example(self, example: Dict) -> Tuple[float, int, int]:
        sentence_tokens = self._get_sentence_tokens(example)
        steps = sorted(example['steps'],
                       key=lambda s: example['steps'].index(s))

        total_loss = 0.0
        correct_predictions = 0
        valid_steps = 0
        generated_token_ids = []

        for step in steps:
            token_id, target_logits = self._prepare_step_data(step)

            input_tensor, target_tensor = self._create_tensors(
                sentence_tokens + generated_token_ids,
                target_logits
            )

            model_logits = self.model(input_tensor)
            last_token_logits = model_logits[:, -1, :]

            loss = self._compute_Kullback_Leibler_Divergence_Loss(
                last_token_logits, target_tensor)
            total_loss += loss.item()

            predicted_idx = torch.argmax(last_token_logits[0]).item()
            target_idx = torch.argmax(target_tensor[0]).item()

            if predicted_idx == target_idx:
                correct_predictions += 1

            valid_steps += 1
            generated_token_ids.append(token_id)

        return total_loss, correct_predictions, valid_steps

    def _get_sentence_tokens(self, example: Dict) -> List[int]:
        return self.tokenizer.encode(example['sentence'], add_special_tokens=False)

    def _prepare_step_data(self, step: Dict) -> Tuple[int, List[float]]:
        token = step['token']
        probabilities = step['probabilities']

        token_id = self.token_to_id.get(token)

        target_logits = self._build_target_logits(probabilities)

        return token_id, target_logits

    def _create_tensors(self, input_ids: List[int], target_logits: List[float]) -> Tuple[torch.Tensor, torch.Tensor]:
        input_tensor = torch.tensor(
            [input_ids], dtype=torch.long, device=self.device)
        target_tensor = torch.tensor(
            [target_logits], dtype=torch.float32, device=self.device)
        return input_tensor, target_tensor

    def _build_target_logits(self, probabilities: Dict[str, float]) -> List[float]:
        target = [-100.0] * len(self.output_token_ids)

        for token_str, logit_value in probabilities.items():
            token_id = self.token_to_id.get(token_str)
            output_idx = self.token_id_to_output_idx.get(token_id)
            target[output_idx] = logit_value

        return target

    def _compute_Kullback_Leibler_Divergence_Loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        teacher_logits = teacher_logits.clone()
        valid_mask = teacher_logits != -100.0
        teacher_logits[~valid_mask] = -1e9

        student_log_probs = F.log_softmax(student_logits / TEMPERATURE, dim=-1)
        teacher_probs = F.softmax(teacher_logits / TEMPERATURE, dim=-1)

        kl_loss = F.kl_div(student_log_probs, teacher_probs,
                           reduction='batchmean') * (TEMPERATURE ** 2)

        return kl_loss

    def save_checkpoint(self, epoch: int, train_loss: float):
        start_time = time()

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
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
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
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
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']

        elapsed_time = time() - start_time
        print(
            f"Checkpoint loaded: {filepath} (epoch {epoch}) -> took {elapsed_time:.2f}s\n")

        return epoch
