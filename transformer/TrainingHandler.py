import torch
import torch.nn as nn
from torch.optim import AdamW
from typing import Tuple, List, Dict

EPOCH_COUNT = 10
LEARNING_RATE = 3e-4


class TrainingHandler:
    def __init__(self, model: nn.Module, token_to_id: Dict[str, int], device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.device = device
        self.token_to_id = token_to_id
        self.optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.CrossEntropyLoss()

        print(f"TrainingHandler initialized on {device}")
        print(f"Model parameters: {model.get_num_parameters():,}")

    def epoch_count(self):
        return EPOCH_COUNT

    def prepare_batch(self, batch_data: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        input_sequences = []
        target_sequences = []

        for item in batch_data:
            input_tokens = []
            target_tokens = []

            steps = item['steps']

            for step in steps:
                token = step['token']
                token_id = self.token_to_id.get(token, 0)

                input_tokens.append(token_id)
                target_tokens.append(token_id)

            input_sequences.append(input_tokens)
            target_sequences.append(target_tokens)

        input_ids = self.pad_sequences(input_sequences)
        target_ids = self.pad_sequences(target_sequences)

        return input_ids, target_ids

    def pad_sequences(self, sequences: List[List[int]]) -> torch.Tensor:
        max_length = max(len(seq) for seq in sequences)

        padded = []
        for seq in sequences:
            padded_seq = seq + [0] * (max_length - len(seq))
            padded.append(padded_seq)

        return torch.tensor(padded, dtype=torch.long)

    def train_epoch(self, batch_handler, batch_start: int, batch_end: int) -> float:
        self.model.train()
        total_loss = 0.0
        batch_count = 0

        for batch_idx in range(batch_start, batch_end):
            batch_data = batch_handler.get_batch(batch_idx)

            input_ids, target_ids = self.prepare_batch(batch_data)
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(input_ids)

            loss = self.calculate_loss(logits, target_ids)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        average_loss = total_loss / batch_count
        return average_loss

    def eval_epoch(self, batch_handler, batch_start: int, batch_end: int) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        with torch.no_grad():
            for batch_idx in range(batch_start, batch_end):
                batch_data = batch_handler.get_batch(batch_idx)

                input_ids, target_ids = self.prepare_batch(batch_data)
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)

                logits = self.model(input_ids)

                loss = self.calculate_loss(logits, target_ids)
                total_loss += loss.item()

                predictions = torch.argmax(logits, dim=-1)
                correct = (predictions == target_ids).sum().item()
                total_correct += correct
                total_tokens += target_ids.numel()

        average_loss = total_loss / (batch_end - batch_start)
        accuracy = total_correct / total_tokens
        return average_loss, accuracy

    def calculate_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, vocab_size = logits.shape

        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)

        loss = self.loss_fn(logits_flat, targets_flat)
        return loss

    def save_checkpoint(self, filepath: str, epoch: int, train_loss: float):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str) -> int:
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Checkpoint loaded from {filepath} (epoch {epoch})")
        return epoch
