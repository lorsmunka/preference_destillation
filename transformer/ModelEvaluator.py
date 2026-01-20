import torch
from typing import Dict, List, Tuple
from time import time
from pathlib import Path

from BatchHandler import BatchHandler
from Transformer import Transformer
from shared import get_device


class ModelEvaluator:
    def __init__(self, checkpoint_dir: str):
        start_time = time()
        self.device = get_device()
        self.model = Transformer().to(self.device)
        self.tokenizer = self.model.tokenizer
        self.vocabulary = self.model.vocabulary
        self.output_token_ids = self.model.output_token_ids

        self.checkpoint_dir = checkpoint_dir
        self.log_dir = Path("evals")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"Evaluator ready on {self.device} -> init took {time() - start_time:.2f}s\n")

    def get_checkpoint_files(self):
        files = sorted(
            Path(self.checkpoint_dir).glob("checkpoint_epoch_*.pt"),
            key=lambda x: int(x.stem.split('_')[-1])
        )
        temp = Path(self.checkpoint_dir) / "temp_checkpoint.pt"
        if temp.exists():
            files.insert(0, temp)
        return files

    def _load_checkpoint(self, filepath: Path):
        print(f"\n[Loading {filepath.name}]")
        ckpt = torch.load(filepath, map_location=self.device,
                          weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def evaluate_all_checkpoints(self, batch_handler: BatchHandler, batch_index: int):
        checkpoints = self.get_checkpoint_files()
        if not checkpoints:
            print(f"No checkpoints found in {self.checkpoint_dir}")
            return

        timestamp = str(int(time() * 1000))
        log_path = self.log_dir / f"epoch_{timestamp}.txt"
        results = []

        with open(log_path, "w", encoding="utf-8") as logf:
            for idx, ckpt_path in enumerate(checkpoints, start=1):
                epoch_name = ckpt_path.stem.replace(
                    "checkpoint_epoch_", "").replace("temp_checkpoint", "temp")
                print(
                    f"\n=== [Epoch {epoch_name}] ({idx}/{len(checkpoints)}) ===")
                logf.write(f"\n=== Evaluating {epoch_name} ===\n")

                self._load_checkpoint(ckpt_path)
                total_correct, total_steps, batch_acc = self.evaluate_batch(
                    batch_index, batch_handler, logf)

                results.append((epoch_name, batch_acc))
                logf.flush()  # ensure per-epoch data is written

            logf.write("\n=== Summary ===\n")
            for name, acc in results:
                logf.write(f"{name} acc: {acc:.4f}\n")
            logf.flush()

        print(f"\nFull evaluation log saved -> {log_path}")

    def evaluate_batch(self, batch_index: int, batch_handler: BatchHandler, logf) -> Tuple[int, int, float]:
        start_time = time()
        batch_data = batch_handler.get_batch(batch_index)
        total_correct = total_steps = 0

        with torch.no_grad():
            for example_idx, example in enumerate(batch_data):
                correct, steps, generated_text = self._evaluate_example(
                    example)
                total_correct += correct
                total_steps += steps
                acc = correct / steps if steps else 0.0
                line = f"Example {example_idx + 1}: {correct}/{steps} correct (acc={acc:.4f})"
                print(" ", line)
                logf.write(line + "\n")
                logf.write(f"Generated: {generated_text}\n\n")

        batch_acc = total_correct / total_steps if total_steps else 0.0
        elapsed = time() - start_time
        summary = (
            f"Batch {batch_index} Results:\n"
            f"  Total Correct: {total_correct}/{total_steps}\n"
            f"  Accuracy: {batch_acc:.4f}\n"
            f"  Time: {elapsed:.2f}s\n"
        )
        print(" ", summary)
        logf.write(summary + "\n")
        logf.flush()
        return total_correct, total_steps, batch_acc

    def _evaluate_example(self, example: Dict) -> Tuple[int, int, str]:
        sentence_tokens = self._get_sentence_tokens(example)
        steps = example["steps"]
        correct_predictions = 0
        generated_ids_model = []

        for step in steps:
            gt_id, target_logits = self._prepare_step_data(step)
            inp = torch.tensor([sentence_tokens + generated_ids_model],
                               dtype=torch.long, device=self.device)
            logits = self.model(inp)
            pred_idx = torch.argmax(logits[:, -1, :][0]).item()
            pred_token_id = self.output_token_ids[pred_idx]
            if pred_token_id == gt_id:
                correct_predictions += 1
            generated_ids_model.append(pred_token_id)

        gen_text = self.tokenizer.decode(
            generated_ids_model, skip_special_tokens=True)
        return correct_predictions, len(steps), gen_text

    def _get_sentence_tokens(self, example: Dict) -> List[int]:
        return self.tokenizer.encode(example["sentence"] + "\n\n", add_special_tokens=False)

    def _prepare_step_data(self, step: Dict) -> Tuple[int, List[float]]:
        token = step["token"]
        logit_vector = step["logits"]

        token_id = self.tokenizer.convert_tokens_to_ids([token])[0]

        return token_id, logit_vector


def main():
    print("=== Model Accuracy Evaluator (All Checkpoints) ===\n")
    checkpoint_dir = "checkpoints"
    batch_index = int(input("Enter batch index (0-based): ").strip())
    bh = BatchHandler()
    evaluator = ModelEvaluator(checkpoint_dir)
    evaluator.evaluate_all_checkpoints(bh, batch_index)


if __name__ == "__main__":
    main()
