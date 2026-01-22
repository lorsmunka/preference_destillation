import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))

from time import time
import torch
import torch.nn.functional as F

from training.model import Transformer
from training.batch_handler import BatchHandler
from shared import get_device, INFERENCE_TEMPERATURE, PROMPT_DELIMITER

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


class Evaluator:
    def __init__(self):
        self.device = get_device()
        self.model = Transformer().to(self.device)
        self.tokenizer = self.model.tokenizer
        self.output_token_ids = self.model.output_token_ids

    def load(self, path: Path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict(self, input_ids: list) -> int:
        tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        logits = self.model(tensor)[:, -1, :][0]
        if not INFERENCE_TEMPERATURE:
            return torch.argmax(logits).item()
        probs = F.softmax(logits / INFERENCE_TEMPERATURE, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()

    def token_to_id(self, token: str) -> int:
        ids = self.tokenizer.encode(token, add_special_tokens=False)
        return ids[0] if ids else self.tokenizer.unk_token_id

    def evaluate(self, example: dict) -> dict:
        prompt = self.tokenizer.encode(example["sentence"] + PROMPT_DELIMITER, add_special_tokens=False)
        steps = example["steps"]

        student_ids, teacher_forced, ground_truth_ids = [], [], []

        for i, step in enumerate(steps):
            gt_id = self.token_to_id(step["token"])
            ground_truth_ids.append(gt_id)

            student_pred = self.output_token_ids[self.predict(prompt + student_ids)]
            student_ids.append(student_pred)

            teacher_context = [self.token_to_id(steps[j]["token"]) for j in range(i)]
            teacher_pred_id = self.output_token_ids[self.predict(prompt + teacher_context)]
            teacher_forced.append((self.tokenizer.decode([teacher_pred_id]), teacher_pred_id == gt_id))

        return {
            "student": self.tokenizer.decode(student_ids, skip_special_tokens=True),
            "student_acc": sum(s == g for s, g in zip(student_ids, ground_truth_ids)) / len(steps),
            "teacher_forced": teacher_forced,
            "teacher_forced_acc": sum(ok for _, ok in teacher_forced) / len(steps),
            "ground_truth": self.tokenizer.decode(ground_truth_ids, skip_special_tokens=True),
        }


def get_checkpoints() -> list:
    checkpoint_dir = Path("checkpoints")
    result = []
    temp = checkpoint_dir / "temp_checkpoint.pt"
    if temp.exists():
        result.append(("temp", temp))
    for p in sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"), key=lambda x: int(x.stem.split('_')[-1])):
        result.append((p.stem.split('_')[-1], p))
    return result


def format_tokens(tokens: list, color: bool) -> str:
    if color:
        return "".join(f"{GREEN if ok else RED}{t}{RESET}" for t, ok in tokens)
    return "".join(t for t, ok in tokens)


def main():
    print("=== Model Evaluator ===\n")

    checkpoints = get_checkpoints()
    if not checkpoints:
        print("No checkpoints found.")
        return

    print("Checkpoints:", ", ".join(n for n, _ in checkpoints), "+ all")
    selection = input("Select: ").strip().lower()

    if selection == "all":
        selected = checkpoints
    elif selection == "temp":
        selected = [(n, p) for n, p in checkpoints if n == "temp"]
    else:
        selected = [(n, p) for n, p in checkpoints if n == selection]

    if not selected:
        print(f"'{selection}' not found.")
        return

    batch_index = int(input("Batch index: ").strip())
    example_count = input("Examples (enter for all): ").strip()
    example_count = int(example_count) if example_count else None

    print("\nLoading model...")
    evaluator = Evaluator()
    batch_handler = BatchHandler()
    batch = batch_handler.get_batch(batch_index)

    if example_count:
        batch = batch[:example_count]

    log_dir = Path("evals")
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"eval_{int(time() * 1000)}.txt"

    with open(log_path, "w", encoding="utf-8") as log:
        for name, path in selected:
            header = f"\n{'='*40}\nCheckpoint: {name}\n{'='*40}"
            print(header)
            log.write(header + "\n")

            evaluator.load(path)
            student_total, tf_total = 0.0, 0.0

            with torch.no_grad():
                for i, example in enumerate(batch):
                    result = evaluator.evaluate(example)
                    student_total += result["student_acc"]
                    tf_total += result["teacher_forced_acc"]

                    output = [
                        f"\n[Example {i + 1}]",
                        f"Student ({result['student_acc']:.2%}): {result['student']}",
                        f"Teacher-Forced ({result['teacher_forced_acc']:.2%}): {format_tokens(result['teacher_forced'], True)}",
                        f"Ground Truth: {result['ground_truth']}"
                    ]
                    for line in output:
                        print(line)

                    log_output = [
                        f"\n[Example {i + 1}]",
                        f"Student ({result['student_acc']:.2%}): {result['student']}",
                        f"Teacher-Forced ({result['teacher_forced_acc']:.2%}): {format_tokens(result['teacher_forced'], False)}",
                        f"Ground Truth: {result['ground_truth']}"
                    ]
                    for line in log_output:
                        log.write(line + "\n")

            n = len(batch)
            summary = f"\nAverage: Student={student_total/n:.2%}, Teacher-Forced={tf_total/n:.2%}"
            print(summary)
            log.write(summary + "\n")

    print(f"\nSaved: {log_path}")


if __name__ == "__main__":
    main()
