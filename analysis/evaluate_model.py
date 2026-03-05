import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))

from time import time
import torch
import torch.nn.functional as F

from training.model import Transformer
from training.batch_handler import BatchHandler
from shared import get_device, get_batches_dir, get_training_run_dir, INFERENCE_TEMPERATURE, PROMPT_DELIMITER

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


class Evaluator:
    def __init__(self, run_config: dict):
        self.device = get_device()
        self.model = Transformer(
            domain=run_config["domain"],
            teacher_model=run_config["teacher_model"],
            hidden_dim=run_config["hidden_dim"],
            num_layers=run_config["num_layers"],
            num_heads=run_config["num_heads"],
            dropout=run_config.get("dropout", 0.15),
            auxiliary_token_percentage=run_config.get("auxiliary_token_percentage", 1.0),
        ).to(self.device)
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


def list_runs() -> list:
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return []
    results = []
    for run_dir in sorted(runs_dir.iterdir()):
        info_path = run_dir / "info.json"
        if info_path.exists():
            with open(info_path, "r", encoding="utf-8") as file:
                info = json.load(file)
            results.append((run_dir.name, info))
    return results


def get_checkpoints(run_name: str) -> list:
    checkpoint_dir = Path(get_training_run_dir(run_name)) / "checkpoints"
    if not checkpoint_dir.exists():
        return []
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

    runs = list_runs()
    if not runs:
        print("No training runs found.")
        return

    print("Available runs:")
    for i, (name, info) in enumerate(runs):
        status = info.get("status", "unknown")
        domain = info.get("domain", "?")
        description = info.get("description", "")
        print(f"  [{i + 1}] {name} ({domain}, {status})")
        if description:
            print(f"      {description}")

    selection = input("\nSelect run number: ").strip()
    try:
        run_index = int(selection) - 1
        run_name, run_config = runs[run_index]
    except (ValueError, IndexError):
        print(f"Invalid selection.")
        return

    print(f"\nSelected: {run_name}")

    checkpoints = get_checkpoints(run_name)
    if not checkpoints:
        print("No checkpoints found for this run.")
        return

    print(f"Checkpoints: {', '.join(n for n, _ in checkpoints)} + all")
    checkpoint_selection = input("Select: ").strip().lower()

    if checkpoint_selection == "all":
        selected = checkpoints
    elif checkpoint_selection == "temp":
        selected = [(n, p) for n, p in checkpoints if n == "temp"]
    else:
        selected = [(n, p) for n, p in checkpoints if n == checkpoint_selection]

    if not selected:
        print(f"'{checkpoint_selection}' not found.")
        return

    domain = run_config["domain"]
    teacher_model = run_config["teacher_model"]
    batches_dir = get_batches_dir(domain, teacher_model)
    training_test_ratio = run_config.get("training_test_ratio", 0.98)

    batch_handler = BatchHandler(batches_dir, training_test_ratio)
    total_batches = batch_handler.get_batch_count()
    train_start, train_end = batch_handler.get_training_batches_radius()
    test_start, test_end = batch_handler.get_test_batches_radius()
    print(f"\nBatches: {total_batches} total (training: 0-{train_end - 1}, test: {test_start}-{test_end - 1})")

    batch_index = int(input("Batch index (0-based): ").strip())
    example_count = input("Examples (enter for all): ").strip()
    example_count = int(example_count) if example_count else None

    print("\nLoading model...")
    evaluator = Evaluator(run_config)
    batch = batch_handler.get_batch(batch_index)

    if example_count:
        batch = batch[:example_count]

    run_dir = Path(get_training_run_dir(run_name))
    log_dir = run_dir / "evals"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"eval_{int(time() * 1000)}.txt"

    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"Run: {run_name}\n")
        log.write(f"Domain: {domain}\n")
        log.write(f"Teacher: {teacher_model}\n\n")

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
