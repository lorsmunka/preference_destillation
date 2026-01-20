import torch
from pathlib import Path
from Transformer import Transformer
from BatchHandler import BatchHandler
from shared import get_device

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


class CheckpointEvaluator:
    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self.device = get_device()

        self.model = Transformer().to(self.device)
        self.tokenizer = self.model.tokenizer
        self.vocabulary = self.model.vocabulary
        self.output_token_ids = self.model.output_token_ids

        self.output_index_to_token_id = {
            index: token_id for index, token_id in enumerate(self.output_token_ids)
        }

        self.batch_handler = BatchHandler()

        print(f"Evaluator ready on {self.device}\n")

    def get_checkpoint_files(self):
        checkpoint_files = sorted(
            Path(self.checkpoint_dir).glob("checkpoint_epoch_*.pt"),
            key=lambda x: int(x.stem.split('_')[-1])
        )
        return checkpoint_files

    def get_temp_checkpoint(self):
        temp_checkpoint = Path(self.checkpoint_dir) / "temp_checkpoint.pt"
        return temp_checkpoint if temp_checkpoint.exists() else None

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint.get('epoch', 0)

    def evaluate_example_with_teacher_forcing(self, example):
        self.model.eval()

        sentence = example['sentence']
        ground_truth_steps = example['steps']

        sentence_with_delimiter = sentence + '\n\n'
        input_ids = self.tokenizer.encode(
            sentence_with_delimiter, add_special_tokens=False)

        colored_output = ""
        correct_count = 0
        total_count = len(ground_truth_steps)

        with torch.no_grad():
            for step in ground_truth_steps:
                ground_truth_token = step['token']
                ground_truth_token_id = self.tokenizer.convert_tokens_to_ids([ground_truth_token])[0]

                input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
                logits = self.model(input_tensor)
                predicted_output_index = torch.argmax(logits[0, -1, :]).item()

                predicted_token_id = self.output_index_to_token_id.get(predicted_output_index)
                predicted_token = self.tokenizer.decode([predicted_token_id]) if predicted_token_id else "<?>"

                is_correct = predicted_token_id == ground_truth_token_id

                if is_correct:
                    colored_output += f"{GREEN}{predicted_token}{RESET}"
                    correct_count += 1
                else:
                    colored_output += f"{RED}{predicted_token}{RESET}"

                input_ids.append(ground_truth_token_id)

        accuracy = correct_count / total_count if total_count > 0 else 0.0
        return colored_output, correct_count, total_count, accuracy

    def evaluate_checkpoint(self, checkpoint_path, examples):
        epoch = self.load_checkpoint(checkpoint_path)
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}")
        print(f"{'='*60}")

        total_correct = 0
        total_tokens = 0

        for example_index, example in enumerate(examples):
            sentence = example['sentence']
            ground_truth_response = example.get('model_response', '')

            colored_output, correct, total, accuracy = self.evaluate_example_with_teacher_forcing(example)

            total_correct += correct
            total_tokens += total

            print(f"\n[Example {example_index + 1}] {sentence[:60]}...")
            print(f"Ground truth: {ground_truth_response}")
            print(f"Prediction:   {colored_output}")
            print(f"Accuracy: {correct}/{total} ({accuracy:.1%})")

        overall_accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
        print(f"\n{'─'*60}")
        print(f"Overall: {total_correct}/{total_tokens} ({overall_accuracy:.1%})")

    def run(self):
        checkpoint_files = self.get_checkpoint_files()
        temp_checkpoint = self.get_temp_checkpoint()

        if not checkpoint_files and not temp_checkpoint:
            print(f"No checkpoints found in {self.checkpoint_dir}/")
            return

        print(f"Available checkpoints:")
        for index, path in enumerate(checkpoint_files):
            print(f"  [{index}] {path.name}")
        if temp_checkpoint:
            print(f"  [temp] {temp_checkpoint.name}")

        checkpoint_input = input("\nSelect checkpoint (number / 'temp' / 'all'): ").strip().lower()

        example_count = int(input("How many examples to evaluate: ").strip())
        batch_index = int(input("Batch index (0-based): ").strip())

        batch_data = self.batch_handler.get_batch(batch_index)
        examples = batch_data[:example_count]

        print(f"\nEvaluating {len(examples)} examples from batch {batch_index + 1}")

        if checkpoint_input == 'all':
            for checkpoint_path in checkpoint_files:
                self.evaluate_checkpoint(checkpoint_path, examples)
            if temp_checkpoint:
                self.evaluate_checkpoint(temp_checkpoint, examples)
        elif checkpoint_input == 'temp':
            if temp_checkpoint:
                self.evaluate_checkpoint(temp_checkpoint, examples)
            else:
                print("No temp checkpoint found")
        else:
            checkpoint_index = int(checkpoint_input)
            self.evaluate_checkpoint(checkpoint_files[checkpoint_index], examples)


if __name__ == "__main__":
    evaluator = CheckpointEvaluator()
    evaluator.run()
