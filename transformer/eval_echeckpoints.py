import torch
from pathlib import Path
from transformers import AutoTokenizer
from transformer.Transformer import Transformer
from Utilities import Utilities


EVAL_SENTENCE = "Oh fuck this happened at my school why"


class CheckpointEvaluator:
    def __init__(self, checkpoint_dir="checkpoints", max_length=75):
        self.checkpoint_dir = checkpoint_dir
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
        self.vocabulary_map = Utilities.build_vocabulary_map(self.tokenizer)
        self.output_token_ids = list(self.vocabulary_map.values())

        self.output_idx_to_token_id = {
            idx: token_id for idx, token_id in enumerate(self.output_token_ids)
        }

        self.model = Transformer(
            input_vocab_size=self.tokenizer.vocab_size,
            output_vocab_size=len(self.output_token_ids),
            output_token_ids=self.output_token_ids,
        ).to(self.device)

    def get_checkpoint_files(self):
        return sorted(
            Path(self.checkpoint_dir).glob("checkpoint_epoch_*.pt"),
            key=lambda x: int(x.stem.split('_')[-1])
        )

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint.get('epoch', 0)

    def generate(self, sentence: str):
        self.model.eval()

        input_ids = self.tokenizer.encode(
            sentence, return_tensors="pt").to(self.device)

        with torch.no_grad():
            for _ in range(self.max_length - input_ids.size(1)):
                logits = self.model(input_ids)
                next_output_idx = torch.argmax(logits[0, -1, :]).item()

                if next_output_idx not in self.output_idx_to_token_id:
                    break

                next_token_id = self.output_idx_to_token_id[next_output_idx]

                if next_token_id == self.tokenizer.eos_token_id:
                    break

                next_input_tensor = torch.tensor(
                    [[next_token_id]], device=self.device)
                input_ids = torch.cat([input_ids, next_input_tensor], dim=1)

                next_token = self.tokenizer.decode([next_token_id])
                if next_token == "}":
                    break

        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

    def evaluate_all_checkpoints(self):
        checkpoint_files = self.get_checkpoint_files()

        if not checkpoint_files:
            print(f"No checkpoints found in {self.checkpoint_dir}/")
            return

        print(f"Using device: {self.device}\n")
        print(f"Catalyst sentence: {EVAL_SENTENCE}\n")

        for checkpoint_path in checkpoint_files:
            epoch = self.load_checkpoint(checkpoint_path)
            generated_text = self.generate(EVAL_SENTENCE)

            print(f"=== Epoch {epoch} ===")
            print(generated_text)
            print("-" * 80)


evaluator = CheckpointEvaluator()
evaluator.evaluate_all_checkpoints()
