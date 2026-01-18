import torch
from pathlib import Path
from transformers import AutoTokenizer
from Transformer import Transformer
from shared import Utilities


EVAL_SENTENCES = [
    "Real thirstposting hours who up",
    "If someone put that spinning on a drill would that fish keep trying?",
    "It's not. Different branches of government actually.",
    "Yeah wtf is up with that I keep getting it and I barely play her anymore",
    "The gods will protect",
    "Its the DBZ live action movie all over again.",
    "Not until it's drenched in my cum",
    "He doesn't get it back. He'll be fine without it.",
    "What if they said that when they were raising him.",
    "If I was ever there I would drink a hot sauce bottle like a beer.",
    "Lol stop. We just want to give women, minorities. and LGBTQ free folk more representation",
    "Imagine if thanos broke mjolnir and no one could return it",
    "Imagine liking in reddit",
    "Why does Sanic look like hes been photoshopped into the film by a high school graphic design student?",
    "You look amazing! It would take every ounce of restraint to keep myself from bending you over that bed instantly!",
    "Haley is literally the definition of work ethic over skill",
    "See theres a word missing from this sentence that makes it so much better and that is the word again after the word ring.",
    "poltards deserve to be mocked in any way possible.",
    "Nobody Pencil niggas yall mind if I STICK?",
    "No it doesn't lol. Europe can be incredibly racist.",
    "Stocks go up and they go down",
    "I googled this to double check it wasn't a very, very convincing fake fact. It's real!",
    "What do u use to get the cropped faces to stay on a moving target",
    "Spoiler gt!It was sad to see the character that started it all go out like that!lt",
    "Yea that sucks because how do people not know the game",
    "Science communication articles which don't link to the source publication...",
    "Swoon! You did an amazing job!!!",
    "This dude put a lv1 barrel stabilizer on his car and a lv4 on his longbow",
    "This is why we cant have nice things.",
    "Yes this comment right here, officer.",
    "Aahhh a master of the meme language, it's a pleasure to have you by.",
    "I despise those cock wombles at Epic.",
    "People really thought Giannis would be locked up the whole series after 1 game lol",
    "The hOund Ned gendrY Stark.",
    "That's SpongeBob and Patrick's baby!",
    "Who else tried saying this in his voice and failed due to extreme laughter?",
    "Dude Sharks PP lmao"
]


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
        ).to(self.device)

    def get_checkpoint_files(self):
        checkpoint_files = sorted(
            Path(self.checkpoint_dir).glob("checkpoint_epoch_*.pt"),
            key=lambda x: int(x.stem.split('_')[-1])
        )

        checkpoint_files = []

        temp_checkpoint = Path(self.checkpoint_dir) / "temp_checkpoint.pt"
        if temp_checkpoint.exists():
            checkpoint_files.insert(0, temp_checkpoint)

        return checkpoint_files

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint.get('epoch', 0)

    def generate(self, sentence: str):
        self.model.eval()

        sentence_with_delimiter = sentence + '\n\n'
        input_ids = self.tokenizer.encode(
            sentence_with_delimiter, add_special_tokens=False, return_tensors="pt").to(self.device)

        generated_text = ""

        with torch.no_grad():
            for step in range(50):
                logits = self.model(input_ids)
                next_output_idx = torch.argmax(logits[0, -1, :]).item()

                if next_output_idx not in self.output_idx_to_token_id:
                    break

                next_token_id = self.output_idx_to_token_id[next_output_idx]

                next_token_str = self.tokenizer.decode([next_token_id])
                generated_text += next_token_str

                encoded_ids = self.tokenizer.encode(
                    next_token_str, add_special_tokens=False)

                if len(encoded_ids) == 0:
                    break

                actual_token_id = encoded_ids[0]

                next_input_tensor = torch.tensor(
                    [[actual_token_id]], device=self.device)
                input_ids = torch.cat([input_ids, next_input_tensor], dim=1)

                if next_token_str == "}":
                    break

        return generated_text

    def evaluate_all_checkpoints(self):
        checkpoint_files = self.get_checkpoint_files()

        if not checkpoint_files:
            print(f"No checkpoints found in {self.checkpoint_dir}/")
            return

        for EVAL_SENTENCE in EVAL_SENTENCES:
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
