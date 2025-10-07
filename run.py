import os
import time
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import glob
from datetime import datetime
from typing import Dict, List, Tuple


class PromptBuilder:
    @staticmethod
    def create_evaluation_prompt(sentence: str) -> str:
        prompt = f"""Analyze this sentence and return your evaluation as JSON:

    Sentence: "{sentence}"

    Provide exactly one value for each field based on the sentence content:
    - tone: aggressive, rude, neutral, polite, friendly
    - sentiment: negative, neutral, positive
    - safety: harmful, safe
    - toxicity: toxic, respectful

    JSON:
    """
        return prompt


class DurationFormatter:
    @staticmethod
    def format(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{int(minutes)}m {secs:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{int(hours)}h {int(minutes)}m"


class TelemetryLogger:
    def __init__(self):
        self.filename = self._create_file()

    def _create_file(self) -> str:
        os.makedirs("./telemetry", exist_ok=True)
        timestamp = int(time.time() * 1000)
        filename = f"./telemetry/run_{timestamp}.txt"

        with open(filename, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("TRAINING DATA GENERATION TELEMETRY\n")
            f.write("=" * 60 + "\n")
            f.write(
                f"Run Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Epoch Timestamp: {timestamp}\n")
            f.write("-" * 60 + "\n\n")

        return filename

    def update(self, generated_count: int, total_examples: int,
               last_generation_time: float, start_time: float,
               batch_start_time: float, batch_examples_count: int):
        elapsed_time = time.time() - start_time
        avg_time_per_example = elapsed_time / \
            generated_count if generated_count > 0 else 0

        batch_elapsed_time = time.time() - batch_start_time
        batch_avg_time = batch_elapsed_time / \
            batch_examples_count if batch_examples_count > 0 else 0

        remaining_examples = total_examples - generated_count
        estimated_remaining_time = batch_avg_time * remaining_examples
        total_estimated_time = elapsed_time + estimated_remaining_time

        progress_percent = (generated_count / total_examples) * 100

        formatter = DurationFormatter()

        with open(self.filename, 'a') as f:
            f.write(
                f"Example #{generated_count:,} | Last Generation Time: {formatter.format(last_generation_time)}\n")
            f.write(
                f"Progress: {generated_count:,}/{total_examples:,} ({progress_percent:.1f}%)\n")
            f.write(
                f"Current Batch Avg Time/Example: {formatter.format(batch_avg_time)}\n")
            f.write(
                f"Overall Avg Time/Example: {formatter.format(avg_time_per_example)}\n")
            f.write(f"Elapsed Time: {formatter.format(elapsed_time)}\n")
            f.write(
                f"Estimated Total Runtime: {formatter.format(total_estimated_time)}\n")
            f.write(
                f"Estimated Time Remaining: {formatter.format(estimated_remaining_time)}\n")
            f.write("-" * 60 + "\n")

    def finalize(self, generated_count: int, total_examples: int,
                 start_time: float, interrupted: bool = False):
        elapsed_time = time.time() - start_time
        status = "INTERRUPTED" if interrupted else "COMPLETED"
        formatter = DurationFormatter()

        with open(self.filename, 'a') as f:
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"RUN {status}\n")
            f.write("=" * 60 + "\n")
            f.write(
                f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(
                f"Total Examples Generated: {generated_count:,}/{total_examples:,}\n")
            f.write(f"Total Runtime: {formatter.format(elapsed_time)}\n")
            if generated_count > 0:
                avg_time = elapsed_time / generated_count
                f.write(
                    f"Average Time per Example: {formatter.format(avg_time)}\n")
            f.write("=" * 60 + "\n")


class VocabularyMapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.token_to_id_map = self._build_vocabulary_map()

    def _build_vocabulary_map(self) -> Dict[str, int]:
        example_result = """
        ```json
        {
            "tone": "aggressive",
            "tone": "rude",
            "tone": "neutral",
            "tone": "polite",
            "tone": "friendly",
            "sentiment": "negative",
            "sentiment": "neutral",
            "sentiment": "positive",
            "safety": "harmful",
            "safety": "safe",
            "toxicity": "toxic",
            "toxicity": "respectful"
        }
        """

        auxiliary_tokens = ["the", "a", "is", "of", "and", "to", "in", "that", "it", "you",
                            "very", "quite", "somewhat", "extremely", "slightly", "moderately"]

        all_text = f"{example_result} {' '.join(auxiliary_tokens)} {PromptBuilder.create_evaluation_prompt('')}"

        tokens = self.tokenizer.tokenize(all_text)
        unique_tokens = sorted(set(tokens))

        token_to_id = {}
        for token_text in unique_tokens:
            token_ids = self.tokenizer.convert_tokens_to_ids([token_text])
            if token_ids[0] != self.tokenizer.unk_token_id:
                token_to_id[token_text] = token_ids[0]

        print(f"Pre-computed {len(token_to_id)} token IDs for vocabulary")
        return token_to_id

    def extract_logits_with_other_token(self, logits) -> Tuple[List[str], List[float]]:
        token_names = []
        token_logits = []
        used_indices = set()

        for token_text, token_id in self.token_to_id_map.items():
            token_logits.append(logits[token_id].item())
            token_names.append(token_text)
            used_indices.add(token_id)

        other_indices = [i for i in range(
            len(logits)) if i not in used_indices]
        if other_indices:
            other_logits = logits[other_indices]
            max_logit = other_logits.max()
            other_logit_sum = max_logit + \
                torch.log(torch.sum(torch.exp(other_logits - max_logit)))
            token_logits.append(other_logit_sum.item())
            token_names.append("__OTHER__")

        return token_names, token_logits

    @staticmethod
    def sort_by_logits(token_names: List[str], token_logits: List[float]) -> Dict[str, float]:
        token_logit_pairs = list(zip(token_names, token_logits))
        token_logit_pairs.sort(key=lambda x: x[1], reverse=True)
        return {token: logit for token, logit in token_logit_pairs}


class ModelManager:
    def __init__(self, model_name: str = "google/gemma-3-4b-it"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.vocab_mapper = None

    def load(self):
        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            dtype=torch.bfloat16
        )
        self.vocab_mapper = VocabularyMapper(self.tokenizer)

    def prepare_inputs_for_gpu(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        return inputs


class ResponseValidator:
    REQUIRED_KEYS = {"tone", "sentiment", "safety", "toxicity"}
    VALID_VALUES = {
        "tone": {"aggressive", "rude", "neutral", "polite", "friendly"},
        "sentiment": {"negative", "neutral", "positive"},
        "safety": {"harmful", "safe"},
        "toxicity": {"toxic", "respectful"}
    }

    @staticmethod
    def extract_json_from_text(text: str) -> str:
        text = text.strip()

        json_match = re.search(
            r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()

        brace_match = re.search(r'\{.*\}', text, re.DOTALL)
        if brace_match:
            return brace_match.group(0)

        return text

    @classmethod
    def is_valid(cls, text: str) -> bool:
        print(f"Validating \n{text}\n")

        cleaned_text = cls.extract_json_from_text(text)

        try:
            parsed = json.loads(cleaned_text)
        except json.JSONDecodeError:
            return False

        if not isinstance(parsed, dict):
            return False

        if parsed.keys() != cls.REQUIRED_KEYS:
            return False

        for key, valid_set in cls.VALID_VALUES.items():
            if parsed.get(key) not in valid_set:
                return False

        return True


class TextGenerator:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    def should_stop_generation(self, token: str, accumulated_text: str) -> bool:
        if token.strip() == "}":
            return True
        if len(accumulated_text) > 500:
            return True
        return False

    def generate_single_step(self, current_sequence):
        current_inputs = {
            'input_ids': current_sequence,
            'attention_mask': torch.ones_like(current_sequence)
        }

        with torch.no_grad():
            logits = self.model_manager.model(
                **current_inputs).logits[0, -1, :]

        token_names, token_logits = self.model_manager.vocab_mapper.extract_logits_with_other_token(
            logits)
        step_probs = VocabularyMapper.sort_by_logits(token_names, token_logits)

        next_token_id = torch.argmax(logits).item()
        next_token = self.model_manager.tokenizer.decode([next_token_id])

        new_sequence = torch.cat([current_sequence, torch.tensor(
            [[next_token_id]], device=current_sequence.device)], dim=1)

        return next_token, step_probs, new_sequence

    def generate_training_example(self, sentence: str) -> Dict:
        prompt = PromptBuilder.create_evaluation_prompt(sentence)
        inputs = self.model_manager.prepare_inputs_for_gpu(prompt)

        steps = []
        generated_text = ""
        current_sequence = inputs['input_ids']

        while len(steps) < 100:
            next_token, step_probs, new_sequence = self.generate_single_step(
                current_sequence)

            steps.append({
                "token": next_token,
                "probabilities": step_probs
            })

            generated_text += next_token
            current_sequence = new_sequence

            if self.should_stop_generation(next_token, generated_text):
                break

        return {
            "sentence": sentence,
            "generated_response": generated_text.strip(),
            "steps": steps
        }


class SentenceLoader:
    @staticmethod
    def load(filepath: str = "./sentences/sentences.jsonl") -> List[str]:
        print("Loading sentences...")
        sentences = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                sentences.append(data["text"])

        print(f"Loaded {len(sentences):,} sentences.")
        return sentences


class BatchManager:
    def __init__(self, output_dir: str = "./training_data"):
        self.output_dir = output_dir

    def save_batch(self, batch_data: List[Dict], batch_num: int):
        os.makedirs(self.output_dir, exist_ok=True)
        filename = f"{self.output_dir}/training_data_batch_{batch_num}.jsonl"

        with open(filename, 'w') as f:
            for example in batch_data:
                f.write(json.dumps(example) + '\n')

        print(f"Saved batch {batch_num} with {len(batch_data)} examples")

    def load_resume_state(self) -> Tuple[int, List[Dict]]:
        print("Checking for existing training data...")

        training_data_files = glob.glob(
            f"{self.output_dir}/training_data_batch_*.jsonl")
        if not training_data_files:
            print("No existing training data found.")
            return 0, []

        latest_batch = 0
        for f in training_data_files:
            match = re.search(r'batch_(\d+)', f)
            if match:
                latest_batch = max(latest_batch, int(match.group(1)))

        if latest_batch == 0:
            print("No existing training data found.")
            return 0, []

        with open(f"{self.output_dir}/training_data_batch_{latest_batch}.jsonl", 'r') as f:
            current_batch = [json.loads(line) for line in f]

        print(
            f"Resuming from batch {latest_batch} with {len(current_batch)} examples.")
        return latest_batch, current_batch


class TrainingDataGenerator:
    def __init__(self, model_name: str = "google/gemma-3-4b-it"):
        self.model_manager = ModelManager(model_name)
        self.text_generator = None
        self.batch_manager = BatchManager()
        self.telemetry_logger = TelemetryLogger()

    def initialize(self):
        self.model_manager.load()
        self.text_generator = TextGenerator(self.model_manager)

    def generate(self, num_examples: int, batch_size: int):
        sentences = SentenceLoader.load()
        batch_number, current_batch = self.batch_manager.load_resume_state()

        generated_count = batch_number * batch_size + len(current_batch)
        start_time = time.time()
        batch_start_time = time.time()
        last_generation_time = 0

        try:
            while generated_count < num_examples:
                sentence = sentences[generated_count % len(sentences)]

                example_start_time = time.time()
                example = self.text_generator.generate_training_example(
                    sentence)
                generation_time = time.time() - example_start_time

                if ResponseValidator.is_valid(example["generated_response"]):
                    current_batch.append(example)
                    generated_count += 1
                    last_generation_time = generation_time
                    print(
                        f"Generated {generated_count}/{num_examples}, time: {generation_time:.2f}s")

                    if len(current_batch) >= batch_size:
                        self.batch_manager.save_batch(
                            current_batch, batch_number)
                        self.telemetry_logger.update(
                            generated_count, num_examples, last_generation_time,
                            start_time, batch_start_time, len(current_batch)
                        )

                        current_batch = []
                        batch_number += 1
                        batch_start_time = time.time()
                else:
                    print("Invalid JSON, skipping this example.")

        except KeyboardInterrupt:
            print("Interrupted! Saving current batch...")
            if current_batch:
                self.batch_manager.save_batch(current_batch, batch_number)
                self.telemetry_logger.update(
                    generated_count, num_examples, last_generation_time,
                    start_time, batch_start_time, len(current_batch)
                )
            self.telemetry_logger.finalize(
                generated_count, num_examples, start_time, interrupted=True)
            return

        if current_batch:
            self.batch_manager.save_batch(current_batch, batch_number)
            self.telemetry_logger.update(
                generated_count, num_examples, last_generation_time,
                start_time, batch_start_time, len(current_batch)
            )

        self.telemetry_logger.finalize(
            generated_count, num_examples, start_time, interrupted=False)


generator = TrainingDataGenerator()
generator.initialize()
generator.generate(num_examples=400_000, batch_size=32)
