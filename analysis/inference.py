import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))

from time import time, sleep
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM
from training.model import Transformer
from shared import (
    get_device,
    get_training_run_dir,
    Utilities,
    INFERENCE_TEMPERATURE,
    MIN_SENTENCE_LENGTH,
    MAX_SENTENCE_LENGTH,
    PROMPT_DELIMITER,
    DOMAIN_MAX_GENERATION_STEPS,
)

DOMAIN_STOP_TOKEN = {
    "reddit_comment_sentiment": "}",
    "math_word_problem": ";",
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
    for path in sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"), key=lambda x: int(x.stem.split('_')[-1])):
        result.append((path.stem.split('_')[-1], path))
    return result


def create_teacher_prompt(domain: str, sentence: str) -> str:
    if domain == "math_word_problem":
        return Utilities.create_math_word_problem_prompt(sentence)
    return Utilities.create_reddit_sentiment_prompt(sentence)


DOMAIN_MAX_INPUT_TOKENS = {
    "reddit_comment_sentiment": MAX_SENTENCE_LENGTH,
    "math_word_problem": 500,
}


def validate_sentence(tokenizer, sentence, domain):
    max_tokens = DOMAIN_MAX_INPUT_TOKENS.get(domain, MAX_SENTENCE_LENGTH)
    token_count = len(tokenizer.encode(sentence, add_special_tokens=False))
    if token_count < MIN_SENTENCE_LENGTH:
        return False, f"Too short: {token_count} tokens (min {MIN_SENTENCE_LENGTH})"
    if token_count > max_tokens:
        return False, f"Too long: {token_count} tokens (max {max_tokens})"
    return True, token_count


class InferenceLogger:
    def __init__(self, run_name, sentences, checkpoint_name):
        self.log_dir = Path(get_training_run_dir(run_name)) / "inferences"
        self.log_dir.mkdir(exist_ok=True)
        self.log_path = self.log_dir / f"inference_{int(time() * 1000)}.txt"
        self.sentences = sentences
        self.checkpoint_name = checkpoint_name

    def create_file(self):
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("INFERENCE LOG\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Sentences: {len(self.sentences)}\n")
            f.write(f"Student: {self.checkpoint_name}\n")
        print(f"\nLog: {self.log_path}")
        print("Opening in 2 seconds...")
        sleep(2)

    def write(self, text):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(text)
            f.flush()

    def append_token(self, token):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(token)
            f.flush()


def generate_teacher(model, tokenizer, sentence, domain, device, logger):
    prompt = create_teacher_prompt(domain, sentence)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    stop_token = DOMAIN_STOP_TOKEN[domain]
    max_tokens = DOMAIN_MAX_GENERATION_STEPS[domain]

    gen_start = time()
    generated_ids = []

    with torch.no_grad():
        past_key_values = None
        current_input = inputs.input_ids

        for _ in range(max_tokens):
            outputs = model(current_input, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

            next_token = torch.argmax(logits, dim=-1)
            generated_ids.append(next_token.item())

            token_str = tokenizer.decode([next_token.item()])
            logger.append_token(token_str)

            if next_token.item() == tokenizer.eos_token_id or token_str == stop_token:
                break

            current_input = next_token.unsqueeze(0)

    gen_time = time() - gen_start
    token_count = len(generated_ids)
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    tps = token_count / gen_time

    logger.write(f"\n-> {token_count} tokens | {gen_time:.3f}s | {tps:.1f} TPS\n")
    return {"gen_time": gen_time, "tokens": token_count, "tps": tps, "response": response}


def generate_student(model, tokenizer, output_token_ids, sentence, domain, device, logger):
    prompt = sentence + PROMPT_DELIMITER
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    stop_token = DOMAIN_STOP_TOKEN[domain]
    max_tokens = DOMAIN_MAX_GENERATION_STEPS[domain]

    generated_ids = []
    gen_start = time()

    with torch.no_grad():
        for _ in range(max_tokens):
            context = input_ids + generated_ids
            tensor = torch.tensor([context], dtype=torch.long, device=device)
            logits = model(tensor)[:, -1, :][0]

            if INFERENCE_TEMPERATURE:
                probs = F.softmax(logits / INFERENCE_TEMPERATURE, dim=-1)
                pred_index = torch.multinomial(probs, num_samples=1).item()
            else:
                pred_index = torch.argmax(logits).item()

            pred_token_id = output_token_ids[pred_index]
            generated_ids.append(pred_token_id)

            token_str = tokenizer.decode([pred_token_id])
            logger.append_token(token_str)

            if token_str == stop_token:
                break

    gen_time = time() - gen_start
    token_count = len(generated_ids)
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    tps = token_count / gen_time

    logger.write(f"\n-> {token_count} tokens | {gen_time:.3f}s | {tps:.1f} TPS\n")
    return {"gen_time": gen_time, "tokens": token_count, "tps": tps, "response": response}


def main():
    print("=== Inference Comparison ===\n")

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
        print("Invalid selection.")
        return

    domain = run_config["domain"]
    teacher_model = run_config["teacher_model"]
    print(f"\nSelected: {run_name} (domain: {domain})")

    checkpoints = get_checkpoints(run_name)
    if not checkpoints:
        print("No checkpoints found for this run.")
        return

    print(f"Checkpoints: {', '.join(name for name, _ in checkpoints)}")
    checkpoint_selection = input("Select: ").strip().lower()

    checkpoint_path = None
    for name, path in checkpoints:
        if name == checkpoint_selection:
            checkpoint_path = path
            break

    if not checkpoint_path:
        print(f"'{checkpoint_selection}' not found.")
        return

    tokenizer = AutoTokenizer.from_pretrained(teacher_model)

    sentences = []
    max_tokens = DOMAIN_MAX_INPUT_TOKENS.get(domain, MAX_SENTENCE_LENGTH)
    print(f"\nEnter sentences ({MIN_SENTENCE_LENGTH}-{max_tokens} tokens, use \\n for newlines). Empty line to finish.\n")

    while True:
        sentence = input(f"[{len(sentences) + 1}] ").strip()
        if not sentence:
            if sentences:
                break
            print("Enter at least one sentence.")
            continue

        sentence = sentence.replace("\\n", "\n")

        valid, result = validate_sentence(tokenizer, sentence, domain)
        if valid:
            sentences.append(sentence)
            print(f"    ({result} tokens)")
        else:
            print(f"    {result}")

    print(f"\n{len(sentences)} sentence(s)")

    device = get_device()
    print(f"Device: {device}")

    logger = InferenceLogger(run_name, sentences, checkpoint_path.name)
    logger.create_file()

    # Teacher: all sentences
    logger.write("\n" + "=" * 60 + "\n")
    logger.write(f"TEACHER: {teacher_model}\n")
    logger.write("=" * 60 + "\n")

    print(f"\n[Teacher] Loading {teacher_model}...")
    load_start = time()
    teacher_model_instance = AutoModelForCausalLM.from_pretrained(teacher_model, torch_dtype=torch.bfloat16, device_map=device)
    teacher_model_instance.eval()
    teacher_load = time() - load_start
    print(f"[Teacher] Loaded in {teacher_load:.2f}s")

    logger.write(f"\nLoaded in {teacher_load:.2f}s\n")

    teacher_results = []
    for idx, sentence in enumerate(sentences):
        print(f"[Teacher] {idx + 1}/{len(sentences)}")
        logger.write(f"\n[{idx + 1}] {sentence}\n")
        result = generate_teacher(teacher_model_instance, tokenizer, sentence, domain, device, logger)
        teacher_results.append(result)

    del teacher_model_instance
    torch.cuda.empty_cache() if device == "cuda" else None

    # Student: all sentences
    logger.write("\n" + "=" * 60 + "\n")
    logger.write(f"STUDENT: {checkpoint_path.name}\n")
    logger.write("=" * 60 + "\n")

    print(f"\n[Student] Loading...")
    load_start = time()
    student_model = Transformer(
        domain=domain,
        teacher_model=run_config["teacher_model"],
        hidden_dim=run_config["hidden_dim"],
        num_layers=run_config["num_layers"],
        num_heads=run_config["num_heads"],
        dropout=run_config.get("dropout", 0.15),
        auxiliary_token_percentage=run_config.get("auxiliary_token_percentage", 1.0),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    student_model.load_state_dict(checkpoint["model_state_dict"])
    student_model.eval()
    student_load = time() - load_start
    print(f"[Student] Loaded in {student_load:.2f}s")

    logger.write(f"\nLoaded in {student_load:.2f}s\n")

    student_tokenizer = student_model.tokenizer
    output_token_ids = student_model.output_token_ids

    student_results = []
    for idx, sentence in enumerate(sentences):
        print(f"[Student] {idx + 1}/{len(sentences)}")
        logger.write(f"\n[{idx + 1}] {sentence}\n")
        result = generate_student(student_model, student_tokenizer, output_token_ids, sentence, domain, device, logger)
        student_results.append(result)

    # Summary
    n = len(sentences)
    matches = sum(1 for t, s in zip(teacher_results, student_results) if t["response"].strip() == s["response"].strip())

    avg_t_gen = sum(r["gen_time"] for r in teacher_results) / n
    avg_s_gen = sum(r["gen_time"] for r in student_results) / n
    avg_t_tps = sum(r["tps"] for r in teacher_results) / n
    avg_s_tps = sum(r["tps"] for r in student_results) / n

    gen_speedup = avg_t_gen / avg_s_gen
    tps_speedup = avg_s_tps / avg_t_tps
    load_speedup = teacher_load / student_load

    summary = f"""
{'='*60}
SUMMARY ({n} sentences)
{'='*60}

Accuracy: {matches}/{n} ({matches/n*100:.1f}%)

                    Teacher      Student      Speedup
{'-'*60}
Load time          {teacher_load:>7.2f}s     {student_load:>7.2f}s      {load_speedup:>5.1f}x
Avg generation     {avg_t_gen:>7.3f}s     {avg_s_gen:>7.3f}s      {gen_speedup:>5.1f}x
Avg TPS            {avg_t_tps:>7.1f}      {avg_s_tps:>7.1f}       {tps_speedup:>5.1f}x
"""
    logger.write(summary)

    print("\n" + "=" * 50)
    print(f"SUMMARY ({n} sentences)")
    print("=" * 50)
    print(f"\nAccuracy: {matches}/{n} ({matches/n*100:.1f}%)")
    print(f"\n{'':18} {'Teacher':>10} {'Student':>10} {'Speedup':>10}")
    print("-" * 50)
    print(f"{'Load time':18} {teacher_load:>9.2f}s {student_load:>9.2f}s {load_speedup:>9.1f}x")
    print(f"{'Avg generation':18} {avg_t_gen:>9.3f}s {avg_s_gen:>9.3f}s {gen_speedup:>9.1f}x")
    print(f"{'Avg TPS':18} {avg_t_tps:>10.1f} {avg_s_tps:>10.1f} {tps_speedup:>9.1f}x")
    print(f"\nLog: {logger.log_path}")


if __name__ == "__main__":
    main()
