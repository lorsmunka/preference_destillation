import os
import time
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import glob
from datetime import datetime, timedelta


def create_evaluation_prompt(sentence):
    prompt = f"""Analyze the following sentence and return a JSON object with these evaluations:

    Sentence: "{sentence}"

    Return JSON format:
    {{
        "tone": "aggressive | rude | neutral | polite | friendly",
        "sentiment": "negative | neutral | positive",
        "safety": "harmful | safe",
        "toxicity": "toxic | respectful"
    }}

    JSON:
    """
    return prompt


def create_telemetry_file():
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


def update_telemetry(telemetry_file, generated_count, total_examples, generation_time, start_time):
    elapsed_time = time.time() - start_time
    avg_time_per_example = elapsed_time / \
        generated_count if generated_count > 0 else 0
    remaining_examples = total_examples - generated_count
    estimated_remaining_time = avg_time_per_example * remaining_examples
    total_estimated_time = avg_time_per_example * total_examples

    progress_percent = (generated_count / total_examples) * 100

    with open(telemetry_file, 'a') as f:
        f.write(
            f"Example #{generated_count:,} | Generation Time: {format_duration(generation_time)}\n")
        f.write(
            f"Progress: {generated_count:,}/{total_examples:,} ({progress_percent:.1f}%)\n")
        f.write(
            f"Average Time/Example: {format_duration(avg_time_per_example)}\n")
        f.write(f"Elapsed Time: {format_duration(elapsed_time)}\n")
        f.write(
            f"Estimated Total Runtime: {format_duration(total_estimated_time)}\n")
        f.write(
            f"Estimated Time Remaining: {format_duration(estimated_remaining_time)}\n")
        f.write("-" * 60 + "\n")


def finalize_telemetry(telemetry_file, generated_count, total_examples, start_time, interrupted=False):
    elapsed_time = time.time() - start_time
    status = "INTERRUPTED" if interrupted else "COMPLETED"

    with open(telemetry_file, 'a') as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"RUN {status}\n")
        f.write("=" * 60 + "\n")
        f.write(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(
            f"Total Examples Generated: {generated_count:,}/{total_examples:,}\n")
        f.write(f"Total Runtime: {format_duration(elapsed_time)}\n")
        if generated_count > 0:
            avg_time = elapsed_time / generated_count
            f.write(f"Average Time per Example: {format_duration(avg_time)}\n")
        f.write("=" * 60 + "\n")


def format_duration(seconds):
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


def get_output_vocabulary():
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")

    example_result = """
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

    all_text = example_result + " " + " ".join(auxiliary_tokens)

    tokens = tokenizer.tokenize(all_text)
    return sorted(set(tokens))


output_vocabulary = get_output_vocabulary()


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
    # model = AutoModelForCausalLM.from_pretrained(
    #     "google/gemma-3-4b-it", device_map="auto")

    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-4b-it",
        device_map="auto",
        dtype=torch.bfloat16
    )

    return tokenizer, model


def prepare_inputs_for_gpu(tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    return inputs


def extract_logits_with_other_token(tokenizer, logits):
    token_names = []
    token_logits = []
    used_indices = set()

    for token_text in output_vocabulary:
        token_ids = tokenizer.convert_tokens_to_ids([token_text])
        if token_ids[0] != tokenizer.unk_token_id:
            token_id = token_ids[0]
            token_logits.append(logits[token_id].item())
            token_names.append(token_text)
            used_indices.add(token_id)

    other_indices = [i for i in range(len(logits)) if i not in used_indices]
    if other_indices:
        other_logits = logits[other_indices]
        max_logit = other_logits.max()
        other_logit_sum = max_logit + \
            torch.log(torch.sum(torch.exp(other_logits - max_logit)))
        token_logits.append(other_logit_sum.item())
        token_names.append("__OTHER__")

    return token_names, token_logits


def sort_by_logits(token_names, token_logits):
    token_logit_pairs = list(zip(token_names, token_logits))
    token_logit_pairs.sort(key=lambda x: x[1], reverse=True)
    return {token: logit for token, logit in token_logit_pairs}


def load_sentences():
    print("Loading sentences...")
    sentences = []
    with open("./sentences/sentences.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            sentences.append(data["text"])

    print(f"Loaded {len(sentences):,} sentences.")
    return sentences


def load_resume_state():
    print("Checking for existing training data...")

    training_data_files = glob.glob(
        "./training_data/training_data_batch_*.jsonl")
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

    with open(f"./training_data/training_data_batch_{latest_batch}.jsonl", 'r') as f:
        current_batch = [json.loads(line) for line in f]

    print(
        f"Resuming from batch {latest_batch} with {len(current_batch)} examples.")
    return latest_batch, current_batch


def save_training_batch(batch_data, batch_num):
    os.makedirs("./training_data", exist_ok=True)
    filename = f"./training_data/training_data_batch_{batch_num}.jsonl"

    with open(filename, 'w') as f:
        for example in batch_data:
            f.write(json.dumps(example) + '\n')

    print(f"Saved batch {batch_num} with {len(batch_data)} examples")


def is_valid_json_response(text):
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError as e:
        return False, f"JSON parsing failed: {e}. Input text: '{text}'"

    if not isinstance(parsed, dict):
        return False, f"Expected dict, got {type(parsed).__name__}. Parsed content: {parsed}"

    required_keys = {"tone", "sentiment", "safety", "toxicity"}
    missing_keys = required_keys - set(parsed.keys())
    if missing_keys:
        return False, f"Missing required keys: {missing_keys}. Found keys: {set(parsed.keys())}"

    valid_values = {
        "tone": {"aggressive", "rude", "neutral", "polite", "friendly"},
        "sentiment": {"negative", "neutral", "positive"},
        "safety": {"harmful", "safe"},
        "toxicity": {"toxic", "respectful"}
    }

    for key, expected_values in valid_values.items():
        if parsed[key] not in expected_values:
            return False, f"Invalid value for '{key}': '{parsed[key]}'. Expected one of: {expected_values}"

    extra_keys = set(parsed.keys()) - required_keys
    if extra_keys:
        return False, f"Unexpected extra keys found: {extra_keys}. Only expected: {required_keys}"

    return True, "Valid JSON response"


def should_stop_generation(token, accumulated_text):
    if token.strip() == "}":
        return True
    if len(accumulated_text) > 500:
        return True
    return False


def generate_single_step(model, tokenizer, current_sequence):
    current_inputs = {
        'input_ids': current_sequence,
        'attention_mask': torch.ones_like(current_sequence)
    }

    with torch.no_grad():
        logits = model(**current_inputs).logits[0, -1, :]

    token_names, token_logits = extract_logits_with_other_token(
        tokenizer, logits)
    step_probs = sort_by_logits(token_names, token_logits)

    next_token_id = torch.argmax(logits).item()
    next_token = tokenizer.decode([next_token_id])

    new_sequence = torch.cat([current_sequence, torch.tensor(
        [[next_token_id]], device=current_sequence.device)], dim=1)

    return next_token, step_probs, new_sequence


def generate_training_example(sentence, model, tokenizer):
    prompt = create_evaluation_prompt(sentence)
    inputs = prepare_inputs_for_gpu(tokenizer, prompt)

    steps = []
    generated_text = ""
    current_sequence = inputs['input_ids']

    while len(steps) < 100:
        next_token, step_probs, new_sequence = generate_single_step(
            model, tokenizer, current_sequence)

        steps.append({
            "token": next_token,
            "probabilities": step_probs
        })

        generated_text += next_token
        current_sequence = new_sequence

        if should_stop_generation(next_token, generated_text):
            break

    return {
        "sentence": sentence,
        "generated_response": generated_text.strip(),
        "steps": steps
    }


def generate_training_data(num_examples, batch_size):
    sentences = load_sentences()
    tokenizer, model = load_model_and_tokenizer()
    batch_number, current_batch = load_resume_state()

    generated_count = batch_number * batch_size + len(current_batch)
    start_time = time.time()
    telemetry_file = create_telemetry_file()
    batch_generation_times = []

    try:
        while generated_count < num_examples:
            sentence = sentences[generated_count % len(sentences)]

            example_start_time = time.time()
            example = generate_training_example(sentence, model, tokenizer)
            generation_time = time.time() - example_start_time

            if is_valid_json_response(example["generated_response"]):
                current_batch.append(example)
                generated_count += 1
                batch_generation_times.append(generation_time)
                print(f"Generated {generated_count}/{num_examples}")

                save_training_batch(current_batch, batch_number)

                if len(current_batch) >= batch_size:
                    batch_total_time = sum(batch_generation_times)
                    update_telemetry(telemetry_file, generated_count,
                                     num_examples, batch_total_time, start_time)

                    current_batch = []
                    batch_number += 1
                    batch_generation_times = []
            else:
                print("Invalid JSON, retrying...")

    except KeyboardInterrupt:
        print("Interrupted! Saving current batch...")
        if current_batch:
            save_training_batch(current_batch, batch_number)
            if batch_generation_times:
                batch_total_time = sum(batch_generation_times)
                update_telemetry(telemetry_file, generated_count,
                                 num_examples, batch_total_time, start_time)
        finalize_telemetry(telemetry_file, generated_count,
                           num_examples, start_time, interrupted=True)

    if current_batch:
        save_training_batch(current_batch, batch_number)
        if batch_generation_times:
            batch_total_time = sum(batch_generation_times)
            update_telemetry(telemetry_file, generated_count,
                             num_examples, batch_total_time, start_time)

    finalize_telemetry(telemetry_file, generated_count,
                       num_examples, start_time, interrupted=False)


generate_training_data(num_examples=400_000, batch_size=32)
