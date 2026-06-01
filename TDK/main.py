import html
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


TDK_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TDK_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "training"))

from shared import PROMPT_DELIMITER, Utilities, get_batches_dir
from training.model import Transformer


RUN_NAME = "exp-kl99to50-t1-1"
CHECKPOINT_NAME = "checkpoint_epoch_2.pt"
STOP_TOKEN = "}"
MAX_NEW_TOKENS = 50


def model_slug(model_name):
    return model_name.replace("/", "_")


def title_separator(title):
    line = "=" * len(title)
    return f"{line}\n{title}\n{line}\n"


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(torch_device):
    if torch_device.type == "mps":
        torch.mps.synchronize()


def tps(token_count, seconds):
    if seconds <= 0:
        return 0.0
    return token_count / seconds


def speedup(value, baseline):
    if value <= 0 or baseline <= 0:
        return "-"
    return f"{value / baseline:.1f}x"


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_checkpoint_and_vocabulary(run_config):
    checkpoint_path = PROJECT_DIR / "runs" / RUN_NAME / "checkpoints" / CHECKPOINT_NAME
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    expected_input_size = checkpoint["model_state_dict"]["input_embedding.weight"].shape[0]

    expected_path = PROJECT_DIR / get_batches_dir(run_config["domain"], run_config["teacher_model"]) / "input_vocabulary.json"
    possible_paths = [
        expected_path,
        PROJECT_DIR / "runs" / RUN_NAME / "input_vocabulary.json",
        TDK_DIR / "input_vocabulary.json",
        Path.home() / "Desktop" / RUN_NAME / "input_vocabulary.json",
    ]

    vocabulary_path = None
    for path in possible_paths:
        if path.exists():
            vocabulary_path = path
            break

    if vocabulary_path is None:
        raise FileNotFoundError(
            f"Missing input_vocabulary.json for {RUN_NAME} / {CHECKPOINT_NAME}.\n"
            f"The student cannot tokenize prompts correctly without it.\n"
            f"Put it here: {expected_path}\n"
            f"Or here: {TDK_DIR / 'input_vocabulary.json'}"
        )

    vocabulary = load_json(vocabulary_path)

    actual_input_size = vocabulary["compact_vocab_size"]
    if actual_input_size != expected_input_size:
        raise ValueError(f"Wrong input_vocabulary.json size: {actual_input_size}, expected {expected_input_size}")

    print(f"Using input vocabulary: {vocabulary_path}")
    return checkpoint, vocabulary


def save_token(file, token):
    print(token, end="", flush=True)
    file.write(token)
    file.flush()


def should_stop(tokenizer, token_id, token_text, generated_token_ids):
    if token_id == tokenizer.eos_token_id:
        return True
    if token_text == STOP_TOKEN:
        return True
    return STOP_TOKEN in tokenizer.decode(generated_token_ids, skip_special_tokens=False)


def generate_teacher(model, tokenizer, sentence, torch_device, file):
    prompt = Utilities.create_reddit_sentiment_prompt(sentence)
    inputs = tokenizer(prompt, return_tensors="pt").to(torch_device)
    token_ids = []
    token_texts = []

    sync(torch_device)
    started = perf_counter()

    with torch.no_grad():
        past_key_values = None
        current_input = inputs.input_ids

        for _ in range(MAX_NEW_TOKENS):
            output = model(current_input, past_key_values=past_key_values, use_cache=True)
            past_key_values = output.past_key_values
            next_token = torch.argmax(output.logits[:, -1, :], dim=-1)
            token_id = next_token.item()
            token_text = tokenizer.decode([token_id], skip_special_tokens=False)

            token_ids.append(token_id)
            token_texts.append(token_text)
            save_token(file, token_text)

            if should_stop(tokenizer, token_id, token_text, token_ids):
                break

            current_input = next_token.unsqueeze(0)

    sync(torch_device)
    seconds = perf_counter() - started
    return {
        "text": tokenizer.decode(token_ids, skip_special_tokens=True),
        "ids": token_ids,
        "tokens": token_texts,
        "seconds": seconds,
        "tps": tps(len(token_ids), seconds),
    }


def next_student_token(model, context, torch_device):
    tensor = torch.tensor([context], dtype=torch.long, device=torch_device)
    logits = model(tensor)[:, -1, :][0]
    token_index = torch.argmax(logits).item()
    return model.output_token_ids[token_index]


def generate_student(model, tokenizer, sentence, torch_device, file):
    prompt_ids = tokenizer.encode(sentence + PROMPT_DELIMITER, add_special_tokens=False)
    compact_prompt_ids = model.remap_input_tokens(prompt_ids)
    generated_ids = []
    generated_compact_ids = []
    generated_tokens = []

    sync(torch_device)
    started = perf_counter()

    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            context = compact_prompt_ids + generated_compact_ids
            token_id = next_student_token(model, context, torch_device)
            token_text = tokenizer.decode([token_id], skip_special_tokens=False)

            generated_ids.append(token_id)
            generated_tokens.append(token_text)
            generated_compact_ids.append(model.remap_input_tokens([token_id])[0])
            save_token(file, token_text)

            if should_stop(tokenizer, token_id, token_text, generated_ids):
                break

    sync(torch_device)
    seconds = perf_counter() - started
    return {
        "text": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "ids": generated_ids,
        "tokens": generated_tokens,
        "seconds": seconds,
        "tps": tps(len(generated_ids), seconds),
    }


def teacher_forced_student(model, tokenizer, sentence, teacher_ids, torch_device):
    prompt_ids = tokenizer.encode(sentence + PROMPT_DELIMITER, add_special_tokens=False)
    compact_prompt_ids = model.remap_input_tokens(prompt_ids)
    predicted_ids = []
    predicted_tokens = []
    matches = []

    sync(torch_device)
    started = perf_counter()

    with torch.no_grad():
        for token_index, teacher_id in enumerate(teacher_ids):
            teacher_context = teacher_ids[:token_index]
            compact_teacher_context = model.remap_input_tokens(teacher_context)
            predicted_id = next_student_token(model, compact_prompt_ids + compact_teacher_context, torch_device)

            predicted_ids.append(predicted_id)
            predicted_tokens.append(tokenizer.decode([predicted_id], skip_special_tokens=False))
            matches.append(predicted_id == teacher_id)

    sync(torch_device)
    seconds = perf_counter() - started
    return {
        "text": tokenizer.decode(predicted_ids, skip_special_tokens=True),
        "tokens": predicted_tokens,
        "matches": matches,
        "seconds": seconds,
        "tps": tps(len(predicted_ids), seconds),
        "accuracy": sum(matches) / len(teacher_ids) if teacher_ids else 0.0,
    }


def compare(student_ids, teacher_ids):
    matches = []
    for token_index, student_id in enumerate(student_ids):
        matches.append(token_index < len(teacher_ids) and student_id == teacher_ids[token_index])
    accuracy = sum(matches) / len(teacher_ids) if teacher_ids else 0.0
    return matches, accuracy


def colored(tokens, matches):
    output = []
    for token, is_good in zip(tokens, matches):
        class_name = "good" if is_good else "bad"
        output.append(f'<span class="{class_name}">{html.escape(token)}</span>')
    return "".join(output)


def write_html(path, rows, summary, report_title):
    parts = []
    for row_number, row in enumerate(rows, start=1):
        parts.append(f"""
<section>
<h2>{row_number}. {html.escape(row["sentence"])}</h2>
<h3>Teacher</h3><pre>{html.escape(row["teacher"]["text"])}</pre>
<h3>Student</h3><pre>{colored(row["student"]["tokens"], row["student_matches"])}</pre>
<h3>Teacher-forced</h3><pre>{colored(row["forced"]["tokens"], row["forced"]["matches"])}</pre>
</section>
""")

    table_rows = "\n".join(
        f"<tr><td>{name}</td><td>{load_time:.2f}s</td><td>{generation_time:.2f}s</td><td>{tokens}</td><td>{speed:.1f}</td><td>{speedup_text}</td><td>{match}</td></tr>"
        for name, load_time, generation_time, tokens, speed, speedup_text, match in summary
    )

    path.write_text(f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{html.escape(report_title)}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 28px; color: #1f2933; background: #f7f8fa; }}
main {{ max-width: 960px; margin: auto; }}
section {{ border-bottom: 1px solid #d8dee6; margin-bottom: 22px; padding-bottom: 22px; }}
h1 {{ font-size: 24px; margin: 0 0 18px; }}
h2 {{ font-size: 17px; margin: 0 0 12px; }}
h3 {{ font-size: 13px; margin: 12px 0 6px; color: #5b6470; }}
pre {{ white-space: pre-wrap; word-break: break-word; background: white; border: 1px solid #d8dee6; border-radius: 6px; padding: 12px; line-height: 1.5; }}
.good {{ background: #d9f8df; color: #116329; }}
.bad {{ background: #ffe1df; color: #9f1c1c; }}
table {{ width: 100%; border-collapse: collapse; background: white; }}
td, th {{ padding: 8px 10px; border-bottom: 1px solid #e4e8ee; text-align: right; }}
td:first-child, th:first-child {{ text-align: left; }}
</style>
</head>
<body>
<main>
<h1>{html.escape(report_title)}</h1>
{''.join(parts)}
<table>
<tr><th>Run</th><th>Load</th><th>Generation</th><th>Tokens</th><th>TPS</th><th>Speedup</th><th>Match</th></tr>
{table_rows}
</table>
</main>
</body>
</html>
""", encoding="utf-8")


def ask_sentences():
    sentences = []
    print("\nSentences. Empty enter starts.\n")
    while True:
        sentence = input(f"{len(sentences) + 1}: ").strip()
        if sentence:
            sentences.append(sentence.replace("\\n", "\n"))
        elif sentences:
            return sentences
        else:
            print("Write at least one.")


def main():
    torch.set_grad_enabled(False)

    run_config = load_json(PROJECT_DIR / "runs" / RUN_NAME / "info.json")
    checkpoint, vocabulary = load_checkpoint_and_vocabulary(run_config)
    torch_device = device()
    tokenizer = AutoTokenizer.from_pretrained(run_config["teacher_model"], local_files_only=True)
    teacher_slug = model_slug(run_config["teacher_model"])
    student_slug = f"{RUN_NAME} / {CHECKPOINT_NAME}"
    report_title = f"{teacher_slug} / {student_slug}"
    sentences = ask_sentences()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    test_dir = TDK_DIR / "tests" / timestamp
    test_dir.mkdir(parents=True, exist_ok=True)
    text_path = test_dir / "test.txt"
    html_path = test_dir / "test.html"

    rows = []

    with open(text_path, "w", encoding="utf-8", buffering=1) as file:
        file.write(f"{timestamp} | {torch_device.type}\n\n")
        file.write(title_separator(f"Teacher: {teacher_slug}"))

        print("\nLoading teacher...")
        load_started = perf_counter()
        teacher = AutoModelForCausalLM.from_pretrained(
            run_config["teacher_model"],
            dtype=torch.bfloat16,
            local_files_only=True,
        )
        teacher.to(torch_device)
        teacher.eval()
        sync(torch_device)
        teacher_load = perf_counter() - load_started
        print(f"Teacher load: {teacher_load:.2f}s")

        for sentence_number, sentence in enumerate(sentences, start=1):
            print(f"\nTeacher {sentence_number}/{len(sentences)}")
            file.write(f"\n[{sentence_number}] {sentence}\nTeacher:\n")
            teacher_result = generate_teacher(teacher, tokenizer, sentence, torch_device, file)
            file.write(f"\n{len(teacher_result['ids'])} tokens | {teacher_result['seconds']:.2f}s | {teacher_result['tps']:.1f} TPS\n")
            rows.append({"sentence": sentence, "teacher": teacher_result})

        del teacher
        torch.mps.empty_cache() if torch_device.type == "mps" else None

        file.write("\n" + title_separator(f"Student: {student_slug}"))

        print("\nLoading student...")
        load_started = perf_counter()
        student = Transformer(
            domain=run_config["domain"],
            teacher_model=run_config["teacher_model"],
            hidden_dim=run_config["hidden_dim"],
            num_layers=run_config["num_layers"],
            num_heads=run_config["num_heads"],
            dropout=run_config["dropout"],
            auxiliary_token_percentage=run_config["auxiliary_token_percentage"],
            input_vocabulary=vocabulary,
        )
        student.load_state_dict(checkpoint["model_state_dict"])
        student.to(torch_device)
        student.eval()
        sync(torch_device)
        student_load = perf_counter() - load_started
        print(f"Student load: {student_load:.2f}s")

        for row_number, row in enumerate(rows, start=1):
            print(f"\nStudent {row_number}/{len(rows)}")
            file.write(f"\n[{row_number}] {row['sentence']}\nStudent:\n")
            student_result = generate_student(student, student.tokenizer, row["sentence"], torch_device, file)
            student_matches, student_accuracy = compare(student_result["ids"], row["teacher"]["ids"])
            row["student"] = student_result
            row["student_matches"] = student_matches
            row["student_accuracy"] = student_accuracy
            file.write(f"\n{len(student_result['ids'])} tokens | {student_result['seconds']:.2f}s | {student_result['tps']:.1f} TPS | match {student_accuracy:.1%}\n")

        file.write("\n" + title_separator(f"Teacher-forced: {student_slug}"))
        for row_number, row in enumerate(rows, start=1):
            forced = teacher_forced_student(student, student.tokenizer, row["sentence"], row["teacher"]["ids"], torch_device)
            row["forced"] = forced
            file.write(f"[{row_number}] {forced['text']}\n{len(forced['tokens'])} tokens | {forced['seconds']:.2f}s | {forced['tps']:.1f} TPS | match {forced['accuracy']:.1%}\n")

        teacher_generation = sum(row["teacher"]["seconds"] for row in rows)
        student_generation = sum(row["student"]["seconds"] for row in rows)
        forced_generation = sum(row["forced"]["seconds"] for row in rows)
        teacher_tokens = sum(len(row["teacher"]["ids"]) for row in rows)
        student_tokens = sum(len(row["student"]["ids"]) for row in rows)
        forced_tokens = sum(len(row["forced"]["tokens"]) for row in rows)
        student_match = sum(row["student_accuracy"] for row in rows) / len(rows)
        forced_match = sum(row["forced"]["accuracy"] for row in rows) / len(rows)
        teacher_tps = tps(teacher_tokens, teacher_generation)
        student_tps = tps(student_tokens, student_generation)
        forced_tps = tps(forced_tokens, forced_generation)
        total_generation = teacher_generation + student_generation + forced_generation
        total_tokens = teacher_tokens + student_tokens + forced_tokens
        total_tps = tps(total_tokens, total_generation)

        summary = [
            ("Teacher", teacher_load, teacher_generation, teacher_tokens, teacher_tps, "1.0x", "-"),
            ("Student", student_load, student_generation, student_tokens, student_tps, speedup(student_tps, teacher_tps), f"{student_match:.1%}"),
            ("Teacher-forced", 0.0, forced_generation, forced_tokens, forced_tps, "-", f"{forced_match:.1%}"),
            ("Total", teacher_load + student_load, total_generation, total_tokens, total_tps, "-", "-"),
        ]

        file.write("\nSummary\n")
        file.write(f"{'Run':<16} {'Load':>8} {'Gen':>8} {'Tokens':>8} {'TPS':>8} {'Speedup':>8} {'Match':>8}\n")
        for name, load_time, generation_time, token_count, speed, speedup_text, match in summary:
            file.write(f"{name:<16} {load_time:>7.2f}s {generation_time:>7.2f}s {token_count:>8} {speed:>8.1f} {speedup_text:>8} {match:>8}\n")

    write_html(html_path, rows, summary, report_title)

    print(f"\nTotal load: {teacher_load + student_load:.2f}s")
    print(f"Total generation: {teacher_generation + student_generation + forced_generation:.2f}s")
    print(f"Text: {text_path}")
    print(f"HTML: {html_path}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        print(f"\n{error}\n")
        raise SystemExit(1)
