"""Price test of the game-language agent's Gemma passes (library/domain/game_dsl/PLAN.md, "the price test"), on a
rented card in bf16 (never the local 4-bit copy), 2026-11-07:

  commands  Gemma words the commands of 160 train problems (commands.prompt_for / check_command, 32 per call):
            seconds per command and the share that passes the check
  scoring   the real block of 96 of them, teacher-forced, with three things in front of the input: the full manual
            (~14k tokens, the domain's teacher_prompt), the short reference card (~2k) and nothing. Per variant:
            seconds per record, Gemma's top guess = the real token, mean CE on the real token, and how far the card's
            and nothing's distributions over the output vocabulary are from the manual's (mean KL, manual || other)
  answers   Gemma writes the block itself (full manual, greedy) on 32 test problems: seconds per answer. The answers
            go through the app's compiler at home (report).

    python experiments/game_price_test.py sample    # at home: draws the problems -> runs/game_price_test/sample.jsonl
    python experiments/game_price_test.py run       # on the card (~20 min)
    python experiments/game_price_test.py report    # at home: the answers through the checks, the summary
"""

import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from library.domain.game_dsl import FENCE, GameDslDomain, PACKAGE  # noqa: E402
from library.domain.game_dsl.commands import check_command, prompt_for, style_of  # noqa: E402
from library.domain.game_dsl.problems import student_input  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "runs" / "game_price_test"
PROBLEMS = PACKAGE / "problems" / "problems.jsonl"
COMMAND_COUNT, SCORE_COUNT, ANSWER_COUNT = 160, 96, 32
VARIANTS = {"manual": 4, "card": 16, "nothing": 32}       # what goes in front of the input -> records per call


def read_jsonl(path):
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample():
    path = OUTPUT / "sample.jsonl"
    if path.exists():
        print(f"{path} exists")
        return
    problems = read_jsonl(PROBLEMS)
    generator = random.Random(0)
    train = generator.sample([p for p in problems if p["split"] == "train" and style_of(p) != "empty"], COMMAND_COUNT)
    test = generator.sample([p for p in problems if p["split"] == "test" and style_of(p) != "empty"], ANSWER_COUNT)
    write_jsonl(path, [{**p, "role": "train"} for p in train] + [{**p, "role": "test"} for p in test])
    print(f"{len(train)} train + {len(test)} test problems -> {path}")


def run():
    import numpy as np
    import torch
    from library.data_gen.model_handler import ModelHandler
    problems = read_jsonl(OUTPUT / "sample.jsonl")
    train = [p for p in problems if p["role"] == "train"]
    test = [p for p in problems if p["role"] == "test"]
    domain = GameDslDomain()
    manual = domain.manual
    card = (PACKAGE / "reference_card.md").read_text(encoding="utf-8")
    handler = ModelHandler("google/gemma-4-12B-it", domain, load_in_4bit=False, chat_template=True)
    summary = {"card": torch.cuda.get_device_name(0)}

    def prompt_tokens(prompts):
        return sum(len(handler.tokenizer(prompt, add_special_tokens=False)["input_ids"]) for prompt in prompts) / len(prompts)

    # commands, as commands.main does it (no manual in the prompt, its default), plus test commands for the answers
    commands = {}
    for name, chunk_problems in (("train", train), ("test", test)):
        start = time.time()
        for first in range(0, len(chunk_problems), 32):
            chunk = chunk_problems[first:first + 32]
            styles = [style_of(problem) for problem in chunk]
            prompts = [prompt_for(problem, style, None) for problem, style in zip(chunk, styles)]
            texts = handler.generate_texts(prompts, max_new_tokens=120)
            for problem, style, text in zip(chunk, styles, texts):
                commands[problem["id"]] = (style, check_command(problem, style, text), text)
        if name == "train":
            seconds = time.time() - start
            summary["commands"] = {"seconds_per_command": seconds / len(chunk_problems),
                                   "kept": sum(commands[p["id"]][1] is not None for p in chunk_problems) / len(chunk_problems),
                                   "prompt_tokens": prompt_tokens(prompts)}
            print("commands", summary["commands"], flush=True)
    write_jsonl(OUTPUT / "commands.jsonl", [{"id": key, "style": value[0], "command": value[1], "reply": value[2]}
                                            for key, value in commands.items()])

    def record(problem):
        style, command, _ = commands[problem["id"]]
        shown = problem if style in ("terse", "detailed") else {**problem, "named_blocks": {}}
        return student_input(shown, command), f"{FENCE}\n{problem['target']}\n{FENCE}"

    scored = [p for p in train if commands[p["id"]][1] is not None][:SCORE_COUNT]
    texts, targets = zip(*[record(problem) for problem in scored])
    token_index = {token: index for index, token in enumerate(handler.vocabulary["token_list"])}
    log_probabilities = {}
    summary["scoring"] = {}
    for variant, batch in VARIANTS.items():
        domain._manual = {"manual": manual, "card": card, "nothing": ""}[variant]
        handler.score_training_examples(list(texts[:2]), list(targets[:2]))      # warm-up, not timed
        torch.cuda.synchronize()
        start = time.time()
        rows = []
        for first in range(0, len(texts), batch):
            rows += handler.score_training_examples(list(texts[first:first + batch]), list(targets[first:first + batch]))
        torch.cuda.synchronize()
        seconds = time.time() - start
        top, cross_entropy, steps_seen, kept = 0, 0.0, 0, 0
        variant_log_probabilities = []
        for example, reason in rows:
            if example is None:
                variant_log_probabilities.append(None)
                continue
            kept += 1
            logits = np.array([step["logits"] for step in example["steps"]], dtype=np.float32)
            logits -= logits.max(axis=1, keepdims=True)
            log_softmax = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
            real = [token_index[step["token"]] for step in example["steps"]]
            top += sum(int(step["predicted_token_index"] == index) for step, index in zip(example["steps"], real))
            cross_entropy -= float(sum(log_softmax[position, index] for position, index in enumerate(real)))
            steps_seen += len(real)
            variant_log_probabilities.append(log_softmax.astype(np.float16))
        log_probabilities[variant] = variant_log_probabilities
        summary["scoring"][variant] = {"seconds_per_record": seconds / len(texts), "records": len(texts), "kept": kept,
                                       "top_guess_is_real": top / steps_seen, "mean_ce_on_real": cross_entropy / steps_seen,
                                       "target_tokens": steps_seen,
                                       "prompt_tokens": prompt_tokens([handler.build_prompt(text) for text in texts])}
        print(variant, summary["scoring"][variant], flush=True)
    domain._manual = manual
    for variant in ("card", "nothing"):
        divergences = []
        for reference, other in zip(log_probabilities["manual"], log_probabilities[variant]):
            if reference is None or other is None or reference.shape != other.shape:
                continue
            reference32, other32 = reference.astype(np.float32), other.astype(np.float32)
            divergences += list((np.exp(reference32) * (reference32 - other32)).sum(axis=1))
        summary["scoring"][variant]["mean_kl_from_manual"] = float(np.mean(divergences))

    # Gemma's own answers, full manual, greedy
    answerable = [p for p in test if commands[p["id"]][1] is not None]
    prompts = [domain.teacher_prompt(record(problem)[0]) for problem in answerable]
    start = time.time()
    answers = []
    for first in range(0, len(prompts), 8):
        answers += handler.generate_texts(prompts[first:first + 8], max_new_tokens=600)
    seconds = time.time() - start
    summary["answers"] = {"seconds_per_answer": seconds / max(len(answers), 1), "answers": len(answers),
                          "prompt_tokens": prompt_tokens(prompts)}
    print("answers", summary["answers"], flush=True)
    write_jsonl(OUTPUT / "answers.jsonl", [{"id": problem["id"], "target": problem["target"], "answer": answer}
                                           for problem, answer in zip(answerable, answers)])
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


def report():
    from library.domain.game_dsl import language
    summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
    print(json.dumps(summary, indent=1))
    answers = read_jsonl(OUTPUT / "answers.jsonl")
    exact = sum(answer["answer"].strip().strip("`").strip() == answer["target"].strip() for answer in answers)
    print(f"answers equal to the real block (exact text): {exact}/{len(answers)}")
    print("(compile / check through library.domain.game_dsl.check by hand: see runs/game_price_test/answers.jsonl)")
    del language


if __name__ == "__main__":
    {"sample": sample, "run": run, "report": report}[sys.argv[1] if len(sys.argv) > 1 else "report"]()
