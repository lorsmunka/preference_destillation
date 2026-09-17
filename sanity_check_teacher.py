"""Run a handful of prompts from every domain through a teacher before generating data
with it — see the answers, the thinking traces, and whether the reduced vocabulary would
accept the outputs.

    python sanity_check_teacher.py                            # Gemma 4 12B-it, 4-bit, 4 examples per domain
    python sanity_check_teacher.py google/gemma-3-4b-it       # a different teacher
    python sanity_check_teacher.py raw reddit_comment_sentiment math_word_problem
                                                              # only these domains; "raw" = no chat template

Prints, per domain: the prompt tail, the response, the thinking trace (when enabled), and
any tokens the domain's reduced vocabulary would reject.
"""

import gc
import json
import sys

import torch
from itertools import islice

from library.data_gen.model_handler import ModelHandler
from library.domain import registered_domains, get_domain

ARGUMENTS = sys.argv[1:]
TEACHER = next((a for a in ARGUMENTS if "/" in a), "google/gemma-4-12B-it")
RAW_PROMPT = "raw" in ARGUMENTS                                   # skip the chat template (Gemma 3 style)
ONLY_DOMAINS = [a for a in ARGUMENTS if "/" not in a and a != "raw"]
EXAMPLES_PER_DOMAIN = 4
FOUR_BIT = "12B" in TEACHER or "27b" in TEACHER.lower() or "31B" in TEACHER   # bf16 teachers that would not fit a 12 GB card
GEMMA_4 = "gemma-4" in TEACHER.lower()


def first_texts(domain, count):
    with open(domain.corpus_path, "r", encoding="utf-8") as file:
        return [json.loads(line)["text"] for line in islice(file, count)]


def run(domain_name: str, thinking: bool):
    domain = get_domain(domain_name)
    print(f"\n{'=' * 78}\n{domain_name}   thinking={thinking}\n{'=' * 78}")
    handler = ModelHandler(TEACHER, domain, load_in_4bit=FOUR_BIT, chat_template=GEMMA_4, thinking=thinking)
    for text in first_texts(domain, EXAMPLES_PER_DOMAIN):
        example, skip_reason = handler.generate_training_example(text)
        print(f"\n--- input: {text[:140]!r}")
        if example is None:
            print(f"    SKIPPED: {skip_reason}   raw text: {handler.last_generated_text[:300]!r}")
            continue
        if example.get("thinking"):
            print(f"    thinking ({len(example['thinking'])} chars): {example['thinking'][:300]!r}")
        print(f"    response ({len(example['steps'])} steps): {example['model_response'][:300]!r}")
    del handler
    gc.collect()
    torch.cuda.empty_cache()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    for domain_name in ONLY_DOMAINS or registered_domains():
        run(domain_name, thinking=False)
        if GEMMA_4 and not ONLY_DOMAINS and domain_name in ("long_context_recall",):
            run(domain_name, thinking=True)   # maths thinking dropped: 5-17 min per example and off-format


if __name__ == "__main__":
    main()
