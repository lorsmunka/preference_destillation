"""The coding agent's output vocabulary: every token a reply may contain, most frequent first. Built
from the format texts, every base project (its code, and every alternative name a version can use,
also as a file name) and Gemma's replies in the pass (every conversations file). Build it after the pass and before saving the
teacher data: rebuilding it later reorders the logit columns of existing batches.

    python -m library.domain.code_agent.vocabulary
"""

import json
import sys
from collections import Counter
from typing import List

from . import OUTPUT_TOKENS, PACKAGE
from .project import load_base, project_list

FORMAT_TEXTS = [
    "<scope>\naccounts.py 12-30\nfees.py 7-11\nmain.py 1-9\n</scope>", "<found/>", "<explanation>The</explanation>",
    "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 "
    "41 42 43 44 45 46 47 48 49 50 51 52 53 54 55",
    # Gemma's tokenizer glues punctuation and spaces onto a closing tag: every form
    "two.</explanation> two </explanation> two?</explanation> two!</explanation> two,</explanation> `two`.</explanation>",
    "two.\n</explanation> 30\n</scope>",
]


def vocabulary_texts() -> List[str]:
    texts = list(FORMAT_TEXTS)
    for project in project_list():
        base = load_base(project)
        texts += list(base["files"].values()) + list(base["files"])
        for name, alternatives in base["renames"].items():
            texts += [f"`{alternative}` {alternative} {alternative}.py {alternative}()" for alternative in [name] + alternatives]
    for path in sorted((PACKAGE / "conversations").glob("*.jsonl")):   # every variant's replies
        for line in open(path, encoding="utf-8"):
            turn = json.loads(line)["turn"]
            texts += [turn[key] for key in ("scope", "output") if turn.get(key)]
    return texts


def build_output_tokens() -> None:
    from transformers import AutoTokenizer
    from library.shared.config import teacher_weights_path

    tokenizer = AutoTokenizer.from_pretrained(teacher_weights_path("google/gemma-4-12B-it"))
    texts = vocabulary_texts()
    counts = Counter()
    for start in range(0, len(texts), 5000):
        for ids in tokenizer(texts[start:start + 5000], add_special_tokens=False)["input_ids"]:
            counts.update(tokenizer.convert_ids_to_tokens(ids))
    vocabulary = tokenizer.get_vocab()
    for token in list(counts):   # each word in both forms: line start ("cost") and mid-line ("▁cost")
        other = token[1:] if token.startswith("▁") else "▁" + token
        if len(other) > 1 and other in vocabulary and other not in counts:
            counts[other] = 1
    tokens = [token for token, _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))]
    replies = sum(1 for path in (PACKAGE / "conversations").glob("*.jsonl") for _ in open(path, encoding="utf-8"))
    with open(OUTPUT_TOKENS, "w", encoding="utf-8") as file:   # derived file, rebuilt each time
        json.dump({"tokens": tokens, "texts": len(texts), "played_items": replies, "projects": project_list()},
                  file, ensure_ascii=False, indent=0)
    print(f"{len(tokens):,} output tokens from {len(texts):,} texts ({replies:,} played items) -> {OUTPUT_TOKENS}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    build_output_tokens()
