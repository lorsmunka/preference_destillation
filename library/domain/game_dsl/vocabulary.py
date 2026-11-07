"""The output vocabulary: every Gemma token an answer may contain, most frequent first. Built from the targets of
every problem (with their fences), every game and version, and everything a version can put into a game (the name
pool with its suffixes, the colour palette, the sound and music names). Build it after problems.py and before the
teacher data: rebuilding it later reorders the logit columns of existing batches.

    python -m library.domain.game_dsl.vocabulary
"""

import json
from collections import Counter

from . import FENCE, OUTPUT_TOKENS
from .gemma_writes import BLOCKS
from .manual import sound_and_music_names
from .problems import OUTPUT
from .versions import PALETTE, WORDS


def vocabulary_texts():
    texts = [f"{FENCE}\n", f"\n{FENCE}", " ".join(str(number) for number in range(0, 2001)),
             " ".join(f"-{number}" for number in range(0, 1001)), "0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.5 0.6 0.7 0.75"]
    for word in WORDS:
        variants = [word] + [word + str(digit) for digit in range(10)] + [word + letter for letter in "ABCDEFGH"] \
            + [word + other.capitalize() for other in WORDS[:40]]
        texts += [f'"{variant}"' for variant in variants]
    texts += [f'"{colour}" "80{colour}" "#{colour}"' for colour in PALETTE]
    texts += [f'sound("{name}") music("{name}")' for name in sound_and_music_names()]
    for line in open(OUTPUT / "versions.jsonl", encoding="utf-8"):
        texts.append(json.loads(line)["source"])
    for line in open(OUTPUT / "problems.jsonl", encoding="utf-8"):
        problem = json.loads(line)
        texts.append(f"{FENCE}\n{problem['target']}\n{FENCE}")
    if BLOCKS.exists():     # Gemma's kept blocks (the targets since 2026-11-07): 18% of them had a token outside
        for line in open(BLOCKS, encoding="utf-8"):      # the vocabulary built from the real blocks alone
            texts.append(f"{FENCE}\n{json.loads(line)['block']}\n{FENCE}")
    return texts


def main():
    from transformers import AutoTokenizer
    from library.shared.config import teacher_weights_path
    tokenizer = AutoTokenizer.from_pretrained(teacher_weights_path("google/gemma-4-12B-it"))
    counts = Counter()
    for text in vocabulary_texts():
        counts.update(tokenizer.tokenize(text))
    tokens = [token for token, _ in counts.most_common()]
    OUTPUT_TOKENS.write_text(json.dumps({"tokens": tokens}, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(tokens):,} tokens -> {OUTPUT_TOKENS}")


if __name__ == "__main__":
    main()
