"""The agentic math tutor as a Domain: the tagged conversation (conversation.py) in, one
<solution> or <explanation> block out. One variant per annotation format and teacher mode,
each with its own name (and so its own batches folder):

    teacher "generate"  Gemma writes the block itself (from the prompt's example)
    teacher "forced"    Gemma reads the gold block (annotations.py) in one forward pass and
                        its soft labels are stored at every position: correct targets plus
                        the teacher's distribution

All variants share one output vocabulary (output_tokens.json): the answer and format tokens,
every token of the corpus texts (labels are written in the problem's words) and the gold
targets of every format. Build it with:

    python -m library.domain.math_tutor.agentic
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import List, Optional

from library.shared.metrics.task import ExactAnswerMetric
from ..base import Domain

PACKAGE = Path(__file__).resolve().parent
OUTPUT_TOKENS = PACKAGE / "output_tokens.json"
ANNOTATIONS = ("answer", "working", "named", "code", "facts")
TEACHER_MODES = ("generate", "forced")

HEADER = (
    "You are a maths tutor for primary-school pupils. Below is the tutoring conversation so far. "
    "The last block, <message>, is the pupil's new message.\n"
    "- If it is a new task, answer with a <solution> block{format}\n"
    "- If it asks about your working, answer with an <explanation> block: one to three short "
    "sentences a child understands, then </explanation>.\n"
    "Write only that one block, nothing before or after it.\n\n{conversation}"
)
REMAINDERS = " A division that does not come out even is written like: 23 / 5 = 4 remainder 3."
FORMATS = {
    "answer": ": only the line Answer: <number>, then </solution>. Example:\n<solution>\nAnswer: 5\n</solution>",
    "working": (": one line per calculation that is needed, written as calculation = result, using only "
                "+ - * /; then the line Answer: <number>; then </solution>." + REMAINDERS + " Example:\n"
                "<solution>\n4 * 2 = 8\n13 - 8 = 5\nAnswer: 5\n</solution>"),
    "named": (": one line per calculation that is needed, written as what it is = calculation = result, "
              "naming the quantity with words from the problem and using only + - * /; then the line "
              "Answer: <number>; then </solution>." + REMAINDERS + " Example:\n"
              "<solution>\ncost of 4 more tetras = 4 * 2 = 8\nmoney left = 13 - 8 = 5\nAnswer: 5\n</solution>"),
    "facts": (": first one line for each number you use, written as what it is = number (words from the problem; "
              "a number from an earlier answer counts too; numbers without units); then one line per calculation that is needed, written as "
              "what it is = calculation = result, using only + - * / (every calculation line ends with = and its "
              "result); then the line Answer: <number>; then </solution>."
              + REMAINDERS + " Example:\n<solution>\ntetras bought later = 4\nprice of a tetra = 2\nmoney left before = 13\n"
              "cost of 4 more tetras = 4 * 2 = 8\nmoney left = 13 - 8 = 5\nAnswer: 5\n</solution>"),
    "code": (": one line of Python per step, giving a value to a name taken from the problem, using only "
             "+ - * // % and earlier names; then the line Answer: <name>; then </solution>. The program is "
             "run for you. Example:\n<solution>\ncost_of_more_tetras = 4 * 2\nmoney_left = 13 - cost_of_more_tetras\n"
             "Answer: money_left\n</solution>"),
}


class TutorAnswerMetric(ExactAnswerMetric):
    """Answer: N inside a <solution> block (no closing semicolon in this format). Turns
    without an answer (explanations) are skipped, as ExactAnswerMetric does."""

    def _extract_answer(self, text: str) -> Optional[str]:
        match = re.search(r"Answer:\s*([^\n<]+)", text)
        return match.group(1).strip().lower() if match else None


class MathTutorAgenticDomain(Domain):
    stop_token = "</solution>"
    stop_strings = ["</solution>", "</explanation>"]   # batched generation stops at either block's end
    max_steps = 120     # explanations run longer than solutions (gold solutions max 62 tokens)
    max_input_tokens = 320

    def __init__(self, annotation: str = "named", teacher: str = "generate"):
        if annotation not in ANNOTATIONS or teacher not in TEACHER_MODES:
            raise ValueError(f"annotation in {ANNOTATIONS}, teacher in {TEACHER_MODES}")
        self.annotation = annotation
        self.teacher = teacher
        self.name = f"math_tutor_{annotation}" + ("_forced" if teacher == "forced" else "")
        self._example_tokens: Optional[List[str]] = None

    @property
    def example_tokens(self) -> List[str]:
        if self._example_tokens is None:
            if not OUTPUT_TOKENS.exists():
                raise FileNotFoundError("output_tokens.json missing: python -m library.domain.math_tutor.agentic")
            self._example_tokens = json.load(open(OUTPUT_TOKENS, encoding="utf-8"))["tokens"]
        return self._example_tokens

    @property
    def corpus_path(self) -> str:
        return str(PACKAGE / f"corpus_{self.annotation}.jsonl")

    def build_corpus(self, count: Optional[int] = None) -> None:
        from .corpus import build_agentic_corpus
        build_agentic_corpus(self.corpus_path, self.annotation, count=count)

    def teacher_prompt(self, text: str) -> str:
        return HEADER.format(format=FORMATS[self.annotation], conversation=text)

    def is_stop(self, last_token_decoded: str, generated_text: str) -> bool:
        return "</solution>" in generated_text or "</explanation>" in generated_text

    def task_metric(self) -> TutorAnswerMetric:
        return TutorAnswerMetric()

    def structural_validity(self, output: str) -> bool:
        output = output.strip()
        if output.startswith("<solution>"):
            return output.endswith("</solution>") and re.search(r"Answer:\s*\S+", output) is not None
        return output.startswith("<explanation>") and output.endswith("</explanation>")

    def answer_matches(self, response: str, ground_truth) -> Optional[bool]:
        match = re.search(r"Answer:\s*(\d+)", response)
        return match is not None and int(match.group(1)) == int(ground_truth)


FORMAT_TEXTS = [
    "<solution>\nAnswer: 1234567890\n</solution>", "<explanation>Because</explanation>",
    "4 * 2 = 8\n13 - 8 = 5\n52 / 8 = 6\n7 + 1 = 8\n(16 + 3 - 1) // 3 = 6\n23 % 5 = 3",
    "35-23=12\n4*6=24\n8/2=4\n3+4=7",
    "16 / 3 = 5 remainder 1, rounded up = 6\nremainder of 23 / 5 = 3\n23 / 5 = 4 remainder 3",
    "cost_of_more = 4 * price\nAnswer: cost_of_more",
    # Gemma's tokenizer glues punctuation and spaces onto a closing tag (".</", "▁</"): every form
    "two.</explanation> two </explanation> two?</explanation> two!</explanation> two,</explanation>",
    "5.\n</solution> 5 </solution> two.</solution>\n</explanation>",
    # words explanations of arithmetic need even when no problem text uses them
    "The Multiplication multiplication Subtraction subtraction Addition addition Division division "
    "calculate calculated calculation initial initially difference product sum quotient remainder "
    "Because because each times equal groups share shared altogether total fewer less twice half double "
    "I subtracted added multiplied divided found took first then next so means that's it's we you",
]


def build_output_tokens() -> None:
    """Every token a tutor block may contain, most frequent first: the format texts, the gold
    targets of every annotation format, and every token of the corpus texts (plus the
    line-start form of each word, since a label starts a line)."""
    from transformers import AutoTokenizer
    from library.shared.config import teacher_weights_path
    from .annotations import RENDERERS, render
    from .conversation import usable

    tokenizer = AutoTokenizer.from_pretrained(teacher_weights_path("google/gemma-4-12B-it"))
    items = [json.loads(line) for line in open(PACKAGE / "items" / "items.jsonl", encoding="utf-8")]
    texts = list(FORMAT_TEXTS)
    for item in items:
        texts.append(item["problem"])
        texts += [follow_up["question"] for follow_up in item["follow_ups"]]
        for annotation in RENDERERS:
            if usable(item, annotation):
                texts += render(item, annotation)
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
    json.dump({"tokens": tokens, "texts": len(texts), "built_from": "items.jsonl problems, follow-ups, gold targets"},
              open(OUTPUT_TOKENS, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(f"{len(tokens):,} output tokens from {len(texts):,} texts -> {OUTPUT_TOKENS}")


if __name__ == "__main__":
    build_output_tokens()
