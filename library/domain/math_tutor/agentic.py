"""The agentic math tutor as a Domain: the tagged conversation (conversation.py) in, one
<solution> or <explanation> block out. One variant per annotation format and teacher mode,
each with its own name (and so its own batches folder):

    teacher "generate"  Gemma writes the block itself (from the prompt's example)
    teacher "forced"    Gemma reads the gold block (annotations.py) in one forward pass and
                        its soft labels are stored at every position: correct targets plus
                        the teacher's distribution

All variants share one output vocabulary (output_tokens.json): the answer and format tokens,
every token of the corpus texts (labels are written in the problem's words), the gold
targets of every format and the explanation threads' targets. Build it after the thread pass,
and see what a corpus would lose to it, per turn type:

    python -m library.domain.math_tutor.agentic
    python -m library.domain.math_tutor.agentic report facts
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from library.shared.metrics.task import ExactAnswerMetric
from ..base import Domain

PACKAGE = Path(__file__).resolve().parent
OUTPUT_TOKENS = PACKAGE / "output_tokens.json"
ANNOTATIONS = ("answer", "working", "named", "code", "facts")
TEACHER_MODES = ("generate", "forced")

HEADER = (
    "You are a maths tutor for primary-school pupils. Below is the tutoring conversation so far. "
    "The last block, <message>, is the pupil's new message.\n"
    "- If it is a new task, or asks what the answer would be if a number were different, or gives "
    "new numbers to work with (a new calculation), answer with a <solution> block{format}\n"
    "- If it asks about your working (why you did a step, what a number means, where it came from, "
    "whether it is right, or to see a step again another way, with smaller numbers or more quickly), "
    "answer with an <explanation> block (if the pupil asks for smaller numbers but gives none, pick small "
    "numbers yourself; if the pupil gives the numbers, it is a new task): at most two short, plain sentences a "
    "child understands, "
    "then </explanation>. Your solutions above have been checked and are right. If the pupil says "
    "one is wrong, say firmly that it is right and show why. If the pupil asks about something a "
    "solution does not use, say why it is not needed.\n"
    "- If it is not about this maths at all, write only <out_of_scope>\n"
    "Write only that one block, nothing before or after it. Never repeat these instructions.\n\n{conversation}"
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
              "result); then the line Answer: <number>; then </solution>. No numbering or bullet points."
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
    stop_strings = ["</solution>", "</explanation>", "<out_of_scope>"]   # generation stops at any reply's end
    max_steps = 150     # runaway guard: longest measured target 96 tokens (facts-first), ~110 expected at scale, +35%

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
        return any(stop in generated_text for stop in self.stop_strings)

    def task_metric(self) -> TutorAnswerMetric:
        return TutorAnswerMetric()

    def structural_validity(self, output: str) -> bool:
        output = output.strip()
        if output.startswith("<solution>"):
            return output.endswith("</solution>") and re.search(r"Answer:\s*\S+", output) is not None
        if output == "<out_of_scope>":   # only the tag: the harness writes the reply
            return True
        return output.startswith("<explanation>") and output.endswith("</explanation>")

    def answer_matches(self, response: str, ground_truth) -> Optional[bool]:
        match = re.search(r"Answer:\s*(\d+)", response)
        return match is not None and int(match.group(1)) == int(ground_truth)


FORMAT_TEXTS = [
    "<solution>\nAnswer: 1234567890\n</solution>", "<explanation>Because</explanation>", "<out_of_scope>",
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


def vocabulary_texts(items: List[Dict], thread_records: List[Dict]) -> List[str]:
    """Every text the output vocabulary is built from: the format texts, the items' problems and
    follow-ups (labels use their words), the gold solutions of every annotation format, and the
    targets of the explanation threads (Gemma's explanations exactly as the explain records hold
    them, "<explanation>Yes, ..." included, and the what-if solutions)."""
    from .annotations import RENDERERS, render
    from .conversation import labelled_turns, thread_turns

    texts = list(FORMAT_TEXTS)
    for item in items:
        texts.append(item["problem"])
        texts += [follow_up["question"] for follow_up in item["follow_ups"]]
        for annotation in RENDERERS:
            turns = labelled_turns(item, annotation)
            if turns:   # a rewording has no named / code labels: render refuses it
                texts += render(item, annotation, turns=turns)
    by_id = {item["id"]: item for item in items}
    thread_targets = {}   # the same explanation is a target in every annotation: once is enough
    for thread_record in thread_records:
        item = by_id.get(thread_record["item_id"])
        if item is None:
            continue
        for annotation in RENDERERS:
            for record in thread_turns(item, annotation, thread_record):
                thread_targets[record["target"]] = True
    return texts + list(thread_targets)


def build_output_tokens() -> None:
    """Every token a tutor block may contain, most frequent first: the tokens of vocabulary_texts
    (plus the line-start form of each word, since a label starts a line). Build it after the
    thread pass and before any data generation: data generation drops every target with a token
    outside it, and rebuilding it later reorders the logit columns of existing batches."""
    from transformers import AutoTokenizer
    from library.shared.config import teacher_weights_path
    from .threads import THREADS_PATH

    tokenizer = AutoTokenizer.from_pretrained(teacher_weights_path("google/gemma-4-12B-it"))
    items = [json.loads(line) for line in open(PACKAGE / "items" / "items.jsonl", encoding="utf-8")]
    thread_records = []
    if THREADS_PATH.exists():
        thread_records = [json.loads(line) for line in open(THREADS_PATH, encoding="utf-8")]
    else:
        print(f"WARNING: no explanation threads yet ({THREADS_PATH}). This vocabulary has NO explanation "
              f"targets, so data generation would drop about half of the explain records ('unexpected_tokens'). "
              f"Run the thread pass first (python -m library.domain.math_tutor.threads), then build again.")
    texts = vocabulary_texts(items, thread_records)
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
    json.dump({"tokens": tokens, "texts": len(texts), "explanation_threads": len(thread_records),
               "built_from": "items.jsonl problems, follow-ups, gold targets" + (", thread targets" if thread_records else "")},
              open(OUTPUT_TOKENS, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(f"{len(tokens):,} output tokens from {len(texts):,} texts ({len(thread_records):,} explanation threads) -> {OUTPUT_TOKENS}")
    print("Check a corpus against it: python -m library.domain.math_tutor.agentic report <annotation>")


def outside_vocabulary_report(records: Iterable[Dict], tokenizer, allowed_tokens: set) -> Dict[str, Dict]:
    """Per turn type (solve / explain / route): how many targets there are and how many have a
    token outside the output vocabulary. Data generation drops those ("unexpected_tokens", in
    forced scoring too), so this is what would be lost, with the commonest such tokens."""
    targets, dropped, outside = Counter(), Counter(), {}
    for record in records:
        turn_type = record.get("turn_type", "solve")
        missing = [token for token in tokenizer.tokenize(record["target"]) if token not in allowed_tokens]
        targets[turn_type] += 1
        dropped[turn_type] += bool(missing)
        outside.setdefault(turn_type, Counter()).update(missing)
    return {turn_type: {"targets": targets[turn_type], "dropped": dropped[turn_type],
                        "commonest_outside_tokens": dict(outside[turn_type].most_common(10))}
            for turn_type in sorted(targets)}


def print_vocabulary_report(annotation: str) -> None:
    """python -m library.domain.math_tutor.agentic report facts: the report above for one
    annotation's corpus file (the records data generation will read)."""
    from transformers import AutoTokenizer
    from library.shared.config import teacher_weights_path
    from library.shared.vocabulary import WHITESPACE_TOKENS

    domain = MathTutorAgenticDomain(annotation)
    if not Path(domain.corpus_path).exists():
        print(f"no corpus at {domain.corpus_path}: build it first (MathTutorAgenticDomain('{annotation}').build_corpus())")
        return
    tokenizer = AutoTokenizer.from_pretrained(teacher_weights_path("google/gemma-4-12B-it"))
    records = [json.loads(line) for line in open(domain.corpus_path, encoding="utf-8")]
    report = outside_vocabulary_report(records, tokenizer, set(domain.example_tokens) | set(WHITESPACE_TOKENS))
    for turn_type, counts in report.items():
        share = 100 * counts["dropped"] / max(1, counts["targets"])
        print(f"{turn_type:8} {counts['targets']:>8,} targets, {counts['dropped']:>7,} ({share:.1f}%) would be dropped"
              f"  {counts['commonest_outside_tokens']}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")   # tokens such as "▁represents" (the Windows console is cp1252)
    if len(sys.argv) >= 2 and sys.argv[1] == "report":
        print_vocabulary_report(sys.argv[2] if len(sys.argv) >= 3 else "facts")
    else:
        build_output_tokens()
