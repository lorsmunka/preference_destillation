"""The coding agent (user idea, decided 2026-10-20): a small synthetic Python project (or a site of
plain HTML pages, added 2026-10-21) in the context, the user asks to find something or to have it
explained. The model writes a scope (the files and lines that matter), the harness narrows the
context to it, and the model explains or says it found it. Parts:

  projects/   hand-written base projects (10-15 files of 20-50 lines, project.json: topics and
              renames); project.py makes seeded randomized versions with exact true places
              (html_files.py: the same for HTML pages)
  requests.py the user's requests (find / explain, by meaning or by name) -> items.jsonl
  harness.py  the two steps with their checks, Gemma's pass (the training data) and the records
  checks.py   reading and checking scopes and explanations
  vocabulary.py the output vocabulary, from the projects and Gemma's replies

    python -m library.domain.code_agent.requests     # items.jsonl
    python -m library.domain.code_agent.harness      # GPU: Gemma plays the items
    python -m library.domain.code_agent.vocabulary   # the output vocabulary, with the replies
    then the corpus (CodeAgentDomain().build_corpus()), the teacher data and the students
"""

import json
import re
from pathlib import Path
from typing import List, Optional

from library.shared.metrics.task import ExactAnswerMetric
from ..base import Domain
from .checks import canonical_scope

PACKAGE = Path(__file__).resolve().parent
OUTPUT_TOKENS = PACKAGE / "output_tokens.json"

HEADER = (
    "You are the code assistant inside a tool that shows you a small project (Python files or HTML pages); every "
    "file's lines are numbered.\n"
    "- When you see the whole <project>: pick the lines that matter for the user's <message>. Reply only with a scope "
    "block, one range per line as \"file.py start-end\" (or \"page.html start-end\"), covering every part needed to "
    "answer and as little else as possible. Example:\n<scope>\naccounts.py 12-30\nfees.py 7-11\n</scope>\n"
    "- When you see a narrowed <context> (only the lines picked, with the files' own line numbers): if the user asked "
    "for an explanation, explain how this code works in a few plain sentences for a beginner, naming the functions and "
    "classes (or the page's elements) involved, inside <explanation> and </explanation>. If the user only asked where "
    "something is, reply with exactly <found/>.\n"
    "Write only that one block, nothing before or after it.\n\n{text}"
)


class ScopeMetric(ExactAnswerMetric):
    """Task accuracy on scope records: the student's scope covers exactly the teacher's lines.
    Answer records (explanations, <found/>) have no scope and are skipped."""

    def _extract_answer(self, text: str) -> Optional[str]:
        return canonical_scope(text)


class CodeAgentDomain(Domain):
    """One variant per conversations file (conversations/<name>.jsonl), each with its own name, so
    its own corpus and batches folder: "v1" is the real data (one Gemma setup), others are small
    local trials (e.g. "pilot_nf4": Gemma in 4 bits on the 4070) that never mix with it."""
    stop_token = "</scope>"
    stop_strings = ["</scope>", "</explanation>", "<found/>"]   # generation stops at any reply's end
    max_steps = 400   # runaway guard: explanations up to ~250 tokens in the pilot, +35% and some
    teacher = "forced"   # the teacher data: Gemma reads its own replies back (every record says so too)

    def __init__(self, conversations: str = "v1"):
        self.conversations = conversations
        self.name = "code_agent" if conversations == "v1" else f"code_agent_{conversations}"
        self._example_tokens: Optional[List[str]] = None

    @property
    def conversations_path(self) -> Path:
        return PACKAGE / "conversations" / f"{self.conversations}.jsonl"

    @property
    def corpus_path(self) -> str:
        return str(PACKAGE / f"corpus_{self.conversations}.jsonl")

    @property
    def output_tokens_path(self) -> Path:
        """Each variant has its own output vocabulary, so rebuilding a trial's never reorders the
        logit columns of another variant's teacher data."""
        return OUTPUT_TOKENS if self.conversations == "v1" else PACKAGE / f"output_tokens_{self.conversations}.json"

    @property
    def example_tokens(self) -> List[str]:
        if self._example_tokens is None:
            if not self.output_tokens_path.exists():
                raise FileNotFoundError(f"{self.output_tokens_path.name} missing: "
                                        f"python -m library.domain.code_agent.vocabulary {self.conversations}")
            self._example_tokens = json.load(open(self.output_tokens_path, encoding="utf-8"))["tokens"]
        return self._example_tokens

    def teacher_prompt(self, text: str) -> str:
        return HEADER.format(text=text)

    def is_stop(self, last_token_decoded: str, generated_text: str) -> bool:
        return any(stop in generated_text for stop in self.stop_strings)

    def task_metric(self) -> ScopeMetric:
        return ScopeMetric()

    def structural_validity(self, output: str) -> bool:
        output = output.strip()
        if output.startswith("<scope>"):
            return output.endswith("</scope>") and re.search(r"\.(?:py|html)\s+\d+", output) is not None
        return output == "<found/>" or (output.startswith("<explanation>") and output.endswith("</explanation>"))

    def build_corpus(self, count: Optional[int] = None) -> None:
        from .corpus import build_corpus
        build_corpus(self.corpus_path, self.conversations_path, count=count)
