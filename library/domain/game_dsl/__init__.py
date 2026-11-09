"""The game-language agent (QR Game Studio, decided 2026-11-07; plan in PLAN.md): a code assistant for the app's
game language. `/insert <command>` writes a new block at the cursor, `/edit <command>` rewrites the block under
it. The model reads a short input built by rules (problems.student_input: the Window's settings, the game's names,
where the cursor is, the block around it, the code of the #names) and writes the code in a fenced block.

  language.py  parser + checks the app's compiler does not make + "not hollow"
  compiler.py  the app's own compiler (tool/dsl_check.dart as a standalone program)
  games.py     the games: games/human (the user's 26, 8 held out) and games/new/<family> (written for the data)
  manual.py    manual.md (the app's Docs) + reference.json
  check.py     python -m library.domain.game_dsl.check <files>
  versions.py  seeded versions (names and colours replaced)
  problems.py  /insert cuts and /edit pairs with their short inputs -> problems/problems.jsonl
  commands.py  the commands' styles and check (Gemma's command pass, used in the first price test)
  sonnet_commands.py  the commands written by Claude Sonnet agents -> problems/commands.jsonl
  gemma_writes.py  Gemma writes the blocks (pass 1), the checks at home (pass 2); the kept ones are scored (pass 3)
  vocabulary.py the output vocabulary (from every target and everything a version can contain)
  corpus.py    problems + commands -> corpus.jsonl (the framework's teacher-forced records)

What Gemma reads in front of the input (`prompt`, the same for writing, scoring and its test answers): "card"
(reference_card.md, ~2.5k tokens; chosen by the user 2026-11-07: "fixed is it"), "manual" (manual.md, ~15k; the
first price test) or "none". "card2" (v2, 2026-11-09): the card + CONVENTIONS (how new things are named, what
"this" means), so the naming rule the user asked for (PLAN.md v2 item 5) is Gemma's too.
"""

import json
from pathlib import Path
from typing import List, Optional

from library.shared.metrics.task import ExactAnswerMetric
from ..base import Domain

PACKAGE = Path(__file__).resolve().parent
OUTPUT_TOKENS = PACKAGE / "output_tokens.json"
FENCE = "```"
HEADER = (
    "You are the code assistant inside the editor of a small game language (its manual is below). The user typed a "
    "command at their cursor. You see the game's window settings, the names it has, where the cursor is, the block "
    "under or around the cursor (<cursor> marks the spot), the code of the names the command mentions, and the "
    "command.\n"
    "- /insert: write only the new code that goes at the cursor.\n"
    "- /edit: write the whole block under the cursor again, changed as asked.\n"
    "Reply with the code in one ``` block and nothing else.\n\n{manual}\n---\n\n{text}"
)
CONVENTIONS = (
    "\n\n## Conventions of the assistant\n\n"
    "- A new entity, bar, group or signal is named after what the command calls it, one lower-case word: \"add a red "
    "knight\" -> id(\"knight\"), \"a light red circle\" -> id(\"circle\"), \"create a hero\" -> id(\"hero\"). When "
    "that name is already taken in the game (see NAMES), take the next free one: hero_2, then hero_3, ... (never "
    "_1, never a number when the plain word is free).\n"
    "- In an /edit, \"this\" and \"it\" mean the block under the cursor. Keep its id and everything the command "
    "does not ask to change.\n"
    "- A text, label, title or message the command asks for at a place where an Entity can go is an Entity whose "
    "Type is text(\"...\", size), placed where the player sees it.\n"
    "- A new Level is a whole level: its own Entities (walls, hazards, pickups, enemies), in the style of the game's "
    "other levels, with ids no other level uses.\n"
    "- Actions are separated by spaces or new lines, never by \";\".\n"
    "- Commands are typed quickly: read past typos and missing words."
)


class GameDslDomain(Domain):
    name = "game_dsl"
    stop_token = FENCE
    max_steps = 2000   # runaway guard: the longest target is ~1,460 tokens (a big Entity), +35%
    teacher = "forced"   # the targets are Gemma's own checked blocks (gemma_writes.py), scored in a second pass
    PROMPTS = ("card", "card2", "manual", "none")

    def __init__(self, prompt: str = "card"):
        if prompt not in self.PROMPTS:
            raise ValueError(f"prompt must be one of {self.PROMPTS}")
        self.prompt = prompt
        self._manual: Optional[str] = None
        self._card: Optional[str] = None
        self._example_tokens: Optional[List[str]] = None

    @property
    def manual(self) -> str:
        if self._manual is None:
            self._manual = (PACKAGE / "manual.md").read_text(encoding="utf-8")
        return self._manual

    @property
    def example_tokens(self) -> List[str]:
        """The output vocabulary (vocabulary.py) once it is built; before that (the command pass, which keeps
        no logits) nothing."""
        if self._example_tokens is None:
            self._example_tokens = json.load(open(OUTPUT_TOKENS, encoding="utf-8"))["tokens"] \
                if OUTPUT_TOKENS.exists() else []
        return self._example_tokens

    def reference(self, text: str) -> str:
        if self.prompt == "manual":
            return self.manual
        if self.prompt in ("card", "card2"):
            if self._card is None:
                self._card = (PACKAGE / "reference_card.md").read_text(encoding="utf-8")
            return self._card + (CONVENTIONS if self.prompt == "card2" else "")
        return ""

    def teacher_prompt(self, text: str) -> str:
        return HEADER.format(manual=self.reference(text), text=text)

    def is_stop(self, last_token_decoded: str, generated_text: str) -> bool:
        return generated_text.count(FENCE) >= 2       # the opening fence, the code, the closing fence

    def task_metric(self) -> ExactAnswerMetric:
        return ExactAnswerMetric()

    @property
    def corpus_path(self) -> str:
        return str(PACKAGE / "corpus.jsonl")

    def build_corpus(self, count: Optional[int] = None) -> None:
        from .corpus import build_corpus
        build_corpus(count=count)

    def structural_validity(self, output: str) -> bool:
        output = output.strip()
        return output.startswith(FENCE) and output.endswith(FENCE) and output.count(FENCE) == 2


OUTPUT_TOKENS_V2 = PACKAGE / "output_tokens_v2.json"
CORPUS_V2 = PACKAGE / "corpus_v2.jsonl"


class GameDslV2Domain(GameDslDomain):
    """v2 (2026-11-09, PLAN.md "v2's corpus"): asks from the user's point of view (asks.py), Gemma's "card2" prompt,
    and its own output vocabulary, corpus and batches folder (v1's stay as they are)."""
    name = "game_dsl_v2"

    def __init__(self):
        super().__init__("card2")

    @property
    def example_tokens(self) -> List[str]:
        if self._example_tokens is None:
            self._example_tokens = json.load(open(OUTPUT_TOKENS_V2, encoding="utf-8"))["tokens"] \
                if OUTPUT_TOKENS_V2.exists() else []
        return self._example_tokens

    @property
    def corpus_path(self) -> str:
        return str(CORPUS_V2)

    def build_corpus(self, count: Optional[int] = None) -> None:
        from .v2_corpus import build_corpus
        build_corpus(count=count)
