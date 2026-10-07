"""Question templates: the user's follow-up messages, written once by template writers and filled
in per conversation. Domain-independent: a domain turns its conversation into foci (the things a
message can be about, each with slot values and features); this module picks a template that
applies and fills it.

A template is one JSON line in a bank file:

    {"kind": "why", "route": "explanation", "focus": "step", "requires": ["divide"],
     "depth": "first", "text": "Why did you divide {a} by {b}?"}

  kind      what the user does (the domain lists its kinds)
  route     the reply the agent should give: "solution", "explanation", or "out_of_scope"
            (the harness answers that one itself)
  focus     what the message is about: a focus type the domain produces; "none" = nothing
  requires  features the focus must have, all of them: this is what makes a template niche
  depth     "first" opens a thread, "follow" only comes after an explanation, "any" either
  text      the message; {name} is a slot, filled from the focus

A focus offers slots (name -> value) and features. A template applies to a focus when the types
match, it requires only features the focus has, and every slot it uses is offered.
"""

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

SLOT = re.compile(r"\{(\w+)\}")
DEPTHS = ("first", "follow", "any")
NONE = "none"


@dataclass
class Focus:
    type: str                                   # e.g. "step", "fact", "answer", "none"
    slots: Dict[str, object] = field(default_factory=dict)
    features: Set[str] = field(default_factory=set)
    key: str = ""                               # identity inside one conversation
    links: List[str] = field(default_factory=list)   # keys of related foci (a cascade can move there)
    data: Dict[str, object] = field(default_factory=dict)   # domain extras (a what-if's gold answer...)


@dataclass(frozen=True)
class Template:
    kind: str
    route: str
    focus: str
    text: str
    requires: tuple = ()
    depth: str = "any"
    source: str = ""

    @property
    def slots(self) -> Set[str]:
        return set(SLOT.findall(self.text))

    def applies(self, focus: Focus, depth: int) -> bool:
        if self.focus != focus.type:
            return False
        if (self.depth == "first" and depth > 0) or (self.depth == "follow" and depth == 0):
            return False
        return set(self.requires) <= focus.features and self.slots <= set(focus.slots)

    def fill(self, focus: Focus) -> str:
        return SLOT.sub(lambda match: str(focus.slots[match.group(1)]), self.text)


@dataclass
class Message:
    text: str
    kind: str
    route: str
    focus: Focus
    template: Template


def template_problem(record: Dict, kinds: Dict[str, str], focus_slots: Dict[str, Set[str]],
                     features: Set[str]) -> Optional[str]:
    """Why a bank line is not a usable template, or None. `kinds` maps kind -> route,
    `focus_slots` focus type -> the slots it offers, `features` every feature name."""
    for name in ("kind", "focus", "text"):
        if not isinstance(record.get(name), str) or not record[name].strip():
            return f"missing {name}"
    if record["kind"] not in kinds:
        return f"unknown kind {record['kind']!r}"
    if record.get("route", kinds[record["kind"]]) != kinds[record["kind"]]:
        return f"kind {record['kind']} has route {kinds[record['kind']]!r}"
    if record["focus"] not in focus_slots:
        return f"unknown focus {record['focus']!r}"
    if record.get("depth", "any") not in DEPTHS:
        return f"depth must be one of {DEPTHS}"
    unknown = set(record.get("requires", [])) - features
    if unknown:
        return f"unknown features {sorted(unknown)}"
    missing = set(SLOT.findall(record["text"])) - focus_slots[record["focus"]]
    if missing:
        return f"focus {record['focus']} has no slots {sorted(missing)}"
    if re.search(r"[{}]", SLOT.sub("", record["text"])):
        return "stray brace"
    return None


def normalised_text(text: str) -> str:
    return " ".join(text.lower().split())


def load_bank(paths: Iterable[Path], kinds: Dict[str, str], excluded: Iterable[str] = ()) -> List[Template]:
    """Every template in the bank files (bad lines are skipped: check them with template_problem),
    except the excluded texts (a list kept beside the bank, with the reason for each)."""
    templates, seen = [], set()
    excluded = {normalised_text(text) for text in excluded}
    for path in paths:
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            record = json.loads(line)
            key = (record.get("kind"), record.get("focus"), normalised_text(str(record.get("text", ""))))
            if key in seen or record.get("kind") not in kinds or key[2] in excluded:
                continue
            seen.add(key)
            templates.append(Template(kind=record["kind"], route=kinds[record["kind"]], focus=record["focus"],
                                      text=record["text"].strip(), requires=tuple(record.get("requires", [])),
                                      depth=record.get("depth", "any"), source=Path(path).name))
    return templates


class QuestionGenerator:
    """Draws one message: first a kind (by weight, among kinds that have an applicable template),
    then a (template, focus) pair of that kind. Niche templates weigh more (2 per required
    feature), so they are used whenever their condition holds."""

    def __init__(self, templates: Sequence[Template], kind_weights: Dict[str, float]):
        self.templates = list(templates)
        self.kind_weights = kind_weights

    def candidates(self, foci: Sequence[Focus], depth: int, allowed_kinds: Optional[Set[str]] = None):
        pairs: Dict[str, List] = {}
        for template in self.templates:
            if allowed_kinds is not None and template.kind not in allowed_kinds:
                continue
            for focus in foci:
                if template.applies(focus, depth):
                    pairs.setdefault(template.kind, []).append((template, focus))
        return pairs

    def draw(self, foci: Sequence[Focus], depth: int, random_generator: random.Random,
             allowed_kinds: Optional[Set[str]] = None, kind_weights: Optional[Dict[str, float]] = None) -> Optional[Message]:
        weights = kind_weights or self.kind_weights
        pairs = self.candidates(foci, depth, allowed_kinds)
        kinds = [kind for kind in pairs if weights.get(kind, 0) > 0]
        if not kinds:
            return None
        kind = random_generator.choices(kinds, weights=[weights[kind] for kind in kinds])[0]
        options = pairs[kind]
        template, focus = random_generator.choices(options, weights=[2 ** len(t.requires) for t, _ in options])[0]
        return Message(template.fill(focus), kind, template.route, focus, template)


def related(foci: Sequence[Focus], previous: Optional[Focus]) -> List[Focus]:
    """For a follow-up message: the previous message's focus, the foci linked to it, and none."""
    if previous is None:
        return list(foci)
    keys = {previous.key, *previous.links}
    return [focus for focus in foci if focus.key in keys or focus.type == NONE]
