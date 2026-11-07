"""The game language of QR Game Studio (C:/Users/Ors/qrcode_dev/qr_flutter) in Python: the same tokenizer and
parser as the app's Dart compiler (lib/compiler/tokenizer, lib/compiler/parser), with the character span of
every block, plus the checks the Dart compiler does not make. The compiler rejects an unknown keyword but skips
a known keyword in the wrong block (lib/compiler/keyword_check.dart says so), and it accepts a missing argument,
a bad colour or an unknown id. So a program is only "clean" when it compiles AND passes `problems` here; a game
written for the training data must also pass `hollow` (no empty blocks, no template never used, ...)."""

import re
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Set, Tuple


@dataclass
class Token:
    kind: str      # number, string, identifier, punctuation, eof
    value: str
    start: int
    end: int


@dataclass
class Node:
    name: str
    args: List[object]
    children: List["Node"] = field(default_factory=list)
    start: int = 0
    end: int = 0
    arg_spans: List[Tuple[int, int]] = field(default_factory=list)   # source span of each argument

    @property
    def key(self) -> str:
        return self.name.lower()

    def child(self, key: str) -> Optional["Node"]:
        return next((child for child in self.children if child.key == key), None)

    def first_string(self) -> Optional[str]:
        return self.args[0] if self.args and isinstance(self.args[0], str) else None


def tokenize(source: str) -> List[Token]:
    tokens = []
    position = 0
    while position < len(source):
        char = source[position]
        if char.isspace():
            position += 1
        elif char.isdigit() or (char == "-" and position + 1 < len(source) and source[position + 1].isdigit()):
            end = position + 1
            while end < len(source) and (source[end].isdigit() or source[end] == "."):
                end += 1
            tokens.append(Token("number", source[position:end], position, end))
            position = end
        elif char == '"':
            end = source.find('"', position + 1)
            if end < 0:
                raise ValueError(f"unterminated string at {position}")
            tokens.append(Token("string", source[position + 1:end], position, end + 1))
            position = end + 1
        elif char.isalpha() or char == "_":
            end = position + 1
            while end < len(source) and (source[end].isalnum() or source[end] == "_"):
                end += 1
            tokens.append(Token("identifier", source[position:end], position, end))
            position = end
        elif char in "(),{}":
            tokens.append(Token("punctuation", char, position, position + 1))
            position += 1
        elif char == "#":
            while position < len(source) and source[position] != "\n":
                position += 1
        else:
            raise ValueError(f"unexpected character {char!r} at {position}")
    tokens.append(Token("eof", "", len(source), len(source)))
    return tokens


class Parser:
    def __init__(self, source: str):
        self.tokens = tokenize(source)
        self.index = 0

    @property
    def current(self) -> Token:
        return self.tokens[self.index]

    def at(self, value: str) -> bool:
        return self.current.kind == "punctuation" and self.current.value == value

    def expect(self, value: str) -> Token:
        if not self.at(value):
            raise ValueError(f"expected {value!r}, got {self.current.value!r} at {self.current.start}")
        token = self.current
        self.index += 1
        return token

    def parse_program(self) -> Node:
        root = self.parse_call()
        if self.current.kind != "eof":
            raise ValueError(f"code after the root block at {self.current.start}")
        return root

    def parse_many(self) -> List[Node]:
        nodes = []
        while self.current.kind != "eof":
            nodes.append(self.parse_call())
        return nodes

    def parse_call(self) -> Node:
        token = self.current
        if token.kind != "identifier":
            raise ValueError(f"expected a keyword, got {token.value!r} at {token.start}")
        self.index += 1
        node = Node(token.value, [], [], token.start, token.end)
        if self.at("("):
            self.index += 1
            while not self.at(")"):
                if self.current.kind == "eof":
                    raise ValueError("unclosed (")
                if self.at(","):
                    self.index += 1
                    continue
                if self.current.kind == "identifier" and self.tokens[self.index + 1].value == "(":
                    if self.current.value.lower() != "random":
                        raise ValueError(f"node {self.current.value!r} inside a parameter list")
                    random_start = self.current.start
                    self.index += 2
                    inner = []
                    while not self.at(")"):
                        if self.current.kind == "eof":
                            raise ValueError("unclosed random(")
                        if not self.at(","):
                            inner.append(self.current.value)
                        self.index += 1
                    node.args.append(("random", inner))
                    node.arg_spans.append((random_start, self.current.end))
                    self.index += 1
                    continue
                if self.current.kind not in ("string", "number", "identifier"):
                    raise ValueError(f"unexpected {self.current.value!r} at {self.current.start}")
                value = self.current.value
                node.args.append(float(value) if self.current.kind == "number" else value)
                node.arg_spans.append((self.current.start, self.current.end))
                self.index += 1
            node.end = self.expect(")").end
        if self.at("{"):
            self.index += 1
            while not self.at("}"):
                if self.current.kind == "eof":
                    raise ValueError("unclosed {")
                node.children.append(self.parse_call())
            node.end = self.expect("}").end
        return node


def parse_program(source: str) -> Node:
    return Parser(source).parse_program()


def parse_fragment(source: str) -> List[Node]:
    return Parser(source).parse_many()


def walk(node: Node, parent: Optional[Node] = None) -> Iterator[Tuple[Node, Optional[Node]]]:
    yield node, parent
    for child in node.children:
        yield from walk(child, node)


# ---------- where each keyword may sit and how many arguments it takes ----------

# what the compiler reads at the level of a Window (single level) or of a Level block
# (entity_builder.dart: build; game_def_builder.dart: _synthesizeWaveManager)
LEVEL_CHILDREN = ("entity", "dpad", "joystick", "paddle", "goal", "hostilepatrol", "wallbox", "wave")


@dataclass
class Reference:
    placements: Set[Tuple[str, str]] = field(default_factory=set)       # (parent, child), lower case
    arities: Dict[str, Set[int]] = field(default_factory=dict)

    def add_tree(self, node: Node, parent_key: Optional[str]):
        for child, parent in walk(node):
            parent_name = parent.key if parent is not None else parent_key
            if parent_name is not None:
                self.placements.add((parent_name, child.key))
            self.arities.setdefault(child.key, set()).add(len(child.args))

    def to_json(self) -> dict:
        return {"placements": sorted(self.placements),
                "arities": {key: sorted(value) for key, value in sorted(self.arities.items())}}

    @classmethod
    def from_json(cls, data: dict) -> "Reference":
        return cls({tuple(pair) for pair in data["placements"]},
                   {key: set(value) for key, value in data["arities"].items()})


# argument counts the docs describe in words (optional arguments) but no example shows
DOCUMENTED_ARITIES = {"wave": {1, 2}, "star": {2, 3}, "ring": {1, 2}, "cross": {2, 3}, "bounceback": {0, 1},
                      "chase": {1, 2}, "lives": {2, 3}, "hostile": {2, 3}, "key": {1, 2, 3}, "door": {1, 2},
                      "wallbox": {4, 5, 6}, "spawnhere": {1, 3}, "flash": {2, 3}, "burst": {2, 3},
                      "gridstep": {2, 3}, "setscale": {1, 2}, "setopacity": {1, 2}, "setvisible": {1, 2},
                      "setenable": {1, 2}, "applyimpulse": {2, 3}, "setvelocity": {2, 3}, "deleteentity": {0, 1},
                      "winat": {1, 2, 3, 4}, "updatebar": {1, 2}, "dpad": {1, 2}, "clearfullrows": {6, 7},
                      "clearfullcolumns": {6, 7}, "clearfullboxes": {7, 8},
                      # action_factories.dart: slide(dx, dy, mergeVar?, signal?), slideAll(..., mergeVar?, signal?)
                      "slide": {2, 3, 4}, "slideall": {4, 5, 6}}


def build_reference(programs: List[str], fragments: List[str]) -> Reference:
    reference = Reference()
    for source in programs:
        reference.add_tree(parse_program(source), None)
    for source in fragments:
        for node in parse_fragment(source):
            reference.add_tree(node, None)
    for parent in ("window", "level"):
        for child in LEVEL_CHILDREN:
            reference.placements.add((parent, child))
    for key, counts in DOCUMENTED_ARITIES.items():
        reference.arities.setdefault(key, set()).update(counts)
    return reference


COLOR = re.compile(r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")
COLOR_KEYWORDS = {"color", "background", "setcolor"}
# keywords whose first string argument names an entity (or a template) that must exist
REFERENCE_KEYWORDS = {"oncollision", "chase", "retarget", "spawn", "spawnhere", "spawnatempty", "camerafollow",
                      "setdirection", "joystick", "dpad", "hostile"}
# keywords that define an entity id with their first argument
DEFINING_KEYWORDS = {"id", "paddle", "goal", "hostilepatrol"}
SUGAR_IDS = {"dpad": ["padUp", "padDown", "padLeft", "padRight", "floor"],
             "wallbox": ["wallL", "wallR", "wallT", "wallB"],
             "joystick": ["stick"]}


def defined_ids(root: Node) -> Set[str]:
    ids = set()
    for node, _ in walk(root):
        if node.key in DEFINING_KEYWORDS and node.first_string():
            ids.add(node.first_string())
        ids.update(SUGAR_IDS.get(node.key, []))
    return ids


def problems(nodes: List[Node], parent_key: Optional[str], reference: Reference,
             known_ids: Set[str]) -> Dict[str, List[str]]:
    """What the Dart compiler lets through, on the given nodes (a block whose parent is `parent_key`,
    or a whole program's root with parent_key=None)."""
    found = {"placement": [], "arity": [], "color": [], "reference": []}
    for top in nodes:
        for node, parent in walk(top):
            parent_name = parent.key if parent is not None else parent_key
            if parent_name is not None and (parent_name, node.key) not in reference.placements:
                found["placement"].append(f"{node.name} inside {parent_name}")
            if len(node.args) not in reference.arities.get(node.key, set()):
                found["arity"].append(f"{node.name} with {len(node.args)} arguments")
            if node.key in COLOR_KEYWORDS and node.first_string() and not COLOR.match(node.first_string()):
                found["color"].append(f"{node.name}({node.first_string()!r})")
            if node.key in REFERENCE_KEYWORDS and node.first_string() \
                    and node.first_string() not in known_ids and node.first_string() != "hero":
                found["reference"].append(f"{node.name}({node.first_string()!r}): no such id")
    return {key: value for key, value in found.items() if value}


# ---------- "not hollow": what a game written for the training data must also pass ----------

# Actions that act on the "source" (the entity whose event fired) unless an id comes first, with the argument
# count of the form without an id. A bar's OnValue (also when a ramp passes it) runs with no source at all
# (action_executor.dart: "force a null source, like bar OnValue triggers"), and a Wave runs on a hidden entity at
# (-1, -1) (game_def_builder.dart, "__waves__"): there these forms do nothing visible.
SOURCE_ONLY = {"bounceback": None, "updatespeed": None, "setcolor": None, "changevar": None,
               "deletecollider": None, "spawnhere": None, "rotate": None, "lock": None, "slide": None,
               "whenvar": None, "setvelocity": 2, "applyimpulse": 2, "gridstep": 2, "setscale": 1,
               "setopacity": 1, "setvisible": 1, "setenable": 1, "deleteentity": 0, "flash": 2, "burst": 2}


def sourceless(root: Node) -> List[str]:
    found = []
    for node, parent in walk(root):
        if node.key not in ("onvalue", "wave"):
            continue
        for inner, _ in walk(node):
            if inner is node or inner.key not in SOURCE_ONLY:
                continue
            count = SOURCE_ONLY[inner.key]
            if count is None or len(inner.args) == count:
                where = "a bar's OnValue (no entity runs it)" if node.key == "onvalue" else \
                    "a Wave (it runs on a hidden entity off screen)"
                found.append(f"{inner.name}(...) without an id inside {where}: name the entity it acts on")
    return found


EVENT_KEYS = {"oncollision", "ontapped", "onelapsedtime", "onsignal", "ongridstep", "ongridblocked", "onswipe",
              "onvalue", "phase", "wave", "delay", "whenbar", "whenvar"}
ENDING_KEYS = {"gameover", "winat", "lives", "goal", "hostile", "hostilepatrol", "nextlevel"}


def string_uses(root: Node) -> Dict[str, int]:
    """How often each string appears as an argument anywhere, but in id(...)."""
    uses: Dict[str, int] = {}
    for node, _ in walk(root):
        if node.key == "id":
            continue
        for value in node.args:
            if isinstance(value, str):
                uses[value] = uses.get(value, 0) + 1
    return uses


def hollow(root: Node) -> List[str]:
    """Signs of a game that compiles but does not work: empty blocks, a template nobody spawns, a signal
    nobody listens to (or that nothing sends), a bar nothing changes, no way to win or lose, too few parts."""
    found = []
    uses = string_uses(root)
    entity_count = 0
    keys = set()
    for node, parent in walk(root):
        keys.add(node.key)
        if node.key == "entity":
            entity_count += 1
            if node.child("template") is not None:
                template_id = node.child("id").first_string() if node.child("id") else None
                if template_id and uses.get(template_id, 0) == 0:
                    found.append(f'template "{template_id}" is never spawned or used')
        # an empty branch of a randomAction is the "sometimes nothing happens" idiom (Last Light uses it)
        if node.key in ("actions", "events") and not node.children \
                and not (parent is not None and parent.key == "randomaction"):
            found.append(f"empty {node.name} block")
        if node.key in EVENT_KEYS and node.child("actions") is None:
            found.append(f"{node.name} without an Actions block")
        if node.key == "broadcast" and node.first_string():
            # door("x") is sugar for an OnSignal("x") that opens the door (entity_builder.dart)
            listeners = [other for other, _ in walk(root)
                         if other.key in ("onsignal", "door") and other.first_string() == node.first_string()]
            if not listeners:
                found.append(f'broadcast("{node.first_string()}") but nothing listens to it')
        if node.key == "onsignal" and node.first_string() and uses.get(node.first_string(), 0) < 2:
            found.append(f'OnSignal("{node.first_string()}") but nothing sends it')
        if node.key == "bar":
            bar_id = node.first_string()
            if bar_id and uses.get(bar_id, 0) < 2:
                found.append(f'Bar("{bar_id}") is never changed or read')
    found += sourceless(root)
    if not keys & ENDING_KEYS:
        found.append("no way to win or lose (no gameOver, winAt, lives, goal, hostile or nextLevel)")
    if entity_count + sum(1 for node, _ in walk(root) if node.key in LEVEL_CHILDREN[1:]) < 3:
        found.append("fewer than 3 entities")
    return sorted(set(found))
