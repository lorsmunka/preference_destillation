"""The problems: from every version of every game, `/insert` cuts (a block taken out; the cursor sits where it
was) and `/edit` pairs (a block changed by a rule; the target is always the real block, the "before" is the
changed one). Each problem carries the short input the model sees (no agentic step: the cursor gives the
place, the #names the parts):

    WINDOW   the Window's own settings
    NAMES    one line per entity / group / bar / signal of the game (without the cut block)
    CURSOR   where the cursor is
    BLOCK    for /edit the block under the cursor; for /insert inside an entity, that entity with <cursor>
    #name    the full code of every name the target uses (at most MAX_NAMED, each at most MAX_NAMED_CHARS)

The command itself is written later by Gemma (it sees the same plus the target). Splits: the held-out human games
and ~15% of the new games (per game, stable) are test.

    python -m library.domain.game_dsl.problems [versions per train game]     (default 30; test games get 5)
"""

import json
import random
import re
import sys
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .check import check_sources
from .compiler import compile_programs
from .games import human_games, new_games
from .language import LEVEL_CHILDREN, Node, parse_fragment, parse_program, walk
from .manual import sound_and_music_names
from .versions import PALETTE, game_names, make_versions

PACKAGE = Path(__file__).resolve().parent
OUTPUT = PACKAGE / "problems"
CURSOR = "<cursor>"
INSERTS_PER_VERSION = 10
EDITS_PER_VERSION = 5
INSERT_KINDS = {"entity": 0.35, "event": 0.30, "actions": 0.25, "window": 0.10}
WINDOW_ITEMS = {"bar", "lives", "winat", "wave", "dpad", "joystick", "wallbox", "paddle", "goal", "hostilepatrol"}
EVENT_NAMES = {"oncollision", "ontapped", "onelapsedtime", "onsignal", "ongridstep", "ongridblocked", "onswipe",
               "sequence"}
MAX_NAMED = 3
MAX_EVENTS_LISTED = 5
MAX_NAMED_CHARS = 1500
TEST_SHARE_NEW = 15   # percent of the new games
TEST_VERSIONS = 5     # versions per test game: more copies of the same held-out games add little


# ---------- spans and text ----------

def line_span(source: str, start: int, end: int) -> Tuple[int, int]:
    """Widen a block's span to whole lines when nothing else shares them."""
    line_start = source.rfind("\n", 0, start) + 1
    if source[line_start:start].strip() == "":
        start = line_start
    line_end = source.find("\n", end)
    line_end = len(source) if line_end < 0 else line_end
    if source[end:line_end].strip() == "":
        end = min(line_end + 1, len(source))
    return start, end


def block_text(source: str, start: int, end: int) -> str:
    """The text of a span, its indentation taken off (the indentation of the line it starts on)."""
    line_start = source.rfind("\n", 0, start) + 1
    indent = len(source[line_start:]) - len(source[line_start:].lstrip(" \t"))
    lines = source[start:end].rstrip("\n").split("\n")
    lines = [lines[0].lstrip(" \t")] + [line[indent:] if line[:indent].strip() == "" else line.lstrip(" \t")
                                         for line in lines[1:]]
    return "\n".join(lines)


def label(node: Node, source: str) -> str:
    """A short name for a block, as the cursor line shows it: #id for an entity, the keyword with its
    arguments for anything else (shortened), the bare keyword for a block without arguments."""
    if node.key == "entity":
        entity_id = node.child("id").first_string() if node.child("id") else None
        return f"#{entity_id}" if entity_id else "an Entity"
    if node.key in ("oncollision", "onsignal", "bar") and node.first_string():
        return f"{node.name} #{node.first_string()}"
    if not node.args:
        return node.name
    arguments = source[node.arg_spans[0][0]:node.arg_spans[-1][1]]
    return f"{node.name}({arguments if len(arguments) <= 30 else arguments[:27] + '...'})"


# ---------- the short input ----------

MOVES = {"automove": "AutoMove", "control": "Control", "path": "Path", "chase": "chase", "gridmove": "GridMove"}


def name_lines(root: Node) -> List[str]:
    lines, signals = [], set()
    for node, _ in walk(root):
        if node.key == "entity" and node.child("id"):
            entity_id = node.child("id").first_string()
            kind = "group " if node.child("template") else "entity"
            shape = next((shape.name for shape_block in node.children if shape_block.key == "type"
                          for shape in shape_block.children), "")
            move = next((MOVES[move.key] for movement in node.children if movement.key == "movement"
                         for move in movement.children if move.key in MOVES), "")
            events = []
            for events_block in node.children:
                for event in (events_block.children if events_block.key == "events" else []):
                    event_label = event.name + (f" #{event.first_string()}" if event.first_string() else "")
                    if event_label not in events:
                        events.append(event_label)
            if len(events) > MAX_EVENTS_LISTED:
                events = events[:MAX_EVENTS_LISTED] + [f"+{len(events) - MAX_EVENTS_LISTED} more"]
            lines.append(f"#{entity_id} {kind} {shape}{', ' + move if move else ''}"
                         f"{'; ' + ', '.join(events) if events else ''}")
        elif node.key in ("paddle", "goal", "hostilepatrol") and node.first_string():
            lines.append(f"#{node.first_string()} entity ({node.name})")
        elif node.key == "bar":
            lines.append(f"#{node.first_string() or 'score'} bar" + ("" if node.first_string() else " (default)"))
        elif node.key == "lives":
            lines.append("#lives bar")
        elif node.key in ("broadcast", "onsignal") and node.first_string():
            signals.add(node.first_string())
    return lines + [f"#{signal} signal" for signal in sorted(signals)]


def window_line(root: Node, source: str) -> str:
    keep = ("size", "worldsize", "camerafollow", "background", "gravity", "music")
    return "  ".join(source[child.start:child.end] for child in root.children if child.key in keep)


def entity_blocks(root: Node, source: str) -> Dict[str, str]:
    blocks = {}
    for node, _ in walk(root):
        if node.key == "entity" and node.child("id") and node.child("id").first_string():
            start, end = line_span(source, node.start, node.end)
            blocks[node.child("id").first_string()] = block_text(source, start, end)
        elif node.key == "bar" and node.first_string():
            start, end = line_span(source, node.start, node.end)
            blocks[node.first_string()] = block_text(source, start, end)
    return blocks


def used_names(text: str, names: set) -> List[str]:
    found = []
    for node in parse_fragment(text):
        for inner, _ in walk(node):
            for value in inner.args:
                if isinstance(value, str) and value in names and value not in found and inner.key != "id":
                    found.append(value)
    return found


def short_input(game: str, root: Node, target: str, own_id: Optional[str]) -> Dict:
    names = game_names(root) | {"lives"}
    blocks = entity_blocks(root, game)
    mentioned = [name for name in used_names(target, names | {own_id} if own_id else names) if name != own_id]
    defined = set(blocks) | {line.split()[0][1:] for line in name_lines(root)}
    named = {}
    for name in mentioned:
        if name in blocks and len(blocks[name]) <= MAX_NAMED_CHARS and len(named) < MAX_NAMED:
            named[name] = blocks[name]
    return {"window": window_line(root, game), "names": name_lines(root), "mentions": mentioned,
            "new_names": [name for name in ([own_id] if own_id else []) + mentioned if name not in defined],
            "named_blocks": named}


# ---------- /insert ----------

def cursor_line(path: List[Node], before: Optional[Node], after: Optional[Node], source: str) -> str:
    inside = " > ".join(label(node, source) if node.key not in ("window",) else "top level" for node in path)
    around = []
    if before is not None:
        around.append(f"after {label(before, source)}")
    if after is not None:
        around.append(f"before {label(after, source)}")
    return inside + (", " + ", ".join(around) if around else "")


def insert_candidates(root: Node):
    """(kind, parent path, first child index, last child index) of every block that can be cut."""
    candidates = []

    def visit(node: Node, path: List[Node]):
        for index, child in enumerate(node.children):
            if child.key == "entity" and node.key in ("window", "level") and child.child("id") is not None:
                candidates.append(("entity", path + [node], index, index))
            elif child.key in EVENT_NAMES and node.key == "events":
                candidates.append(("event", path + [node], index, index))
            elif child.key in WINDOW_ITEMS and node.key in ("window", "level") and child.key != "entity":
                candidates.append(("window", path + [node], index, index))
            visit(child, path + [node])
        if node.key == "actions" and node.children:
            count = len(node.children)
            for length in (1, 2, 3):
                for first in range(0, count - length + 1):
                    candidates.append(("actions", path + [node], first, first + length - 1))
    visit(root, [])
    return candidates


def make_insert(source: str, root: Node, candidate, problem_id: str) -> Optional[Dict]:
    kind, path, first, last = candidate
    parent = path[-1]
    start, end = line_span(source, parent.children[first].start, parent.children[last].end)
    target = block_text(source, start, end)
    game = source[:start] + source[end:]
    try:
        game_root = parse_program(game)
    except ValueError:
        return None
    own_id = None
    if kind == "entity":
        own_id = parent.children[first].child("id").first_string()
    elif parent.children[first].key == "bar":
        own_id = parent.children[first].first_string()
    before = parent.children[first - 1] if first > 0 else None
    after = parent.children[last + 1] if last + 1 < len(parent.children) else None
    enclosing = next((node for node in reversed(path) if node.key in ("entity", "bar", "wave", "hostilepatrol")),
                     None)
    enclosing_text = None
    if enclosing is not None:
        enclosing_start, enclosing_end = line_span(source, enclosing.start, enclosing.end)
        cut_start, cut_end = start - enclosing_start, end - enclosing_start
        raw = source[enclosing_start:enclosing_end]
        indent = len(source[start:]) - len(source[start:].lstrip(" \t")) if source[start - 1:start] == "\n" else 0
        marked = raw[:cut_start] + " " * indent + CURSOR + ("\n" if raw[cut_start - 1:cut_start] == "\n" else " ") \
            + raw[cut_end:]
        enclosing_text = block_text(marked, 0, len(marked))
    return {"id": problem_id, "task": "insert", "kind": kind, "target": target,
            "cursor": cursor_line(path, before, after, source), "block": enclosing_text,
            "cut": [start, end], **short_input(game, game_root, target, own_id)}


# ---------- /edit ----------

NUMBER_KEYS = {"speed", "startposition", "rectangle", "circle", "diamond", "star", "ring", "polygon", "triangle",
               "onelapsedtime", "repeatcount", "shake", "burst", "flash", "initial", "updatebar", "setvelocity",
               "applyimpulse", "waypoint", "dwell", "gridmove", "direction", "setscale", "setopacity", "delay",
               "onvalue", "chase", "updatespeed", "text", "var", "changevar", "phase", "gravity"}
SHAPE_SWAPS = {"circle": lambda size: f"circle({size // 2})", "rectangle": lambda size: f"rectangle({size}, {size})",
               "diamond": lambda size: f"diamond({size}, {size})", "star": lambda size: f"star(5, {size // 2})",
               "polygon": lambda size: f"polygon(6, {size // 2})", "ring": lambda size: f"ring({size // 2})",
               "triangle": lambda size: f'triangle({size}, "up")'}
FLAGS = {"solid", "oneway", "pushable", "facing", "gravity"}
# the kind of change is drawn first (then one change of that kind), so the rare kinds are not drowned out
MUTATION_WEIGHTS = {"number": 0.25, "drop_action": 0.18, "drop_event": 0.12, "colour": 0.08, "sound": 0.07,
                    "shape": 0.1, "drop_movement": 0.1, "drop_property": 0.1}


def shape_size(shape: Node) -> int:
    numbers = [value for value in shape.args if isinstance(value, float)]
    if shape.key in ("circle", "ring") and numbers:
        return int(numbers[0] * 2)
    if shape.key in ("star", "polygon") and len(numbers) > 1:
        return int(numbers[1] * 2)
    return int(numbers[0]) if numbers else 40


def format_number(value: float) -> str:
    return str(int(value)) if value == int(value) else f"{value:.2f}".rstrip("0")


def mutations(block: str, random_generator: random.Random, sounds: List[str]) -> List[Tuple[str, int, int, str]]:
    """Every possible single change: (name, start, end, replacement) within the block's text."""
    found = []
    nodes = parse_fragment(block)
    for top in nodes:
        for node, parent in walk(top):
            if node.key in NUMBER_KEYS:
                for value, (start, end) in zip(node.args, node.arg_spans):
                    if isinstance(value, float) and value not in (0, -1):
                        factor = random_generator.choice((0.5, 2, 1.5, 0.7, 3))
                        changed = value * factor if abs(value) < 50 else value + random_generator.choice(
                            (-200, -100, 100, 200, 300))
                        if node.key in ("startposition", "waypoint", "rectangle", "circle", "initial", "burst",
                                        "repeatcount", "updatebar", "onvalue", "var", "changevar"):
                            changed = round(changed)
                        if changed != value and changed != 0:
                            found.append(("number", start, end, format_number(changed)))
            if node.key in ("color", "setcolor", "flash", "burst", "background"):
                for value, (start, end) in zip(node.args, node.arg_spans):
                    if isinstance(value, str) and re.fullmatch(r"#?[0-9a-fA-F]{6}", value):
                        found.append(("colour", start, end, f'"{random_generator.choice(PALETTE)}"'))
            if node.key == "sound" and node.args:
                start, end = node.arg_spans[0]
                found.append(("sound", start, end, f'"{random_generator.choice(sounds)}"'))
            if node.key == "type" and node.children:
                shape = node.children[0]
                options = [name for name in SHAPE_SWAPS if name != shape.key]
                if shape.key in SHAPE_SWAPS:
                    found.append(("shape", shape.start, shape.end,
                                  SHAPE_SWAPS[random_generator.choice(options)](shape_size(shape))))
            if parent is not None and parent.key == "actions" and len(parent.children) >= 2:
                start, end = line_span(block, node.start, node.end)
                found.append(("drop_action", start, end, ""))
            if parent is not None and parent.key == "events" and len(parent.children) >= 2:
                start, end = line_span(block, node.start, node.end)
                found.append(("drop_event", start, end, ""))
            if parent is not None and parent.key == "entity" and node.key in {"movement"} | FLAGS:
                start, end = line_span(block, node.start, node.end)
                found.append(("drop_" + ("movement" if node.key == "movement" else "property"), start, end, ""))
    return found


def make_edit(source: str, root: Node, node: Node, path: List[Node], problem_id: str,
              random_generator: random.Random, sounds: List[str]) -> Optional[Dict]:
    start, end = line_span(source, node.start, node.end)
    target = block_text(source, start, end)
    options = mutations(target, random_generator, sounds)
    if not options:
        return None
    chosen = []
    for _ in range(1 if random_generator.random() < 0.7 else 2):
        kinds = sorted({option[0] for option in options if option not in chosen})
        if not kinds:
            break
        kind = random_generator.choices(kinds, [MUTATION_WEIGHTS[kind] for kind in kinds])[0]
        chosen.append(random_generator.choice([option for option in options if option[0] == kind
                                               and option not in chosen]))
    chosen.sort(key=lambda change: change[1], reverse=True)
    last_start = len(target) + 1
    before = target
    for _, change_start, change_end, replacement in chosen:
        if change_end > last_start:          # overlapping changes: keep the later one only
            continue
        before = before[:change_start] + replacement + before[change_end:]
        last_start = change_start
    if before == target:
        return None
    game = source[:start] + before + ("\n" if not before.endswith("\n") and source[end - 1:end] == "\n" else "") \
        + source[end:]
    try:
        game_root = parse_program(game)
    except ValueError:
        return None
    own_id = node.child("id").first_string() if node.key == "entity" and node.child("id") else None
    return {"id": problem_id, "task": "edit", "kind": "entity" if node.key == "entity" else "event",
            "mutations": sorted({change[0] for change in chosen}), "target": target, "before": before,
            "cursor": "on " + " > ".join(label(step, source) for step in path[1:] + [node]), "block": before,
            "cut": [start, end], "game_before": game, **short_input(game, game_root, target + "\n" + before, own_id)}


def edit_candidates(root: Node):
    candidates = []

    def visit(node: Node, path: List[Node]):
        for child in node.children:
            if child.key == "entity" and node.key in ("window", "level"):
                candidates.append((child, path + [node]))
            elif child.key in EVENT_NAMES and node.key == "events":
                candidates.append((child, path + [node]))
            visit(child, path + [node])
    visit(root, [])
    return [(node, [step for step in path if step.key not in ("events",)]) for node, path in candidates]


def student_input(problem: Dict, command: str) -> str:
    """The text the model reads; `command` is the whole command line ("/insert ...", "/edit ...", or "/insert"
    alone for autocomplete)."""
    parts = [f"WINDOW {problem['window']}", "NAMES", *problem["names"], f"CURSOR {problem['cursor']}"]
    if problem["block"]:
        parts += ["BLOCK", problem["block"]]
    for name, code in problem["named_blocks"].items():
        parts += [f"#{name}", code]
    parts.append(f"COMMAND {command}")
    return "\n".join(parts)


def as_typed(problem: Dict, command: str) -> Dict:
    """The problem as the app shows it once the command is typed: the app fetches the code of the #names in the
    command only (so none for a loose or empty command)."""
    typed = set(re.findall(r"#([A-Za-z_][A-Za-z0-9_]*)", command))
    return {**problem, "named_blocks": {name: code for name, code in problem["named_blocks"].items() if name in typed}}


# ---------- all of it ----------

def split_of(game: Dict) -> str:
    if game["origin"] == "human":
        return game["split"]
    return "test" if zlib.crc32(game["key"].encode()) % 100 < TEST_SHARE_NEW else "train"


def problems_of_version(source: str, version_id: str, random_generator: random.Random, sounds: List[str]):
    root = parse_program(source)
    found = []
    candidates = insert_candidates(root)
    by_kind = {kind: [candidate for candidate in candidates if candidate[0] == kind] for kind in INSERT_KINDS}
    for index in range(INSERTS_PER_VERSION):
        kinds = [kind for kind in INSERT_KINDS if by_kind[kind]]
        kind = random_generator.choices(kinds, [INSERT_KINDS[kind] for kind in kinds])[0]
        problem = make_insert(source, root, random_generator.choice(by_kind[kind]), f"{version_id}|i{index}")
        if problem is not None:
            found.append(problem)
    edit_options = edit_candidates(root)
    for index in range(EDITS_PER_VERSION):
        node, path = random_generator.choice(edit_options)
        problem = make_edit(source, root, node, path, f"{version_id}|e{index}", random_generator, sounds)
        if problem is not None:
            found.append(problem)
    return found


def main():
    versions_per_game = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    sounds = [name for name in sound_and_music_names() if not name.startswith("bg_")]
    games = human_games() + new_games()
    for game in games:
        game["key"] = game["slug"] if game["origin"] == "human" else f"{game['family']}/{game['slug']}"
    clean = check_sources({game["key"]: game["source"] for game in games})
    games = [game for game in games if not clean[game["key"]] or game["slug"] == "sundown_herd"]
    OUTPUT.mkdir(exist_ok=True)
    counts = {"train": 0, "test": 0}
    versions_file = open(OUTPUT / "versions.jsonl", "w", encoding="utf-8")    # derived files, rebuilt each time
    problems_file = open(OUTPUT / "problems.jsonl", "w", encoding="utf-8")
    for game_index, game in enumerate(games):
        split = split_of(game)
        check = (lambda sources: {name: [] for name in sources}) if game["slug"] == "sundown_herd" else check_sources
        count = versions_per_game if split == "train" else min(TEST_VERSIONS, versions_per_game)
        versions = [game["source"]] + make_versions(game["source"], count - 1, game_index + 1, check)
        problems = []
        for version_index, source in enumerate(versions):
            version_id = f"{game['key']}|v{version_index}"
            versions_file.write(json.dumps({"id": version_id, "game": game["key"], "family": game["family"],
                                            "split": split, "source": source}, ensure_ascii=False) + "\n")
            random_generator = random.Random(zlib.crc32(version_id.encode()))
            problems += problems_of_version(source, version_id, random_generator, sounds)
        befores = compile_programs({problem["id"]: problem.pop("game_before") for problem in problems
                                    if problem["task"] == "edit"})
        for problem in problems:
            if problem["task"] == "edit" and not befores[problem["id"]][0]:
                continue
            problem.update({"game": game["key"], "family": game["family"], "origin": game["origin"], "split": split})
            problems_file.write(json.dumps(problem, ensure_ascii=False) + "\n")
            counts[split] += 1
    versions_file.close()
    problems_file.close()
    print(f"{len(games)} games -> {counts['train']:,} train / {counts['test']:,} test problems -> {OUTPUT}")


if __name__ == "__main__":
    main()
