"""The manual (manual.md): the app's own Docs tab (lib/screens/tabs/docs_tab.dart), its text and examples taken
out of the widget code verbatim, plus the sound and music names and a table of which keywords go inside which
block. Also writes reference.json (placements and argument counts, for language.problems), collected from the
docs examples and all 26 human games: only which keyword sits where and how many arguments it takes, no game
content, so the held-out games are still never shown.

    python -m library.domain.game_dsl.manual
"""

import json
import re
from pathlib import Path

from .compiler import QR_FLUTTER
from .games import human_games
from .language import build_reference, parse_fragment

PACKAGE = Path(__file__).resolve().parent
MANUAL = PACKAGE / "manual.md"
REFERENCE = PACKAGE / "reference.json"
DOCS = QR_FLUTTER / "lib" / "screens" / "tabs" / "docs_tab.dart"
SOUNDS = QR_FLUTTER / "lib" / "audio" / "sound_effects.dart"
KEYWORDS = QR_FLUTTER / "lib" / "compiler" / "dsl_keywords.dart"
ESCAPES = {"n": "\n", "'": "'", '"': '"', "\\": "\\", "$": "$", "t": "\t"}
# What the engine does that the Docs do not say: checked in lib/engine and lib/compiler (the first two), or found
# by several of the agents that wrote the new games (2026-11-07), each reading the engine's source.
ENGINE_NOTES = [
    "A bar's OnValue actions (also when a ramp passes the value) run with no entity: actions that act on the "
    "running entity (setColor, changeVar, spawnHere, burst or flash without an id, ...) do nothing there. Name the "
    "entity, or broadcast a signal.",
    "Wave actions run on a hidden entity at (-1, -1): spawnHere there spawns off screen; spawn with coordinates.",
    "OnValue fires only when an update lands exactly on that value: updateBar(\"x\", -3) can jump over OnValue(0). "
    "loopCount(3) then updateBar(\"x\", -1) steps through every value.",
    "A collision fires once when the overlap starts, not every frame.",
    "restart() puts every bar back to its value at the start of the level, lives included.",
    "A disabled entity (setEnable(false)) still runs OnSignal and its pending delays, but not its timers, taps or "
    "collisions. A hidden one (setVisible(false)) can still be tapped.",
    "setColor only colours the entity that runs it; to recolour another, broadcast a signal it listens to.",
    "OnTapped(\"template\") never matches clones (their ids are template:1, template:2, ...): give the template a "
    "plain OnTapped.",
]


def read_literals(text: str, position: int):
    """Adjacent Dart string literals starting at `position` -> (joined value, end position)."""
    parts = []
    while True:
        while position < len(text) and text[position] in " \t\r\n":
            position += 1
        if position >= len(text) or text[position] not in "'\"":
            return "".join(parts), position
        quote = text[position]
        position += 1
        value = []
        while text[position] != quote:
            if text[position] == "\\":
                value.append(ESCAPES.get(text[position + 1], text[position + 1]))
                position += 2
            else:
                value.append(text[position])
                position += 1
        parts.append("".join(value))
        position += 1


def docs_entries():
    text = DOCS.read_text(encoding="utf-8")
    body = text[text.index("List<Widget> _sections()"):text.index("class _ContentsButton")]
    pattern = re.compile(r"(_Section|_SubSection|_ExampleBlock|_PatternBlock|_SoundList)\(|"
                         r"\b(title|intro|caption|source|heading|explanation|music):")
    entries, current = [], None
    for match in pattern.finditer(body):
        if match.group(1):
            if match.group(1) == "_SoundList":
                entries.append({"kind": "_SoundList", "music": "music: true" in body[match.end():match.end() + 20]})
                current = None
                continue
            current = {"kind": match.group(1)}
            entries.append(current)
        elif current is not None and match.group(2) != "music":
            current[match.group(2)], _ = read_literals(body, match.end())
    return entries


def sound_and_music_names():
    sound_text = SOUNDS.read_text(encoding="utf-8")
    sounds = re.findall(r"'([a-z0-9_]+)'", sound_text[sound_text.index("builtIn = ["):sound_text.index("];")])
    formers = re.findall(r"'([a-z0-9_]+)': '[a-z0-9_]+'", sound_text)      # old names still accepted
    music = [path.stem for path in (QR_FLUTTER / "Sound_effects").glob("bg_*.mp3")]
    return sorted(set(sounds) | set(formers) | set(music))


def docs_examples():
    return [entry["source"] for entry in docs_entries() if entry["kind"] in ("_ExampleBlock", "_PatternBlock")]


def build():
    entries = docs_entries()
    sound_text = SOUNDS.read_text(encoding="utf-8")
    sound_names = re.findall(r"'([a-z0-9_]+)'", sound_text[sound_text.index("builtIn = ["):sound_text.index("];")])
    music_names = sorted(path.stem for path in (QR_FLUTTER / "Sound_effects").glob("bg_*.mp3"))
    fragments = []
    for source in docs_examples():
        try:
            parse_fragment(source)
            fragments.append(source)
        except ValueError:
            pass
    reference = build_reference([game["source"] for game in human_games()], fragments)
    REFERENCE.write_text(json.dumps(reference.to_json(), indent=1) + "\n", encoding="utf-8")

    lines = ["# The game language: a reference manual", "",
             "Games are written in a small language. A program is one Window block. The only grammar rule:",
             "`keyword(parameters) { child blocks }` -- both parts optional. Parentheses hold only strings, numbers,",
             "true/false and random(min, max); everything nested goes in braces. Keywords are case-insensitive, but",
             "write them as below. A keyword in the wrong block is silently ignored, so follow the placement table.",
             ""]
    for entry in entries:
        kind = entry["kind"]
        if kind in ("_Section", "_SubSection"):
            lines += ["", ("## " if kind == "_Section" else "### ") + entry["title"], "", entry["intro"], ""]
            # the Docs tab shows the names as pads built from the sound files (no text to copy), so they are
            # written here from the same sources
            if entry["title"] == "Sound effects":
                lines += ["Sound names: " + ", ".join(sound_names), ""]
            elif entry["title"] == "Background music":
                lines += ["Music tracks: " + ", ".join(music_names), ""]
        elif kind == "_ExampleBlock":
            lines += ([entry["caption"], ""] if entry.get("caption") else []) + ["```", entry["source"], "```", ""]
        elif kind == "_PatternBlock":
            lines += [f"**{entry['heading']}.** {entry['explanation']}", "", "```", entry["source"], "```", ""]

    lines += ["", "## Engine notes (from the engine's source, not in the Docs)", ""] + [f"- {note}" for note in ENGINE_NOTES]
    keyword_case = {word.lower(): word for word in
                    re.findall(r"static const \w+ = '(\w+)';", KEYWORDS.read_text(encoding="utf-8"))}
    children = {}
    for parent, child in reference.placements:
        children.setdefault(parent, set()).add(child)
    lines += ["", "## Placement table: which keywords go inside which block", "",
              "(collected from the built-in games, the examples above and the compiler)", "",
              "In a game with Level blocks, every Entity (and dpad, joystick, paddle, goal, HostilePatrol, wallBox, "
              "Wave) must sit inside a Level: the ones written directly in the Window are ignored. Size, colours, "
              "music, gravity, Bars, lives and winAt stay in the Window.", ""]
    for parent in sorted(children, key=lambda name: (len(children[name]) < 5, name)):
        listed = ", ".join(sorted(keyword_case.get(child, child) for child in children[parent]))
        lines.append(f"- {keyword_case.get(parent, parent)}: {listed}")
    MANUAL.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{MANUAL.name}: {MANUAL.stat().st_size:,} bytes; {REFERENCE.name}: {len(reference.placements)} placements")


CARD = PACKAGE / "reference_card.md"


def build_card():
    """A short version of the manual (~1/4 of its length) for the teacher's prompt: the grammar, units, one example
    call per keyword (the shortest one in the docs or the games with the most common argument count), the
    placement table, sound and music names, the engine notes and one complete program."""
    from collections import Counter
    from .language import walk as walk_nodes, parse_program
    manual = MANUAL.read_text(encoding="utf-8")
    calls = {}
    sources = docs_examples() + [game["source"] for game in human_games("train")]
    for source in sources:
        try:
            nodes = parse_fragment(source)
        except ValueError:
            continue
        for top in nodes:
            for node, _ in walk_nodes(top):
                if node.args:
                    text = source[node.start:node.arg_spans[-1][1] + 1]
                else:
                    text = node.name + (" { ... }" if node.children else "()")
                if "\n" not in text:
                    calls.setdefault(node.key, []).append((len(node.args), text))
    keyword_case = {word.lower(): word for word in
                    re.findall(r"static const \w+ = '(\w+)';", KEYWORDS.read_text(encoding="utf-8"))}
    lines = ["# The game language: reference card", "",
             "A program is one Window block. Grammar: `keyword(parameters) { child blocks }`, both parts optional.",
             "Parentheses hold only strings, numbers, true/false and random(min, max); everything nested goes in braces.",
             "Keywords are case-insensitive; a keyword in the wrong block is silently ignored (see the placement table).",
             "# starts a comment. Positions and sizes are pixels from the bottom-left corner, y up; startPosition is the",
             "bottom-left corner. size(w, h) and worldSize are in units of 100 px. speed() is in units of 100 px/s,",
             "setVelocity / applyImpulse in px/s; times in seconds. Colours: \"RRGGBB\", \"AARRGGBB\" or short \"RGB\".",
             "Many actions act on the entity whose event fired unless an id comes first: setVelocity, applyImpulse,",
             "gridStep, setScale, setOpacity, setVisible, setEnable, deleteEntity, flash, burst.",
             "A template(true) Entity is a blueprint: spawn(\"id\", x, y) / spawnHere(\"id\") make clones; a template's id",
             "matches all its clones in OnCollision. broadcast(\"name\") runs every OnSignal(\"name\").",
             "", "## One example call per keyword", ""]
    for key in sorted(calls, key=lambda name: keyword_case.get(name, name).lower()):
        if key not in keyword_case:
            continue
        common = Counter(count for count, _ in calls[key]).most_common(1)[0][0]
        example = min((text for count, text in calls[key] if count == common), key=len)
        lines.append(f"- {example}")
    sound_line = next(line for line in manual.splitlines() if line.startswith("Sound names:"))
    music_line = next(line for line in manual.splitlines() if line.startswith("Music tracks:"))
    tail = manual[manual.index("## Engine notes"):]
    example = docs_entries()
    program = next(entry["source"] for entry in example if entry.get("heading") == "Paddle + bouncing ball")
    lines += ["", sound_line, music_line, "", "## A complete program", "", "```", program, "```", "", tail]
    CARD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{CARD.name}: {CARD.stat().st_size:,} bytes")


def load_reference():
    from .language import Reference
    return Reference.from_json(json.loads(REFERENCE.read_text(encoding="utf-8")))


if __name__ == "__main__":
    build()
    build_card()
