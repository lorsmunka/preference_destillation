"""The student's input from a live game in the app's editor (2026-11-08, the first QR Game Studio agent): the same
short input the problems were built from (problems.make_insert / make_edit / student_input), from the game's code,
the cursor's offset and the command the user typed. Nothing here sees a target.

    /insert   the cursor is where the new code goes (the user's `//` command line already taken out of the code)
    /edit     the block under the cursor (the innermost Entity or event around it) is rewritten
"""

import re
from typing import Dict, List, Optional, Tuple

from .language import Node, parse_program
from .problems import (CURSOR, EVENT_NAMES, MAX_NAMED, MAX_NAMED_CHARS, block_text, cursor_line, entity_blocks, label,
                       line_span, name_lines, student_input, window_line)


def enclosing_path(root: Node, offset: int) -> List[Node]:
    """The nodes whose span holds `offset`, outermost first (the Window first)."""
    path, node = [root], root
    while True:
        inner = next((child for child in node.children if child.start <= offset < child.end), None)
        if inner is None:
            return path
        path.append(inner)
        node = inner


def named_blocks(root: Node, source: str, command: str, path: List[Node]) -> Dict[str, str]:
    """The code of the #names in the command (as the app fetches it), at most MAX_NAMED, each at most
    MAX_NAMED_CHARS long; never an Entity on the cursor's path (its code is in BLOCK already, and the training
    problems only fetched names the new code itself uses)."""
    blocks = entity_blocks(root, source)
    on_path = {node.child("id").first_string() for node in path if node.key == "entity" and node.child("id")}
    named = {}
    for name in dict.fromkeys(re.findall(r"#([A-Za-z_][A-Za-z0-9_]*)", command)):
        if name not in on_path and name in blocks and len(blocks[name]) <= MAX_NAMED_CHARS and len(named) < MAX_NAMED:
            named[name] = blocks[name]
    return named


def insert_input(source: str, offset: int, command: str, indent: Optional[str] = None) -> Tuple[str, int]:
    """(the student's input, where the new code goes) for /insert at `offset` (the start of a line). `indent`: the
    indentation the user typed the command at (the training problems mark the cursor at the cut block's own
    indentation); by default the indentation of the line at the cursor."""
    if indent is None:
        line = source[offset:].split("\n", 1)[0]
        indent = line[:len(line) - len(line.lstrip(" \t"))]
    root = parse_program(source)
    path = enclosing_path(root, offset)
    parent = path[-1]
    # a node the cursor sits inside but before its children (e.g. on its keyword) counts as its parent's level
    before = next((child for child in reversed(parent.children) if child.end <= offset), None)
    after = next((child for child in parent.children if child.start >= offset), None)
    enclosing = next((node for node in reversed(path) if node.key in ("entity", "bar", "wave", "hostilepatrol")), None)
    block = None
    if enclosing is not None:
        start, end = line_span(source, enclosing.start, enclosing.end)
        marked = source[start:offset] + indent + CURSOR + "\n" + source[offset:end]
        block = block_text(marked, 0, len(marked))
    problem = {"window": window_line(root, source), "names": name_lines(root),
               "cursor": cursor_line(path, before, after, source), "block": block,
               "named_blocks": named_blocks(root, source, command, path)}
    return student_input(problem, command), offset


def edit_input(source: str, offset: int, command: str) -> Optional[Tuple[str, int, int]]:
    """(the student's input, start, end of the block it rewrites) for /edit at `offset`; None when the cursor is on
    no Entity or event."""
    root = parse_program(source)
    path = enclosing_path(root, offset)
    under = next((node for node in reversed(path)
                  if node.key == "entity" or (node.key in EVENT_NAMES)), None)
    if under is None:
        return None
    start, end = line_span(source, under.start, under.end)
    steps = [step for step in path[1:path.index(under) + 1] if step.key not in ("events",)]
    problem = {"window": window_line(root, source), "names": name_lines(root),
               "cursor": "on " + " > ".join(label(step, source) for step in steps),
               "block": block_text(source, start, end), "named_blocks": named_blocks(root, source, command, path)}
    return student_input(problem, command), start, end
