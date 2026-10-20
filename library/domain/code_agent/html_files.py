"""The HTML pages of a base project (user, 2026-10-21: "just add html only projects, keep it easy").

A page's places are its elements with an id (empty elements like <input> aside): the lines from
the start tag to the end tag. A version renames ids and page names only inside the attributes that
point at them (id, href, for, action, ...), never in the text people read, and shuffles the
sections of <main> (the elements directly inside it, one blank line between them; an HTML comment
line right above one moves with it). Every element but the empty ones is closed explicitly.
"""

import random
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Dict, List, Optional, Set, Tuple

VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
LINK_ATTRIBUTES = ("id", "href", "for", "action", "src", "form", "list", "headers", "aria-labelledby",
                   "aria-describedby", "aria-controls")
ATTRIBUTE_VALUE = re.compile(r'(?<![\w-])(' + "|".join(map(re.escape, LINK_ATTRIBUTES)) + r')="([^"]*)"')
NAME = re.compile(r"[A-Za-z_][\w-]*")


def is_page(file_name: str) -> bool:
    return file_name.endswith(".html")


@dataclass
class Element:
    tag: str
    element_id: Optional[str]
    start: int             # line of the start tag
    end: Optional[int]     # line of the end tag
    parent: Optional[int]  # index of the enclosing element


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.elements: List[Element] = []
        self.open: List[int] = []
        self.problems: List[str] = []

    def handle_starttag(self, tag, attributes):
        if tag in VOID_TAGS:
            return
        parent = self.open[-1] if self.open else None
        self.elements.append(Element(tag, dict(attributes).get("id"), self.getpos()[0], None, parent))
        self.open.append(len(self.elements) - 1)

    def handle_startendtag(self, tag, attributes):
        pass   # <br/>: nothing to close

    def handle_endtag(self, tag):
        if tag in VOID_TAGS:
            return
        if not self.open or self.elements[self.open[-1]].tag != tag:
            self.problems.append(f"line {self.getpos()[0]}: </{tag}> does not close the open element")
            return
        self.elements[self.open.pop()].end = self.getpos()[0]


def parse_page(lines: List[str]) -> PageParser:
    parser = PageParser()
    parser.feed("\n".join(lines))
    parser.close()
    parser.problems += [f"line {parser.elements[index].start}: <{parser.elements[index].tag}> is never closed"
                        for index in parser.open]
    return parser


def page_places(lines: List[str]) -> Dict[str, Tuple[int, int]]:
    """id -> (first line, last line) of every element with an id."""
    return {element.element_id: (element.start, element.end) for element in parse_page(lines).elements
            if element.element_id and element.end is not None}


def page_problems(lines: List[str]) -> List[str]:
    parser = parse_page(lines)
    ids = [element.element_id for element in parser.elements if element.element_id]
    return parser.problems + [f"id {element_id} used twice" for element_id in sorted(set(ids)) if ids.count(element_id) > 1]


def page_names(lines: List[str]) -> Set[str]:
    """The names inside the attributes that point at ids and pages (what a version renames)."""
    return {name for match in ATTRIBUTE_VALUE.finditer("\n".join(lines)) for name in NAME.findall(match.group(2))}


def page_words(lines: List[str]) -> Set[str]:
    """Every word of the page: tags, attributes, values and text (what an explanation may name)."""
    return set(re.findall(r"[A-Za-z_]\w*", "\n".join(lines)))


def rename_page(text: str, renames: Dict[str, str]) -> str:
    """Ids and page names replaced inside the attributes that point at them."""
    if not renames:
        return text

    def renamed(match):
        value = NAME.sub(lambda name: renames.get(name.group(0), name.group(0)), match.group(2))
        return f'{match.group(1)}="{value}"'
    return ATTRIBUTE_VALUE.sub(renamed, text)


def reorder_page(text: str, random_generator: random.Random) -> str:
    """The sections of <main> in a shuffled order; the page as it is when it has no <main>, fewer
    than two sections, or anything but blank lines between them."""
    lines = text.split("\n")
    elements = parse_page(lines).elements
    main = next((index for index, element in enumerate(elements) if element.tag == "main"), None)
    if main is None:
        return text
    children = [element for element in elements if element.parent == main]
    if len(children) < 2 or children[0].start <= elements[main].start or children[-1].end >= elements[main].end:
        return text
    starts = []
    for child in children:
        start = child.start
        while start > 1 and lines[start - 2].strip().startswith("<!--"):
            start -= 1
        starts.append(start)
    for child, next_start in zip(children, starts[1:]):
        if next_start <= child.end or any(line.strip() for line in lines[child.end:next_start - 1]):
            return text
    blocks = [lines[start - 1:child.end] for child, start in zip(children, starts)]
    random_generator.shuffle(blocks)
    middle = []
    for block in blocks:
        if middle:
            middle.append("")
        middle += block
    return "\n".join(lines[:starts[0] - 1] + middle + lines[children[-1].end:])
