"""A synthetic project and its randomized versions (decided 2026-10-20).

A base project is a folder under projects/: 10-15 small hand-written Python files (data for the
agent, never run), or HTML pages (html_files.py: their places are the elements with an id, a place
"menu.html:opening-hours"), and project.json:

    {"description": "a bank simulator ...",
     "topics": [{"id": "withdrawal", "places": ["accounts.py:Account.withdraw", "fees.py:overdraft_fee"],
                 "find": ["the code that takes money out of an account"],
                 "explain": ["how a withdrawal works"]}],
     "renames": {"withdraw": ["take_out"], "accounts": ["bank_accounts"], "Transaction": ["Payment"]}}

A place is "file:name", the name a class, a function, "Class.method" or a top-level constant. A
topic phrase may name a renameable thing in braces ("how the {Transaction} class is used").

A version (seeded by project and number) renames some names to one of their alternatives, shuffles
the top-level functions and classes of each file and the methods of each class (__init__ stays
first), and shuffles the order of the files. A name is renamed everywhere in the code (every NAME
token, so calls, imports, attributes and same-named variables alike); in strings and comments only
a name that is not a plain word (Transaction, check_login, MAX_WITHDRAWAL). A renamed module
renames its file. The true places are looked up again in the version's code, so they are exact.
Needs Python 3.12+ (names inside f-strings are separate tokens only from 3.12 on).
"""

import ast
import builtins
import io
import json
import random
import re
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .html_files import is_page, page_names, page_places, page_problems, page_words, rename_page, reorder_page

PACKAGE = Path(__file__).resolve().parent
PROJECTS_DIR = PACKAGE / "projects"
RENAME_CHANCE = 0.6   # each renameable name gets one of its alternatives with this chance
# attribute names of the standard library / builtin types the projects call: never renameable
LIBRARY_ATTRIBUTES = {"append", "strip", "get", "items", "keys", "values", "load", "dump", "dumps", "loads", "exists",
                      "join", "split", "format", "read", "write", "open", "close", "sort", "copy", "update", "pop",
                      "remove", "insert", "lower", "upper", "startswith", "endswith", "replace", "isoformat", "today",
                      "randint", "choice", "makedirs", "path", "count", "index", "add", "extend", "clear", "now", "date"}


@dataclass
class Version:
    project: str
    number: int
    files: Dict[str, List[str]]   # file name -> lines, in the order the context shows them
    renames: Dict[str, str]       # base name -> name in this version (only the renamed ones)

    @property
    def id(self) -> str:
        return f"{self.project}~v{self.number}"

    def name_of(self, base_name: str) -> str:
        """A base name (or "Class.method") as this version calls it."""
        return ".".join(self.renames.get(part, part) for part in base_name.split("."))

    def file_of(self, base_file: str) -> str:
        stem, extension = base_file.rsplit(".", 1)
        return f"{self.renames.get(stem, stem)}.{extension}"

    def place(self, base_place: str) -> Tuple[str, str]:
        """"accounts.py:Account.withdraw" -> (file, name) in this version."""
        base_file, base_name = base_place.split(":")
        return self.file_of(base_file), self.name_of(base_name)

    def fill(self, phrase: str) -> str:
        """A topic phrase with its {Name} slots in this version's names."""
        return re.sub(r"\{([\w-]+)\}", lambda match: self.renames.get(match.group(1), match.group(1)), phrase)


def project_list() -> List[str]:
    return sorted(path.parent.name for path in PROJECTS_DIR.glob("*/project.json"))


def load_base(name: str) -> Dict:
    folder = PROJECTS_DIR / name
    files = {path.name: path.read_text(encoding="utf-8") for path in sorted([*folder.glob("*.py"), *folder.glob("*.html")])}
    meta = json.loads((folder / "project.json").read_text(encoding="utf-8"))
    return {"name": name, "files": files, **meta}


# --- names and places ---------------------------------------------------------------------------

def symbol_lines(files: Dict[str, List[str]]) -> Dict[Tuple[str, str], Tuple[int, int]]:
    """(file, name) -> (first line, last line) of every class, function, method and top-level
    constant, and of every element with an id in a page. A definition's lines include its decorators."""
    places = {}
    for file_name, lines in files.items():
        if is_page(file_name):
            places.update({(file_name, element_id): span for element_id, span in page_places(lines).items()})
            continue
        for node in ast.parse("\n".join(lines)).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                places[(file_name, node.name)] = (first_line(node), node.end_lineno)
                if isinstance(node, ast.ClassDef):
                    for member in node.body:
                        if isinstance(member, ast.FunctionDef):
                            places[(file_name, f"{node.name}.{member.name}")] = (first_line(member), member.end_lineno)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        places[(file_name, target.id)] = (node.lineno, node.end_lineno)
    return places


def symbol_kinds(files: Dict[str, List[str]]) -> Dict[Tuple[str, str], str]:
    """(file, name) -> "class", "function", "method", "constant" or "element" (a page's), for the
    places of symbol_lines."""
    kinds = {}
    for file_name, lines in files.items():
        if is_page(file_name):
            kinds.update({(file_name, element_id): "element" for element_id in page_places(lines)})
            continue
        for node in ast.parse("\n".join(lines)).body:
            if isinstance(node, ast.ClassDef):
                kinds[(file_name, node.name)] = "class"
                kinds.update({(file_name, f"{node.name}.{member.name}"): "method" for member in node.body
                              if isinstance(member, ast.FunctionDef)})
            elif isinstance(node, ast.FunctionDef):
                kinds[(file_name, node.name)] = "function"
            elif isinstance(node, ast.Assign):
                kinds.update({(file_name, target.id): "constant" for target in node.targets if isinstance(target, ast.Name)})
    return kinds


def first_line(node) -> int:
    return min([node.lineno] + [decorator.lineno for decorator in getattr(node, "decorator_list", [])])


def identifiers(files: Dict[str, List[str]]) -> Set[str]:
    """Every identifier the project defines or uses, plus its module and page names (in a page: the
    names its id and link attributes hold)."""
    names = {file_name.rsplit(".", 1)[0] for file_name in files}
    for file_name, lines in files.items():
        if is_page(file_name):
            names |= page_names(lines)
            continue
        for node in ast.walk(ast.parse("\n".join(lines))):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.arg):
                names.add(node.arg)
    return names


def known_names(files: Dict[str, List[str]]) -> Set[str]:
    """What an explanation may name: the identifiers, plus string values that look like names
    ("transfer_in", "savings") and the data file names; in a page, every word it has."""
    names = identifiers(files)
    for file_name, lines in files.items():
        if is_page(file_name):
            names |= page_words(lines)
            continue
        for node in ast.walk(ast.parse("\n".join(lines))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                names.update(re.findall(r"[A-Za-z_]\w*", node.value))
    return names


def defined_names(files: Dict[str, List[str]]) -> Set[str]:
    """Names the project itself defines: modules, classes, functions, methods, top-level constants,
    pages and the ids of their elements."""
    names = {file_name.rsplit(".", 1)[0] for file_name in files}
    for _, name in symbol_lines(files):
        names.update(name.split("."))
    return names


def check_base(base: Dict) -> List[str]:
    """Problems with a base project's project.json (empty when it is fine)."""
    files = {name: text.splitlines() for name, text in base["files"].items()}
    places, defined, used = symbol_lines(files), defined_names(files), identifiers(files)
    problems = [f"{file_name}: {problem}" for file_name, lines in files.items() if is_page(file_name)
                for problem in page_problems(lines)]
    for file_name, lines in files.items():
        if is_page(file_name):
            continue
        body = ast.parse("\n".join(lines)).body
        classes = {node.name for node in body if isinstance(node, ast.ClassDef)}
        problems += [f"{file_name}: {node.name} inherits from {ast.unparse(base)} in the same file (a shuffle can put it first)"
                     for node in body if isinstance(node, ast.ClassDef) for base in node.bases if ast.unparse(base) in classes]
    for topic in base["topics"]:
        problems += [f"topic {topic['id']}: no place {place}" for place in topic["places"]
                     if tuple(place.split(":")) not in places]
        for phrase in topic["find"] + topic["explain"]:
            problems += [f"topic {topic['id']}: slot {{{slot}}} is not renameable" for slot in re.findall(r"\{([\w-]+)\}", phrase)
                         if slot not in base["renames"]]
    taken = {}
    for name, alternatives in base["renames"].items():
        if name not in defined:
            problems.append(f"rename {name}: not defined in the project")
        if name in LIBRARY_ATTRIBUTES or hasattr(builtins, name) or name == "main":
            problems.append(f"rename {name}: also a library / builtin name")
        for alternative in alternatives:
            if alternative in used or hasattr(builtins, alternative) or alternative in LIBRARY_ATTRIBUTES:
                problems.append(f"rename {name} -> {alternative}: already a name")
            if alternative in taken:
                problems.append(f"rename {name} -> {alternative}: also an alternative of {taken[alternative]}")
            taken[alternative] = name
    return problems


# --- the version ----------------------------------------------------------------------------------

def make_version(base: Dict, number: int) -> Version:
    if sys.version_info < (3, 12):
        raise RuntimeError("code_agent versions need Python 3.12+ (f-string names are tokens only from 3.12)")
    random_generator = random.Random(f"{base['name']}-{number}")
    renames = {name: random_generator.choice(alternatives) for name, alternatives in base["renames"].items()
               if random_generator.random() < RENAME_CHANCE}
    files = {}
    for file_name, text in base["files"].items():
        stem, extension = file_name.rsplit(".", 1)
        if is_page(file_name):
            new_text = reorder_page(rename_page(text, renames), random_generator)
        else:
            new_text = reorder(rename(text, renames), random_generator)
        files[f"{renames.get(stem, stem)}.{extension}"] = new_text.splitlines()
    order = list(files)
    random_generator.shuffle(order)
    version = Version(base["name"], number, {name: files[name] for name in order}, renames)
    check_version(base, version)
    return version


def is_plain_word(name: str) -> bool:
    return name.isalpha() and name.islower()


def rename(text: str, renames: Dict[str, str]) -> str:
    """Every NAME token in `renames` replaced; in strings and comments the names that are not
    plain words, as whole words."""
    if not renames:
        return text
    prose = [name for name in renames if not is_plain_word(name)]
    prose_pattern = re.compile(r"\b(" + "|".join(map(re.escape, prose)) + r")\b") if prose else None
    line_starts = [0]
    for line in text.split("\n"):
        line_starts.append(line_starts[-1] + len(line) + 1)
    edits = []   # (start offset, end offset, new text)
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        start = line_starts[token.start[0] - 1] + token.start[1]
        end = line_starts[token.end[0] - 1] + token.end[1]
        if token.type == tokenize.NAME and token.string in renames:
            edits.append((start, end, renames[token.string]))
        elif token.type in (tokenize.STRING, tokenize.COMMENT) and prose_pattern is not None:
            new_string = prose_pattern.sub(lambda match: renames[match.group(1)], token.string)
            if new_string != token.string:
                edits.append((start, end, new_string))
    for start, end, new_text in sorted(edits, reverse=True):
        text = text[:start] + new_text + text[end:]
    return text


def is_main_guard(node) -> bool:
    return isinstance(node, ast.If) and "__name__" in ast.unparse(node.test)


def block_start(node, lines: List[str], offset: int = 0) -> int:
    """The 1-based line a definition's block starts on: its decorators and the comment lines right
    above it (no blank line between) belong to it."""
    start = first_line(node) + offset
    while start > 1 and lines[start - 2].strip().startswith("#"):
        start -= 1
    return start


def reorder(text: str, random_generator: random.Random) -> str:
    """The top-level functions and classes of a file in a shuffled order, and the methods of each
    class (after __init__). The text before the first definition and a closing
    `if __name__ == "__main__":` block stay where they are. A file with other statements between
    its definitions is left as it is."""
    lines = text.split("\n")
    body = ast.parse(text).body
    first = next((index for index, node in enumerate(body) if isinstance(node, (ast.FunctionDef, ast.ClassDef))), None)
    if first is None:
        return text
    definitions = body[first:]
    guard = definitions.pop() if is_main_guard(definitions[-1]) else None
    if any(not isinstance(node, (ast.FunctionDef, ast.ClassDef)) for node in definitions):
        return text
    prefix = lines[:block_start(definitions[0], lines) - 1]
    blocks = []
    for node in definitions:
        block = lines[block_start(node, lines) - 1:node.end_lineno]
        if isinstance(node, ast.ClassDef):
            block = reorder_methods(node, block, block_start(node, lines), random_generator)
        blocks.append(block)
    random_generator.shuffle(blocks)
    parts = [strip_blank(prefix)] + blocks + ([lines[block_start(guard, lines) - 1:guard.end_lineno]] if guard else [])
    joined = []
    for part in parts:
        if joined:
            joined += ["", ""]
        joined += strip_blank(part)
    return "\n".join(joined) + "\n"


def reorder_methods(node: ast.ClassDef, block: List[str], block_first_line: int, random_generator: random.Random) -> List[str]:
    """A class block with its methods after __init__ shuffled (left as it is when anything but
    methods follows the first of them)."""
    members = node.body
    methods = [index for index, member in enumerate(members) if isinstance(member, ast.FunctionDef)]
    if len(methods) < 2:
        return block
    keep = 1 if members[methods[0]].name == "__init__" else 0
    movable = members[methods[0] + keep:]
    if len(movable) < 2 or any(not isinstance(member, ast.FunctionDef) for member in movable):
        return block
    offset = 1 - block_first_line   # file line -> block line (1-based)
    head = block[:block_start(movable[0], block, offset) - 1]
    pieces = [block[block_start(member, block, offset) - 1:member.end_lineno + offset] for member in movable]
    random_generator.shuffle(pieces)
    result = strip_blank(head)
    for piece in pieces:
        result += [""] + strip_blank(piece)
    return result


def strip_blank(lines: List[str]) -> List[str]:
    start, end = 0, len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


def check_version(base: Dict, version: Version) -> None:
    """A version must parse, define the same things under the new names, and contain no
    renamed name in its code any more."""
    base_files = {name: text.splitlines() for name, text in base["files"].items()}
    expected = {version.place(f"{file_name}:{name}") for file_name, name in symbol_lines(base_files)}
    found = set(symbol_lines(version.files))
    if expected != found:
        raise ValueError(f"{version.id}: definitions changed: missing {sorted(expected - found)[:5]}, "
                         f"new {sorted(found - expected)[:5]}")
    left = identifiers(version.files) & set(version.renames)
    if left:
        raise ValueError(f"{version.id}: names not renamed: {sorted(left)}")


# --- what the model sees --------------------------------------------------------------------------

def render(files: Dict[str, List[str]], ranges: Optional[Dict[str, List[Tuple[int, int]]]] = None) -> str:
    """Files with numbered lines; with `ranges` ({file: [(start, end)]}) only those lines."""
    blocks = []
    for file_name, lines in files.items():
        if ranges is not None and file_name not in ranges:
            continue
        shown = [(number, line) for number, line in enumerate(lines, 1)
                 if ranges is None or any(start <= number <= end for start, end in ranges[file_name])]
        body = "\n".join(f"{number:>2}: {line}".rstrip() for number, line in shown)
        blocks.append(f'<file name="{file_name}">\n{body}\n</file>')
    return "\n".join(blocks)


def project_context(version: Version, message: str) -> str:
    """Step 1: the whole project and the user's message."""
    return f"<project>\n{render(version.files)}\n</project>\n<message>{message}</message>"


def scoped_context(version: Version, ranges: Dict[str, List[Tuple[int, int]]], message: str) -> str:
    """Step 2: only the scoped lines and the user's message."""
    return f"<context>\n{render(version.files, ranges)}\n</context>\n<message>{message}</message>"
