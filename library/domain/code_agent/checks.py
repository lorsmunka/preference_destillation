"""Reading and checking the coding agent's replies: the scope (which files and lines matter) and
the explanation (it must not name things the project does not have)."""

import re
from typing import Dict, List, Optional, Set, Tuple

Ranges = Dict[str, List[Tuple[int, int]]]
SCOPE_LINE = re.compile(r"([\w./-]+\.py)\s+(\d+)(?:\s*-\s*(\d+))?")


def parse_scope(text: str) -> Ranges:
    """{file: [(start, end)]} from a <scope> block, one "file.py start-end" (or "file.py line")
    per line; a path keeps only its file name."""
    body = text.split("<scope>")[-1].split("</scope>")[0]
    ranges: Ranges = {}
    for match in SCOPE_LINE.finditer(body):
        start = int(match.group(2))
        ranges.setdefault(match.group(1).split("/")[-1], []).append((start, int(match.group(3) or start)))
    return ranges


def check_scope(text: str, files: Dict[str, List[str]]) -> Optional[str]:
    """None when the scope can be used as it is, otherwise what is wrong with it."""
    if "</scope>" not in text:
        return "cut off: no </scope>"
    ranges = parse_scope(text)
    if not ranges:
        return "no file ranges"
    for file_name, spans in ranges.items():
        if file_name not in files:
            return f"no file {file_name}"
        for start, end in spans:
            if start > end:
                return f"{file_name} {start}-{end}: start after end"
            if start < 1 or end > len(files[file_name]):
                return f"{file_name} {start}-{end}: outside the file's {len(files[file_name])} lines"
    return None


def usable_ranges(text: str, files: Dict[str, List[str]]) -> Ranges:
    """The scope's ranges that point into the project, clipped to the files (what the harness can
    show even when the scope is not fully right)."""
    ranges: Ranges = {}
    for file_name, spans in parse_scope(text).items():
        if file_name not in files:
            continue
        for start, end in spans:
            start, end = max(1, start), min(end, len(files[file_name]))
            if start <= end:
                ranges.setdefault(file_name, []).append((start, end))
    return ranges


def lines_of(ranges: Ranges) -> Set[Tuple[str, int]]:
    return {(file_name, number) for file_name, spans in ranges.items() for start, end in spans
            for number in range(start, end + 1)}


def canonical_scope(text: str) -> Optional[str]:
    """A scope as one comparable string: its lines merged into sorted ranges per file
    ("accounts.py 22-35; fees.py 7-11"). None when the text is not a scope."""
    if "<scope>" not in text:
        return None
    lines = sorted(lines_of(parse_scope(text)))
    parts, start, previous = [], None, None
    for file_name, number in lines:
        if previous and previous[0] == file_name and previous[1] == number - 1:
            previous = (file_name, number)
            continue
        if start:
            parts.append(f"{start[0]} {start[1]}-{previous[1]}")
        start = previous = (file_name, number)
    if start:
        parts.append(f"{start[0]} {start[1]}-{previous[1]}")
    return "; ".join(parts) if parts else "(empty)"


def score_scope(ranges: Ranges, truth: List[Tuple[str, str]], places: Dict[Tuple[str, str], Tuple[int, int]]) -> Dict:
    """A scope against the true places: the share of their lines it covers (recall), the share of
    its lines that are true ones (precision), and how many true places it covers at least half of."""
    entry_lines = [lines_of({file_name: [places[(file_name, name)]]}) for file_name, name in truth]
    true_lines = set().union(*entry_lines)
    chosen = lines_of(ranges)
    return {"recall": round(len(true_lines & chosen) / len(true_lines), 3),
            "precision": round(len(true_lines & chosen) / len(chosen), 3) if chosen else 0.0,
            "places_hit": sum(len(lines & chosen) * 2 >= len(lines) for lines in entry_lines),
            "places": len(truth)}


def made_up_names(explanation: str, names: Set[str]) -> List[str]:
    """Code-looking words of an explanation (in backticks, snake_case, CamelCase or followed by a
    bracket) that the project does not have."""
    candidates = set(re.findall(r"`([A-Za-z_][\w.]*)", explanation))
    candidates |= set(re.findall(r"\b([a-z]+_[a-z_]+|[A-Z][a-z]+[A-Z]\w*)\b", explanation))
    candidates |= set(re.findall(r"\b([A-Za-z_]\w*)\(", explanation))
    parts = {part for candidate in candidates for part in candidate.split(".") if part}
    return sorted(part for part in parts if part not in names and part not in ("py", "json", "self"))


def check_explanation(text: str, names: Set[str]) -> Optional[str]:
    """None for a finished explanation that names only real things."""
    if "</explanation>" not in text:
        return "cut off: no </explanation>"
    made_up = made_up_names(text, names)
    return f"names not in the project: {', '.join(made_up)}" if made_up else None
