"""A square is a (row, column) pair, named like "c3" on a real board."""

import re

from config import BOARD_SIZE

COLUMNS = "abcdefgh"
SQUARE_PATTERN = re.compile(r"([a-h])([1-8])")


def square_name(square):
    """(5, 2) -> "c3"."""
    row, column = square
    return f"{COLUMNS[column]}{BOARD_SIZE - row}"


def parse_square(text):
    """"c3" -> (5, 2), or None."""
    found = SQUARE_PATTERN.fullmatch(text.strip())
    if found is None:
        return None
    return BOARD_SIZE - int(found.group(2)), COLUMNS.index(found.group(1))


def parse_squares(text):
    """"c3xe5" -> [(5, 2), (3, 4)], or None."""
    path = [parse_square(part) for part in re.split(r"[-x]", text)]
    return None if None in path else path


def is_dark(row, column):
    return (row + column) % 2 == 1


def on_board(square):
    return 0 <= square[0] < BOARD_SIZE and 0 <= square[1] < BOARD_SIZE
