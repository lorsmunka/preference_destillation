"""Cell names like B3 (a column letter and a row number) and ranges like A1:B3."""

import re

from cells import FormulaError
from config import BAD_REFERENCE, COLUMNS, ROWS

REFERENCE_PATTERN = re.compile(r"([A-Z])([0-9]+)")


def parse_reference(name):
    """"B3" -> (1, 2): column and row, counted from 0. FormulaError for a cell outside the sheet."""
    match = REFERENCE_PATTERN.fullmatch(name.upper())
    if match is None:
        raise FormulaError(BAD_REFERENCE, f"{name} is not a cell name")
    column, row = ord(match.group(1)) - ord("A"), int(match.group(2)) - 1
    if column >= COLUMNS or not 0 <= row < ROWS:
        raise FormulaError(BAD_REFERENCE, f"{name} is outside the sheet")
    return column, row


def cell_name(column, row):
    return chr(ord("A") + column) + str(row + 1)


def expand_range(text):
    """"A1:B2" -> ["A1", "B1", "A2", "B2"]: every cell of the rectangle, row by row."""
    start, end = text.split(":")
    (first_column, first_row), (last_column, last_row) = parse_reference(start), parse_reference(end)
    return [cell_name(column, row) for row in range(first_row, last_row + 1)
            for column in range(first_column, last_column + 1)]
