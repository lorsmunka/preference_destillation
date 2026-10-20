"""Showing the sheet in the terminal."""

from config import COLUMN_WIDTH, COLUMNS, DECIMALS, ERROR_CODES
from references import cell_name


def format_value(value):
    """12.0 -> "12", 1234.5 -> "1,234.50", None -> ""; text and errors stay as they are."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:,.0f}" if value.is_integer() else f"{value:,.{DECIMALS}f}"
    return value


def cell_text(value):
    """A value padded to COLUMN_WIDTH. A number too wide becomes ####, a long text is cut."""
    text, is_number = format_value(value), isinstance(value, float) or value in ERROR_CODES
    if len(text) > COLUMN_WIDTH:
        return "#" * COLUMN_WIDTH if is_number else text[:COLUMN_WIDTH]
    return text.rjust(COLUMN_WIDTH) if is_number else text.ljust(COLUMN_WIDTH)


def print_grid(sheet):
    """The used part of the sheet and one more row and column, with letters on top."""
    used_columns, used_rows = sheet.used_size()
    columns = range(min(used_columns + 1, COLUMNS))
    print("   |" + "|".join(chr(ord("A") + column).center(COLUMN_WIDTH) for column in columns))
    for row in range(used_rows + 1):
        values = [cell_text(sheet.cell_at(cell_name(column, row)).value) for column in columns]
        print(f"{row + 1:>3}|" + "|".join(values))
