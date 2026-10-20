"""Saving the sheet to a CSV file and reading it back; formulas are saved as typed."""

import csv

from config import COLUMNS, ROWS
from references import cell_name


def import_csv(sheet, path):
    """Type the file into the sheet from A1 on. The (cell, text before) changes: one undo step."""
    changes = []
    with open(path, newline="") as file:
        for row, values in enumerate(list(csv.reader(file))[:ROWS]):
            for column, text in enumerate(values[:COLUMNS]):
                name = cell_name(column, row)
                changes.append((name, sheet.set_cell(name, text)))
    return changes


def export_csv(sheet, path):
    """Write what was typed into the used part of the sheet. The number of rows written."""
    columns, rows = sheet.used_size()
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        for row in range(rows):
            writer.writerow([sheet.cell_at(cell_name(column, row)).text for column in range(columns)])
    return rows
