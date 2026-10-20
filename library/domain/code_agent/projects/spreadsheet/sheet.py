"""The Sheet: every cell, the graph of which cell reads which, and typing into a cell."""

from cells import Cell, FormulaError, literal_value
from config import CYCLE_ERROR
from dependencies import DependencyGraph, references_in
from evaluator import evaluate
from formulas import parse_formula
from references import parse_reference


class Sheet:
    def __init__(self):
        self.cells = {}   # "A1" -> Cell, only for cells typed into
        self.graph = DependencyGraph()

    def cell_at(self, name):
        parse_reference(name)   # FormulaError for a name outside the sheet
        return self.cells.get(name, Cell())

    def set_cell(self, name, text):
        """Type text into a cell and work out again what depends on it. Returns the old text."""
        name = name.upper()
        old_text = self.cell_at(name).text
        self.cells[name] = Cell(text.strip())
        for changed in [name] + self.graph.recalculation_order(name):
            self.recalculate(changed)
        return old_text

    def recalculate(self, name):
        """Work out one cell's value again; for a formula, also what the cell reads now."""
        cell = self.cells[name]
        self.graph.reads[name] = set()
        if not cell.is_formula:
            cell.value = literal_value(cell.text)
            return
        try:
            tree = parse_formula(cell.text)
            self.graph.reads[name] = references_in(tree)
            cell.value = CYCLE_ERROR if self.graph.in_cycle(name) else evaluate(tree, self)
        except FormulaError as error:
            cell.value = error.code

    def used_size(self):
        """(columns, rows) of the block from A1 that holds every filled cell."""
        filled = [parse_reference(name) for name, cell in self.cells.items() if cell.text] or [(-1, -1)]
        return max(column for column, _ in filled) + 1, max(row for _, row in filled) + 1
