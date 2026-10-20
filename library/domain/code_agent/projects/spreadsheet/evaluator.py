"""Working out the number a formula tree stands for."""

from operator import add, mul, sub, truediv
from statistics import mean

from cells import FormulaError
from config import DIVIDE_BY_ZERO, ERROR_CODES, NOT_A_NUMBER, UNKNOWN_FUNCTION
from references import expand_range

OPERATORS = {"+": add, "-": sub, "*": mul, "/": truediv}
FUNCTIONS = {"SUM": sum, "AVG": mean, "MIN": min, "MAX": max}


def evaluate(node, sheet):
    """The value of a tree; the cells it names are looked up in the sheet."""
    kind, value, children = node
    if kind == "number":
        return value
    if kind == "reference":
        return sheet.cell_at(value).number()
    if kind == "range":
        raise FormulaError(NOT_A_NUMBER)   # a range only works inside a function
    if kind == "call":
        return call_function(value, children, sheet)
    left, right = evaluate(children[0], sheet), evaluate(children[1], sheet)
    if value == "/" and right == 0:
        raise FormulaError(DIVIDE_BY_ZERO)
    return OPERATORS[value](left, right)


def call_function(name, arguments, sheet):
    """SUM, AVG, MIN or MAX of the arguments; a range gives all its numbers."""
    if name not in FUNCTIONS:
        raise FormulaError(UNKNOWN_FUNCTION)
    numbers = []
    for argument in arguments:
        numbers += range_numbers(argument[1], sheet) if argument[0] == "range" else [evaluate(argument, sheet)]
    return float(FUNCTIONS[name](numbers or [0.0]))   # no numbers count as 0


def range_numbers(text, sheet):
    """The numbers of a range like A1:B3. Empty and text cells are left out; an error spreads."""
    cells = [sheet.cell_at(name) for name in expand_range(text)]
    return [cell.number() for cell in cells if isinstance(cell.value, float) or cell.value in ERROR_CODES]
