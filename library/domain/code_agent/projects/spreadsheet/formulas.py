"""Reading a formula: splitting it into tokens, then building its tree with the Parser."""

import re

from cells import FormulaError
from config import SYNTAX_ERROR
from parser import Parser

# one named group per kind of token, tried in this order (a range before a single cell)
TOKEN_PATTERN = re.compile(r"\s*(?:(?P<number>\d+(?:\.\d+)?)|(?P<range>[A-Z]\d+:[A-Z]\d+)"
                           r"|(?P<reference>[A-Z]\d+)|(?P<name>[A-Z]+)|(?P<symbol>[-+*/(),]))")


def tokenize(formula):
    """(kind, text) tokens: "=A1*2" -> ("reference", "A1"), ("symbol", "*"), ("number", "2")."""
    text, position, tokens = formula.lstrip("=").upper().rstrip(), 0, []
    while position < len(text):
        match = TOKEN_PATTERN.match(text, position)
        if match is None:
            raise FormulaError(SYNTAX_ERROR)
        tokens.append((match.lastgroup, match.group(match.lastgroup)))
        position = match.end()
    return tokens


def parse_formula(formula):
    """The tree of a whole formula like "=A1*(B2+3)"."""
    parser = Parser(tokenize(formula))
    tree = parser.expression()
    if parser.peek():
        raise FormulaError(SYNTAX_ERROR)   # something is left over
    return tree
