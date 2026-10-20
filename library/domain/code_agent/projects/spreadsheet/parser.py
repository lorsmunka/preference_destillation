"""The Parser turns tokens into a tree of (kind, value, children) nodes, * and / before + and -."""

from cells import FormulaError
from config import SYNTAX_ERROR


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens   # (kind, text) pairs
        self.position = 0

    def peek(self):
        return self.tokens[self.position][1] if self.position < len(self.tokens) else ""

    def take(self, expected=None):
        if self.position == len(self.tokens) or (expected and self.peek() != expected):
            raise FormulaError(SYNTAX_ERROR)
        self.position += 1
        return self.tokens[self.position - 1]

    def expression(self):
        node = self.term()
        while self.peek() in ("+", "-"):
            node = ("binary", self.take()[1], [node, self.term()])
        return node

    def term(self):
        node = self.factor()
        while self.peek() in ("*", "/"):
            node = ("binary", self.take()[1], [node, self.factor()])
        return node

    def factor(self):
        """A number, a cell, a range, (expression) or a function call like SUM(A1:A3, 2)."""
        kind, text = self.take()
        if text == "(":
            node = self.expression()
            self.take(")")
            return node
        if kind == "name":
            self.take("(")
            arguments = [self.expression()]
            while self.peek() == ",":
                self.take()
                arguments.append(self.expression())
            self.take(")")
            return ("call", text, arguments)
        if kind == "symbol":
            raise FormulaError(SYNTAX_ERROR)
        return (kind, float(text) if kind == "number" else text, [])
