"""A Cell keeps what was typed and its value; FormulaError is a formula going wrong."""

from dataclasses import dataclass

from config import ERROR_CODES, NOT_A_NUMBER


class FormulaError(Exception):
    def __init__(self, code, detail=""):
        super().__init__(detail or code)
        self.code = code   # what the cell shows, like "#DIV/0!"


@dataclass
class Cell:
    text: str = ""        # what was typed: "12", "rent" or "=A1*2"
    value: object = None  # a number, a text, an error code, or None when empty

    @property
    def is_formula(self):
        return self.text.startswith("=")

    def number(self):
        """The value as a formula uses it: empty is 0, an error spreads, text is refused."""
        if self.value is None:
            return 0.0
        if self.value in ERROR_CODES:
            raise FormulaError(self.value)
        if isinstance(self.value, str):
            raise FormulaError(NOT_A_NUMBER)
        return self.value


def literal_value(text):
    try:
        return float(text)
    except ValueError:
        return text or None
