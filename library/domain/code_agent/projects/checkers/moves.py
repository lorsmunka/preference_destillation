"""A Move: the squares a piece passes through, and those it jumps."""

from dataclasses import dataclass, field

from squares import square_name


@dataclass
class Move:
    path: list
    captured: list = field(default_factory=list)

    def is_jump(self):
        return bool(self.captured)

    def to_text(self):
        """"c3-d4" for a step, "c3xe5xg7" for a chain of jumps."""
        separator = "x" if self.is_jump() else "-"
        return separator.join(square_name(square) for square in self.path)
