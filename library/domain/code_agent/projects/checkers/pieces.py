"""A checkers piece: a man moves forward only, a king moves both ways."""

from dataclasses import dataclass

from config import BLACK, BOARD_SIZE, RED


@dataclass
class Piece:
    color: str
    king: bool = False

    @property
    def symbol(self):
        """r or b for a man, R or B for a king."""
        letter = "r" if self.color == RED else "b"
        return letter.upper() if self.king else letter

    def directions(self):
        """(row, column) steps: forward only, unless it is a king."""
        forward = -1 if self.color == RED else 1
        rows = [forward, -forward] if self.king else [forward]
        return [(row_step, column_step) for row_step in rows for column_step in (-1, 1)]

    def crown(self):
        self.king = True


def opponent(color):
    return BLACK if color == RED else RED


def king_row(color):
    """The far row, where a man is crowned (red moves up)."""
    return 0 if color == RED else BOARD_SIZE - 1
