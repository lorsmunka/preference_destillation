"""The evaluation function: how good a position looks for one player."""

from config import ADVANCE_BONUS, BOARD_SIZE, CENTER_BONUS, KING_VALUE, MAN_VALUE, RED
from pieces import opponent


def piece_value(piece, square):
    """A king is worth more than a man; a man gains for each row it has
    advanced, and any piece in the middle a bit more."""
    row, column = square
    if piece.king:
        value = KING_VALUE
    else:
        rows_moved = BOARD_SIZE - 1 - row if piece.color == RED else row
        value = MAN_VALUE + ADVANCE_BONUS * rows_moved
    if 2 <= row <= 5 and 2 <= column <= 5:
        value += CENTER_BONUS
    return value


def material(board, color):
    return sum(piece_value(board.piece_at(square), square) for square in board.squares_of(color))


def evaluate(board, color):
    """Above zero when `color` is ahead."""
    return material(board, color) - material(board, opponent(color))
