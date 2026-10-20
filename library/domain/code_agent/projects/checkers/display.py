"""Printing the board and the list of moves."""

from config import BOARD_SIZE
from squares import COLUMNS, is_dark


def board_lines(board):
    """One line per row, row 8 on top, the letters below."""
    lines = []
    for row in range(BOARD_SIZE):
        cells = []
        for column in range(BOARD_SIZE):
            piece = board.piece_at((row, column))
            cells.append(piece.symbol if piece else "." if is_dark(row, column) else " ")
        lines.append(f"{BOARD_SIZE - row} " + " ".join(cells))
    lines.append("  " + " ".join(COLUMNS))
    return lines


def print_board(board):
    print("\n" + "\n".join(board_lines(board)))


def print_history(game):
    """The moves so far, two per line, like a score sheet."""
    texts = [move.to_text() for move, _, _ in game.history]
    for index in range(0, len(texts), 2):
        print(f"{index // 2 + 1}. " + "  ".join(texts[index:index + 2]))
