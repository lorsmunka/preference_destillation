"""The board: a grid of squares, each empty (None) or holding a Piece."""

from copy import deepcopy

from config import BLACK, BOARD_SIZE, RED
from pieces import Piece, king_row
from squares import is_dark

START_ROWS = 3   # rows of men each player starts with


class Board:
    def __init__(self):
        self.grid = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]

    def set_up(self):
        """Men on the dark squares: black at the top, red at the bottom."""
        for row in range(BOARD_SIZE):
            color = BLACK if row < START_ROWS else RED if row >= BOARD_SIZE - START_ROWS else None
            for column in range(BOARD_SIZE):
                if color and is_dark(row, column):
                    self.grid[row][column] = Piece(color)

    def piece_at(self, square):
        return self.grid[square[0]][square[1]]

    def lift_piece(self, square):
        piece = self.piece_at(square)
        self.grid[square[0]][square[1]] = None
        return piece

    def squares_of(self, color):
        return [(row, column) for row, cells in enumerate(self.grid)
                for column, piece in enumerate(cells) if piece and piece.color == color]

    def apply_move(self, move):
        """Move the piece, take off the jumped ones, crown a man on the far row."""
        piece = self.lift_piece(move.path[0])
        for square in move.captured:
            self.lift_piece(square)
        row, column = move.path[-1]
        if row == king_row(piece.color):
            piece.crown()
        self.grid[row][column] = piece

    def clone(self):
        """A copy with its own pieces, to try moves on."""
        copied = Board()
        copied.grid = deepcopy(self.grid)
        return copied
