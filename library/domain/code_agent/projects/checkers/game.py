"""A Game: the board, whose turn it is, and the moves so far."""

from board import Board
from config import DRAW_AFTER_MOVES, RED
from pieces import opponent
from rules import legal_moves


class Game:
    def __init__(self):
        self.board = Board()
        self.board.set_up()
        self.turn = RED
        self.quiet_moves = 0   # moves in a row without a jump
        self.history = []   # (move, board before, quiet moves before)

    def possible_moves(self):
        return legal_moves(self.board, self.turn)

    def play(self, move):
        """Make a legal move and give the turn to the other player."""
        self.history.append((move, self.board.clone(), self.quiet_moves))
        self.board.apply_move(move)
        self.quiet_moves = 0 if move.is_jump() else self.quiet_moves + 1
        self.turn = opponent(self.turn)

    def undo(self):
        """Take back the last move. False when there is none."""
        if not self.history:
            return False
        _, self.board, self.quiet_moves = self.history.pop()
        self.turn = opponent(self.turn)
        return True

    def winner(self):
        """A player with no moves (or no pieces) left has lost."""
        return None if self.possible_moves() else opponent(self.turn)

    def is_draw(self):
        return self.quiet_moves >= DRAW_AFTER_MOVES

    def is_over(self):
        return self.winner() is not None or self.is_draw()
