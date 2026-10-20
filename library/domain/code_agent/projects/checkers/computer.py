"""The computer player: minimax with alpha-beta pruning."""

import random

from config import SEARCH_DEPTH
from evaluation import evaluate
from pieces import opponent
from rules import legal_moves

WIN_SCORE = 1000


class ComputerPlayer:
    def __init__(self, color, depth=SEARCH_DEPTH):
        self.color = color
        self.depth = depth

    def choose_move(self, board):
        """The move with the best minimax score (a random one if tied)."""
        scored = [(self.minimax(after(board, move), self.depth - 1, opponent(self.color)), move)
                  for move in legal_moves(board, self.color)]
        best = max(score for score, _ in scored)
        return random.choice([move for score, move in scored if score == best])

    def minimax(self, board, depth, turn, alpha=-WIN_SCORE, beta=WIN_SCORE):
        """The computer's score if both sides play their best for `depth` moves.
        Alpha-beta: stop once a move cannot change the result."""
        options = legal_moves(board, turn)
        if not options:
            return -WIN_SCORE if turn == self.color else WIN_SCORE
        if depth == 0:
            return evaluate(board, self.color)
        maximizing = turn == self.color
        best = -WIN_SCORE if maximizing else WIN_SCORE
        for move in options:
            score = self.minimax(after(board, move), depth - 1, opponent(turn), alpha, beta)
            if maximizing:
                best, alpha = max(best, score), max(alpha, score)
            else:
                best, beta = min(best, score), min(beta, score)
            if alpha >= beta:
                break
        return best


def after(board, move):
    """The board after a move, made on a copy."""
    copied = board.clone()
    copied.apply_move(move)
    return copied
