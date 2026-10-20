"""Which moves are allowed: steps, jumps, chains of jumps."""

from moves import Move
from pieces import king_row
from squares import on_board


def step_moves(board, square):
    """Steps of one square diagonally onto an empty square."""
    steps = []
    for row_step, column_step in board.piece_at(square).directions():
        target = (square[0] + row_step, square[1] + column_step)
        if on_board(target) and board.piece_at(target) is None:
            steps.append(Move([square, target]))
    return steps


def jump_moves(board, square, piece, path=None, captured=None):
    """Every chain of jumps from `square`; a man crowned on the far row stops."""
    path = path or [square]
    captured = captured or []
    chains = []
    for row_step, column_step in piece.directions():
        middle = (square[0] + row_step, square[1] + column_step)
        landing = (square[0] + 2 * row_step, square[1] + 2 * column_step)
        if not on_board(landing) or middle in captured:
            continue
        jumped = board.piece_at(middle)
        if jumped is None or jumped.color == piece.color:
            continue
        if board.piece_at(landing) is not None and landing != path[0]:
            continue
        chain = Move(path + [landing], captured + [middle])
        if not piece.king and landing[0] == king_row(piece.color):
            chains.append(chain)
            continue
        longer = jump_moves(board, landing, piece, chain.path, chain.captured)
        chains.extend(longer or [chain])
    return chains


def legal_moves(board, color):
    """The player's moves; when a jump is possible, only the jumps."""
    jumps, steps = [], []
    for square in board.squares_of(color):
        jumps.extend(jump_moves(board, square, board.piece_at(square)))
        steps.extend(step_moves(board, square))
    return jumps or steps
