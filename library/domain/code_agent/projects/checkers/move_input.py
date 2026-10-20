"""The player's move, typed like "c3-d4" or "c3xe5xg7" (jumps), or a command."""

from squares import parse_squares

COMMANDS = ("undo", "history", "save", "quit")


def check_typed_move(game, text):
    """(the legal move the text means, None) or (None, the problem)."""
    path = parse_squares(text)
    if path is None or len(path) < 2:
        return None, "Type a move like c3-d4, or c3xe5 for a jump."
    piece = game.board.piece_at(path[0])
    if piece is None or piece.color != game.turn:
        return None, f"There is no {game.turn} piece on that square."
    legal = game.possible_moves()
    for move in legal:
        if move.path == path:
            return move, None
    if any(move.path[:len(path)] == path for move in legal):
        return None, "Keep going: the chain of jumps is not finished."
    if any(move.is_jump() for move in legal):
        return None, "You have to jump when you can."
    return None, "That piece cannot move there."


def ask_move(game):
    """Ask until the player types a legal move or a command."""
    while True:
        text = input(f"{game.turn} to move: ").strip().lower()
        if text in COMMANDS:
            return text
        move, problem = check_typed_move(game, text)
        if move is not None:
            return move
        print(problem)
