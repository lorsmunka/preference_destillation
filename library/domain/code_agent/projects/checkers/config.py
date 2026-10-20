"""Settings for the checkers game."""

BOARD_SIZE = 8
RED = "red"
BLACK = "black"

SEARCH_DEPTH = 4   # moves the computer looks ahead
MAN_VALUE = 1.0
KING_VALUE = 1.6
ADVANCE_BONUS = 0.05   # per row a man has moved forward
CENTER_BONUS = 0.1

DRAW_AFTER_MOVES = 40   # moves in a row without a jump
SAVE_FILE = "saved_game.json"
