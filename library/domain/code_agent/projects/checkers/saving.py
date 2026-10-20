"""A game is saved as its moves; loading plays them again."""

import json
import os

from computer import ComputerPlayer
from config import BLACK, RED, SAVE_FILE
from game import Game
from move_input import check_typed_move


def save_game(game, computer, path=SAVE_FILE):
    moves = [move.to_text() for move, _, _ in game.history]
    with open(path, "w") as file:
        json.dump({"computer": computer.color, "moves": moves}, file)


def load_game(path=SAVE_FILE):
    """(the Game, the computer), or None without a save that replays."""
    if not os.path.exists(path):
        return None
    with open(path) as file:
        data = json.load(file)
    game = Game()
    for text in data["moves"]:
        move, _ = check_typed_move(game, text)
        if move is None:
            return None
        game.play(move)
    return game, ComputerPlayer(data["computer"])


def start_game():
    """The saved game if the player wants it, else a new one."""
    wanted = input("Load the saved game? (y/n) ").strip().lower() == "y"
    saved = load_game() if wanted else None
    if saved is not None:
        return saved
    side = input("Play red or black? Red moves first. ").strip().lower()
    return Game(), ComputerPlayer(RED if side.startswith("b") else BLACK)
