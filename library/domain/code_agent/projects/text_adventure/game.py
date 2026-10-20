"""The Game holds the world and the player and runs one typed command at a time."""

from actions import drop, look, show_inventory, take, use
from combat import attack
from config import POINTS_FOR_WINNING, WINNING_ROOM
from movement import go
from parser import parse_command, show_help
from player import Player
from savegame import load_game, save_game
from world import build_world

COMMANDS = {"go": go, "take": take, "drop": drop, "use": use, "look": look, "inventory": show_inventory,
            "attack": attack, "save": save_game, "load": load_game, "help": show_help}


class Game:
    def __init__(self):
        self.player = Player()
        self.rooms = build_world()
        self.over = False

    def current_room(self):
        return self.rooms[self.player.room_name]

    def handle(self, text):
        """Run one typed command; returns what to print."""
        verb, noun = parse_command(text)
        if verb == "quit":
            self.over = True
            return "Goodbye!"
        if verb not in COMMANDS:
            return "I don't understand. Type help."
        return COMMANDS[verb](self, noun) + self.ending()

    def ending(self):
        """Over when the player dies or reaches WINNING_ROOM."""
        if not self.player.is_alive():
            self.over = True
            return "\nYou have died."
        if self.player.room_name == WINNING_ROOM:
            self.over = True
            self.player.score.award("victory", POINTS_FOR_WINNING)
            return "\nYou made it out. You win!"
        return ""
