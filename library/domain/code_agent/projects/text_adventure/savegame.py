"""Saving the game to a JSON file and loading it back."""

import json
import os

from config import SAVE_FILE
from items import make_item
from monsters import Monster
from player import Player
from score import Score


def save_game(game, noun):
    """Write the player and the state of every room to SAVE_FILE."""
    player = game.player
    data = {"player": [player.hp, player.room_name, player.previous_room, player.score.points,
                       sorted(player.score.counted), [item.name for item in player.inventory.items]],
            "rooms": {name: room_state(room) for name, room in game.rooms.items()}}
    with open(SAVE_FILE, "w") as file:
        json.dump(data, file)
    return "Your game is saved."


def room_state(room):
    monster = room.monster.state() if room.monster else None
    return [[item.name for item in room.items], room.locked, monster]


def load_game(game, noun):
    """Put the saved player and rooms back into the running game."""
    if not os.path.exists(SAVE_FILE):
        return "There is no saved game."
    with open(SAVE_FILE) as file:
        data = json.load(file)
    player = game.player = Player()
    player.hp, player.room_name, player.previous_room, points, counted, item_names = data["player"]
    player.score = Score(points, counted)
    player.inventory.items = [make_item(name) for name in item_names]
    for name, (item_names, locked, monster) in data["rooms"].items():
        room = game.rooms[name]
        room.items = [make_item(item_name) for item_name in item_names]
        room.locked = locked
        room.monster = Monster(*monster) if monster else None
    return "Your saved game is back."
