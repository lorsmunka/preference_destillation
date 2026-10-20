"""The simple commands: look, take, drop, use and inventory."""

from movement import unlock


def look(game, noun):
    return game.current_room().describe()


def take(game, noun):
    """Pick an item up, unless a monster guards the room or the bag gets too heavy."""
    room, player = game.current_room(), game.player
    item = room.find_item(noun)
    if item is None:
        return f"There is no {noun} here."
    if room.is_guarded():
        return f"The {room.monster.name} won't let you."
    if not player.inventory.store(item):
        return f"The {noun} is too heavy."
    room.remove_item(noun)
    if item.kind == "treasure":
        player.score.award(item.name, item.value)
    return "Taken."


def drop(game, noun):
    item = game.player.inventory.remove_item(noun)
    if item is None:
        return f"You have no {noun}."
    game.current_room().items.append(item)
    return "Dropped."


def use(game, noun):
    """Food heals and is gone; a key opens the door it fits in this room."""
    player = game.player
    item = player.inventory.find_item(noun)
    if item is None:
        return f"You have no {noun}."
    if item.kind == "food":
        player.inventory.remove_item(noun)
        player.heal(item.value)
        return f"You feel better: {player.hp} HP."
    if item.kind == "key":
        direction = unlock(game.current_room(), noun)
        return f"The door {direction} opens." if direction else "No door here fits it."
    return "Nothing happens."


def show_inventory(game, noun):
    return game.player.bag_text()
