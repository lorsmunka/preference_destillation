"""Moving between rooms: exits, locked doors and the monsters that block the way."""

from config import POINTS_FOR_NEW_ROOM


def go(game, direction):
    """Walk through an exit unless its door is locked; a monster only lets the player go back."""
    room, player = game.current_room(), game.player
    if direction not in room.exits:
        return "You can't go that way."
    if room.is_guarded() and room.exits[direction] != player.previous_room:
        return f"The {room.monster.name} blocks the way!"
    if direction in room.locked:
        return f"The door is locked. It needs the {room.locked[direction]}."
    player.previous_room = player.room_name
    player.room_name = room.exits[direction]
    player.score.award(player.room_name, POINTS_FOR_NEW_ROOM)
    return game.current_room().describe()


def unlock(room, key_name):
    """Open the door this key fits; it stays open. The door's direction, or None."""
    for direction, needed in room.locked.items():
        if needed == key_name:
            del room.locked[direction]
            return direction
    return None
