"""The map: the rooms, how they connect, what lies in them and which doors are locked."""

from items import make_item
from monsters import Monster
from rooms import Room

# name -> (exits, items, monster as (name, hp, damage, points, loot) or None)
ROOM_DATA = {
    "cellar": ({"up": "hall"}, ["sword"], None),
    "hall": ({"down": "cellar", "north": "library", "east": "kitchen"}, [], None),
    "kitchen": ({"west": "hall"}, ["bread", "knife", "coin"], ("rat", 8, 3, 15, "brass key")),
    "library": ({"south": "hall", "up": "tower"}, ["crown", "potion"], ("ghost", 18, 6, 40, "silver key")),
    "tower": ({"down": "library"}, [], None),
}
LOCKED_EXITS = {("hall", "north"): "brass key", ("library", "up"): "silver key"}


def build_world():
    """The rooms of ROOM_DATA, with the doors of LOCKED_EXITS locked."""
    rooms = {}
    for name, (exits, item_names, monster) in ROOM_DATA.items():
        items = [make_item(item_name) for item_name in item_names]
        rooms[name] = Room(name, dict(exits), items, Monster(*monster) if monster else None)
    for (room_name, direction), key_name in LOCKED_EXITS.items():
        rooms[room_name].locked[direction] = key_name
    return rooms
