"""Things the player can pick up, and ItemBag: what holds them (a room's floor, the player's bag)."""

from dataclasses import dataclass

# name -> (kind, weight, value): a weapon's damage, the HP food heals or a treasure's points
ITEM_DATA = {"sword": ("weapon", 4, 5), "knife": ("weapon", 2, 3), "bread": ("food", 1, 8),
             "potion": ("food", 1, 20), "brass key": ("key", 1, 0), "silver key": ("key", 1, 0),
             "coin": ("treasure", 1, 10), "crown": ("treasure", 3, 40)}


@dataclass
class Item:
    name: str
    kind: str
    weight: int
    value: int


class ItemBag:
    def __init__(self, items):
        self.items = items

    def find_item(self, name):
        return next((item for item in self.items if item.name == name), None)

    def remove_item(self, name):
        item = self.find_item(name)
        if item is not None:
            self.items.remove(item)
        return item


def make_item(name):
    return Item(name, *ITEM_DATA[name])
