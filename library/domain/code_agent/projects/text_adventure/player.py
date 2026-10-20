"""The Player: hit points, where they are, their score and the Inventory they carry."""

from config import MAX_CARRY_WEIGHT, PLAYER_BASE_DAMAGE, PLAYER_MAX_HP, START_ROOM
from items import ItemBag
from score import Score


class Inventory(ItemBag):
    def total_weight(self):
        return sum(item.weight for item in self.items)

    def store(self, item):
        """Put an item in the bag. False when it would make the bag too heavy."""
        if self.total_weight() + item.weight > MAX_CARRY_WEIGHT:
            return False
        self.items.append(item)
        return True

    def best_weapon(self):
        weapons = [item for item in self.items if item.kind == "weapon"]
        return max(weapons, key=lambda item: item.value, default=None)


class Player:
    def __init__(self):
        self.hp = PLAYER_MAX_HP
        self.room_name = START_ROOM
        self.previous_room = None
        self.inventory = Inventory([])
        self.score = Score(counted=[START_ROOM])

    def is_alive(self):
        return self.hp > 0

    def take_damage(self, amount):
        self.hp = max(0, self.hp - amount)

    def heal(self, amount):
        self.hp = min(PLAYER_MAX_HP, self.hp + amount)

    def attack_damage(self):
        weapon = self.inventory.best_weapon()
        return PLAYER_BASE_DAMAGE + (weapon.value if weapon else 0)

    def bag_text(self):
        names = ", ".join(item.name for item in self.inventory.items) or "nothing"
        return f"You carry {names} (weight {self.inventory.total_weight()})."
