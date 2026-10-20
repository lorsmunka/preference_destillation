"""The monsters that guard rooms: their hit points, how hard they hit and what they drop."""

import random


class Monster:
    def __init__(self, name, hp, damage, reward, loot):
        self.name = name
        self.hp = hp
        self.damage = damage
        self.reward = reward    # points for beating it
        self.loot = loot        # the name of the item it drops

    def is_alive(self):
        return self.hp > 0

    def strike(self):
        """This turn's damage, from 0 (a miss) up to the monster's damage."""
        return random.randint(0, self.damage)

    def wound(self, amount):
        self.hp = max(0, self.hp - amount)

    def state(self):
        return [self.name, self.hp, self.damage, self.reward, self.loot]
