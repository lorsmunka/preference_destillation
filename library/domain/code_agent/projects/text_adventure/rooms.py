"""A Room of the map: its exits, the items on its floor and maybe a monster."""

from items import ItemBag


class Room(ItemBag):
    def __init__(self, name, exits, items, monster=None):
        super().__init__(items)
        self.name = name
        self.exits = exits      # direction -> name of the room it leads to
        self.monster = monster
        self.locked = {}        # direction -> name of the key that opens that door

    def describe(self):
        """What the player sees: the room, its items, its monster and its exits."""
        lines = [f"You are in the {self.name}."]
        if self.items:
            lines.append("You see: " + ", ".join(item.name for item in self.items))
        if self.is_guarded():
            lines.append(f"A {self.monster.name} is watching you!")
        exits = [way + (" (locked)" if way in self.locked else "") for way in self.exits]
        return "\n".join(lines + ["Exits: " + ", ".join(exits)])

    def is_guarded(self):
        return self.monster is not None and self.monster.is_alive()
