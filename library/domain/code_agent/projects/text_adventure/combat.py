"""Fighting the monster of a room, one round at a time."""

from items import make_item


def attack(game, noun):
    """One round of combat: the player's blow, then the monster's if it still stands."""
    room, player = game.current_room(), game.player
    if not room.is_guarded():
        return "There is nothing to fight."
    monster = room.monster
    damage = player.attack_damage()
    monster.wound(damage)
    text = f"You hit the {monster.name} for {damage}."
    if not monster.is_alive():
        return text + "\n" + beat_monster(room, player)
    damage = monster.strike()
    player.take_damage(damage)
    return text + f"\nThe {monster.name} hits you for {damage}."


def beat_monster(room, player):
    """The monster falls: it drops its loot and the player gets its points."""
    monster = room.monster
    room.monster = None
    room.items.append(make_item(monster.loot))
    player.score.award(monster.name, monster.reward)
    return f"The {monster.name} dies and drops a {monster.loot}."
