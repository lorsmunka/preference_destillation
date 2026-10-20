"""What is in the kitchen cupboard."""

from storage import load_pantry, save_pantry
from units import to_base


def stock_of(ingredient):
    """The amount at home, in its base unit, e.g. (500, "g"); (0, None) when there is none."""
    amount, unit = load_pantry().get(ingredient, (0, None))
    return (amount, unit) if unit is None else to_base(amount, unit)


def add_to_pantry(ingredient, amount, unit):
    pantry = load_pantry()
    base_amount, base_unit = to_base(amount, unit)
    have, _ = stock_of(ingredient)
    pantry[ingredient] = (have + base_amount, base_unit)
    save_pantry(pantry)


def use_ingredient(ingredient, amount, unit):
    """Take an amount out; never below zero. What was really used."""
    pantry = load_pantry()
    have, base_unit = stock_of(ingredient)
    needed, _ = to_base(amount, unit)
    used = min(have, needed)
    pantry[ingredient] = (have - used, base_unit)
    save_pantry(pantry)
    return used


def missing_ingredients(needed):
    """From {ingredient: (amount, unit)}, what there is not enough of, and how much more is needed."""
    missing = {}
    for ingredient, (amount, unit) in needed.items():
        want, base_unit = to_base(amount, unit)
        have, _ = stock_of(ingredient)
        if have < want:
            missing[ingredient] = (want - have, base_unit)
    return missing
