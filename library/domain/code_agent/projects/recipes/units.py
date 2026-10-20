"""Turning amounts into one unit so they can be added up."""

TO_GRAMS = {"g": 1, "kg": 1000}
TO_MILLILITRES = {"ml": 1, "l": 1000, "cup": 240, "tbsp": 15, "tsp": 5}


def to_base(amount, unit):
    """(1.5, "kg") -> (1500, "g"); (2, "cup") -> (480, "ml"); other units stay as they are."""
    if unit in TO_GRAMS:
        return amount * TO_GRAMS[unit], "g"
    if unit in TO_MILLILITRES:
        return amount * TO_MILLILITRES[unit], "ml"
    return amount, unit


def nice_amount(amount, unit):
    """1500 g -> "1.5 kg"; 250 ml -> "250 ml"; 3 eggs -> "3 eggs"."""
    if unit == "g" and amount >= 1000:
        return f"{amount / 1000:g} kg"
    if unit == "ml" and amount >= 1000:
        return f"{amount / 1000:g} l"
    return f"{amount:g} {unit}"


def same_kind(first_unit, second_unit):
    """True when the two units can be added (both weights, both volumes, or the same unit)."""
    return to_base(1, first_unit)[1] == to_base(1, second_unit)[1]
