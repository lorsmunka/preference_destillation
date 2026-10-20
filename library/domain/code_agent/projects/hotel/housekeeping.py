"""Cleaning the rooms."""

from rooms import set_clean
from storage import load_rooms


def rooms_to_clean():
    """Room numbers that need cleaning, lowest floor first."""
    return sorted(number for number, data in load_rooms().items() if not data["clean"])


def mark_clean(number):
    set_clean(number, True)


def cleaning_list(cleaners):
    """Share the dirty rooms between the cleaners, one by one in turn."""
    plan = {cleaner: [] for cleaner in cleaners}
    for index, number in enumerate(rooms_to_clean()):
        plan[cleaners[index % len(cleaners)]].append(number)
    return plan
