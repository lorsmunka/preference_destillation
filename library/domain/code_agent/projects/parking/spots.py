"""The parking spots: floor and number, and which car is in each."""

from config import FLOORS, SPOTS_PER_FLOOR
from storage import load_spots, save_spots


def spot_name(floor, number):
    """Floor 2, spot 7 -> "2-07"."""
    return f"{floor}-{number:02d}"


def find_free_spot(kind="car"):
    """The lowest free spot. Motorbikes park in the last five spots of a floor. None when full."""
    taken = load_spots()
    for floor in range(1, FLOORS + 1):
        numbers = range(SPOTS_PER_FLOOR - 4, SPOTS_PER_FLOOR + 1) if kind == "motorbike" else range(1, SPOTS_PER_FLOOR - 4)
        for number in numbers:
            if spot_name(floor, number) not in taken:
                return spot_name(floor, number)
    return None


def park(spot, plate):
    spots = load_spots()
    spots[spot] = plate
    save_spots(spots)


def leave(spot):
    spots = load_spots()
    spots.pop(spot, None)
    save_spots(spots)


def free_spots():
    return FLOORS * SPOTS_PER_FLOOR - len(load_spots())
