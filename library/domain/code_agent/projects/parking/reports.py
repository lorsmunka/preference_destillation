"""Numbers for the garage owner."""

from clock import minutes_between, now, same_day
from config import FLOORS, SPOTS_PER_FLOOR
from pricing import parking_fee
from storage import load_spots, load_tickets


def occupancy():
    """The share of spots in use, from 0 to 1."""
    return len(load_spots()) / (FLOORS * SPOTS_PER_FLOOR)


def floor_counts():
    """How many vehicles are on each floor."""
    counts = {floor: 0 for floor in range(1, FLOORS + 1)}
    for spot in load_spots():
        counts[int(spot.split("-")[0])] += 1
    return counts


def todays_income():
    """What the paid tickets of today brought in (an estimate: fees as if paid now)."""
    total = 0
    for data in load_tickets().values():
        if data["paid"] and same_day(data["entered"], now()):
            total += parking_fee(minutes_between(data["entered"], now()), data["kind"])
    return round(total, 2)
