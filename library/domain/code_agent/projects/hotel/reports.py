"""Numbers for the hotel manager."""

from dates import each_night
from storage import load_bookings, load_rooms


def occupancy_rate(day):
    """The share of rooms in use on the night of `day`."""
    booked = [b for b in load_bookings() if b["status"] != "cancelled" and day in each_night(b["arrival"], b["departure"])]
    return len(booked) / len(load_rooms()) if load_rooms() else 0


def income(month):
    """Money from the stays that start in a month ("2026-10"), plus cancellation fees."""
    total = 0
    for booking in load_bookings():
        if booking["arrival"].startswith(month):
            total += booking.get("cancel_fee", 0) if booking["status"] == "cancelled" else booking["price"]
    return round(total, 2)


def popular_room_kind():
    """The kind of room booked most often."""
    rooms = load_rooms()
    counts = {}
    for booking in load_bookings():
        kind = rooms[booking["room"]]["kind"]
        counts[kind] = counts.get(kind, 0) + 1
    return max(counts, key=counts.get) if counts else None
