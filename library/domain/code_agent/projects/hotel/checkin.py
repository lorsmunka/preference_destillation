"""Arriving and leaving: room keys and the bill."""

from guests import add_stay
from rooms import set_clean
from storage import load_bookings, save_bookings


def check_in(number):
    """Hand over the key of a booked room. The room number, or None."""
    bookings = load_bookings()
    booking = bookings[number - 1]
    if booking["status"] != "booked":
        return None
    booking["status"] = "checked_in"
    save_bookings(bookings)
    return booking["room"]


def check_out(number, extras=0):
    """Take the key back, mark the room for cleaning and return the bill (stay + extras)."""
    bookings = load_bookings()
    booking = bookings[number - 1]
    booking["status"] = "checked_out"
    save_bookings(bookings)
    set_clean(booking["room"], False)
    add_stay(booking["guest"])
    return round(booking["price"] + extras, 2)


def in_house():
    """Bookings whose guests are in the hotel right now."""
    return [booking for booking in load_bookings() if booking["status"] == "checked_in"]
