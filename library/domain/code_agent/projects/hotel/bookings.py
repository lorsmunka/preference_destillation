"""Making and cancelling bookings."""

from config import CANCEL_FEE_SHARE, FREE_CANCEL_DAYS
from dates import days_until, nights_between
from pricing import stay_price
from rooms import get_room, is_free
from storage import load_bookings, save_bookings


def make_booking(email, room_number, arrival, departure, people):
    """Book a room. The booking number, or the reason it is not possible."""
    if nights_between(arrival, departure) < 1:
        return "A stay is at least one night."
    room = get_room(room_number)
    if not room.fits(people):
        return f"Room {room_number} has only {room.beds} beds."
    if not is_free(room_number, arrival, departure):
        return f"Room {room_number} is taken on those nights."
    bookings = load_bookings()
    number = len(bookings) + 1
    bookings.append({"number": number, "guest": email, "room": room_number, "arrival": arrival,
                     "departure": departure, "people": people, "status": "booked",
                     "price": stay_price(room.kind, arrival, departure, people)})
    save_bookings(bookings)
    return number


def cancel_booking(number):
    """Cancel a booking. Free up to FREE_CANCEL_DAYS before arrival, later half the price. The fee."""
    bookings = load_bookings()
    booking = bookings[number - 1]
    fee = 0 if days_until(booking["arrival"]) >= FREE_CANCEL_DAYS else round(booking["price"] * CANCEL_FEE_SHARE, 2)
    booking["status"] = "cancelled"
    booking["cancel_fee"] = fee
    save_bookings(bookings)
    return fee


def bookings_of(email):
    return [booking for booking in load_bookings() if booking["guest"] == email]
