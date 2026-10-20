"""The hotel's rooms and which are free."""

from dates import overlaps
from storage import load_bookings, load_rooms, save_rooms


class Room:
    def __init__(self, number, kind, beds, clean=True):
        self.number = number
        self.kind = kind     # "single", "double", "family" or "suite"
        self.beds = beds
        self.clean = clean

    def fits(self, people):
        return people <= self.beds


def get_room(number):
    data = load_rooms()[number]
    return Room(number, data["kind"], data["beds"], data["clean"])


def is_free(number, arrival, departure):
    """True when no active booking of this room shares a night with the stay."""
    for booking in load_bookings():
        if booking["room"] == number and booking["status"] == "booked" and \
                overlaps(booking["arrival"], booking["departure"], arrival, departure):
            return False
    return True


def free_rooms(arrival, departure, people=1):
    """Room numbers free for the whole stay with enough beds, the smallest rooms first."""
    rooms = [get_room(number) for number in load_rooms()]
    found = [room for room in rooms if room.fits(people) and is_free(room.number, arrival, departure)]
    return [room.number for room in sorted(found, key=lambda room: room.beds)]


def set_clean(number, clean):
    rooms = load_rooms()
    rooms[number]["clean"] = clean
    save_rooms(rooms)
