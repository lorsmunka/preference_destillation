"""Guests: name, email and how often they stayed."""

from storage import load_guests, save_guests


def find_or_add_guest(name, email):
    """The guest with this email; a new guest is saved first."""
    guests = load_guests()
    key = email.strip().lower()
    if key not in guests:
        guests[key] = {"name": name, "stays": 0}
        save_guests(guests)
    return key


def add_stay(email):
    guests = load_guests()
    guests[email]["stays"] += 1
    save_guests(guests)


def is_regular(email):
    """Guests with three stays or more are regulars."""
    return load_guests().get(email, {}).get("stays", 0) >= 3


def guest_name(email):
    return load_guests().get(email, {}).get("name", "guest")
