"""Reserving a book that is out: members wait in line, first come, first served."""

from storage import load_reservations, save_reservations


def reserve_book(member_id, isbn):
    """Add the member to the end of the waiting line for this book."""
    reservations = load_reservations()
    line = reservations.setdefault(isbn, [])
    if member_id in line:
        return False
    line.append(member_id)
    save_reservations(reservations)
    return True


def next_in_line(isbn):
    """The member who waited longest for this book, or None."""
    line = load_reservations().get(isbn, [])
    return line[0] if line else None


def cancel_reservation(member_id, isbn):
    reservations = load_reservations()
    if member_id in reservations.get(isbn, []):
        reservations[isbn].remove(member_id)
        save_reservations(reservations)


def place_in_line(member_id, isbn):
    """1 for the first in line; None when the member is not waiting."""
    line = load_reservations().get(isbn, [])
    return line.index(member_id) + 1 if member_id in line else None
