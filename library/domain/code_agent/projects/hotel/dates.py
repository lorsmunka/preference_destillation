"""Stay dates as text ("2026-10-20")."""

from datetime import date, timedelta


def nights_between(arrival, departure):
    return (date.fromisoformat(departure) - date.fromisoformat(arrival)).days


def each_night(arrival, departure):
    """Every night of a stay, as dates."""
    first = date.fromisoformat(arrival)
    return [(first + timedelta(days=offset)).isoformat() for offset in range(nights_between(arrival, departure))]


def is_weekend(day):
    """Friday and Saturday nights count as the weekend."""
    return date.fromisoformat(day).weekday() in (4, 5)


def overlaps(first_arrival, first_departure, second_arrival, second_departure):
    """True when two stays share at least one night."""
    return first_arrival < second_departure and second_arrival < first_departure


def days_until(day):
    return (date.fromisoformat(day) - date.today()).days
