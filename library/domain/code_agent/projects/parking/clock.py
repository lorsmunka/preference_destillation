"""Times as text, like "2026-10-20 14:05"."""

from datetime import datetime

FORMAT = "%Y-%m-%d %H:%M"


def now():
    return datetime.now().strftime(FORMAT)


def minutes_between(start, end):
    """Whole minutes from `start` to `end`."""
    difference = datetime.strptime(end, FORMAT) - datetime.strptime(start, FORMAT)
    return int(difference.total_seconds() // 60)


def same_day(first, second):
    return first[:10] == second[:10]


def hours_text(minutes):
    """135 -> "2 h 15 min"."""
    return f"{minutes // 60} h {minutes % 60} min"
