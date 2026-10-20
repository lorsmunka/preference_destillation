"""Working with dates written as text, like "2026-10-20"."""

from datetime import date, timedelta


def today():
    return date.today().isoformat()


def add_days(day, days):
    """The date `days` after `day`, as text."""
    return (date.fromisoformat(day) + timedelta(days=days)).isoformat()


def days_between(first, second):
    """How many days go from `first` to `second` (negative when second is earlier)."""
    return (date.fromisoformat(second) - date.fromisoformat(first)).days


def is_past(day):
    return days_between(day, today()) > 0
