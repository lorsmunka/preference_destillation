"""Dates as text ("2026-10-20")."""

from datetime import date, timedelta


def today():
    return date.today().isoformat()


def add_days(day, days):
    return (date.fromisoformat(day) + timedelta(days=days)).isoformat()


def days_left(due):
    """Days from today until `due` (0 = due today, negative = overdue)."""
    return (date.fromisoformat(due) - date.today()).days


def read_date(text):
    """A typed date: "today", "tomorrow" or YYYY-MM-DD. None when it cannot be read."""
    text = text.strip().lower()
    if text == "today":
        return today()
    if text == "tomorrow":
        return add_days(today(), 1)
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        return None
