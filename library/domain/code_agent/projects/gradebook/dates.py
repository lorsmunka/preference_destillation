"""Dates as text, like "2026-10-20", and the school term they belong to."""

from datetime import date


def today():
    return date.today().isoformat()


def term_of(day):
    """1 for September to December, 2 for January to March, 3 for April to July (None in August)."""
    month = int(day[5:7])
    if month >= 9:
        return 1
    if month <= 3:
        return 2
    if month <= 7:
        return 3
    return None


def in_term(day, term):
    return term_of(day) == term
