"""Checks on what the user types."""

from config import MAX_TITLE_LENGTH, PRIORITIES


def check_title(title):
    """None for a good title, otherwise what is wrong with it."""
    if not title.strip():
        return "A task needs a title."
    if len(title) > MAX_TITLE_LENGTH:
        return f"Keep the title under {MAX_TITLE_LENGTH} letters."
    return None


def read_priority(text):
    """"h", "high", "H" -> "high"; anything else gives None."""
    text = text.strip().lower()
    for priority in PRIORITIES:
        if text and priority.startswith(text):
            return priority
    return None


def read_number(text, count):
    """A task number from 1 to `count`, or None."""
    return int(text) if text.isdigit() and 1 <= int(text) <= count else None
