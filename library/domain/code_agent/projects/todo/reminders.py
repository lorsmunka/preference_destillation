"""Warnings for tasks that are due soon or already late."""

from config import DUE_SOON_DAYS
from dates import days_left


def overdue(tasks):
    """Unfinished tasks whose due date has passed."""
    return [task for task in tasks if not task["done"] and task["due"] and days_left(task["due"]) < 0]


def due_soon(tasks):
    """Unfinished tasks due today or in the next DUE_SOON_DAYS days."""
    return [task for task in tasks if not task["done"] and task["due"] and 0 <= days_left(task["due"]) <= DUE_SOON_DAYS]


def reminder_text(tasks):
    """One short line for the start screen."""
    late, soon = len(overdue(tasks)), len(due_soon(tasks))
    if late == 0 and soon == 0:
        return "Nothing urgent. Well done!"
    parts = []
    if late:
        parts.append(f"{late} overdue")
    if soon:
        parts.append(f"{soon} due soon")
    return "You have " + " and ".join(parts) + "."
