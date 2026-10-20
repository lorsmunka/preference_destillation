"""Sorting tasks: the most urgent first."""

from config import PRIORITIES


def priority_rank(priority):
    """0 for high, 1 for medium, 2 for low (unknown counts as low)."""
    return PRIORITIES.index(priority) if priority in PRIORITIES else len(PRIORITIES) - 1


def sort_tasks(tasks):
    """Unfinished tasks first; then by priority; then the earliest due date (no date last)."""
    return sorted(tasks, key=lambda task: (task["done"], priority_rank(task["priority"]), task["due"] or "9999"))


def raise_priority(task):
    """Move a task one step up: low becomes medium, medium becomes high."""
    rank = priority_rank(task["priority"])
    if rank > 0:
        task["priority"] = PRIORITIES[rank - 1]
    return task["priority"]
