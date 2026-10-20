"""Tasks that come back: every day or every week."""

from dates import add_days


def next_due(due, repeat):
    """The due date of the next copy of a repeating task."""
    if repeat == "daily":
        return add_days(due, 1)
    if repeat == "weekly":
        return add_days(due, 7)
    return None


def next_copy(task):
    """A fresh, unfinished copy of a finished repeating task with the next due date (or None)."""
    if not task.get("repeat") or not task.get("due"):
        return None
    copy = dict(task)
    copy["done"] = False
    copy["due"] = next_due(task["due"], task["repeat"])
    copy.pop("finished", None)
    return copy


def repeating_tasks(tasks):
    return [task for task in tasks if task.get("repeat")]
