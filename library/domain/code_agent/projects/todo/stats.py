"""How well is the user doing?"""

from collections import Counter

from dates import add_days, today


def completion_rate(tasks):
    """The share of tasks that are done, from 0 to 1 (None for an empty list)."""
    if not tasks:
        return None
    return sum(task["done"] for task in tasks) / len(tasks)


def finished_this_week(tasks):
    """Tasks finished in the last seven days."""
    week_ago = add_days(today(), -7)
    return [task for task in tasks if task["done"] and task.get("finished", "") >= week_ago]


def busiest_day(tasks):
    """The date with the most unfinished tasks due, or None."""
    counts = Counter(task["due"] for task in tasks if task["due"] and not task["done"])
    return counts.most_common(1)[0][0] if counts else None


def print_stats(tasks):
    rate = completion_rate(tasks)
    print(f"Done: {rate:.0%}" if rate is not None else "No tasks yet.")
    print(f"Finished this week: {len(finished_this_week(tasks))}")
