"""A Task on the list, and adding, finishing and deleting tasks."""

from config import DEFAULT_PRIORITY
from dates import today
from storage import load_tasks, save_tasks
from validation import check_title


class Task:
    def __init__(self, title, due=None, priority=DEFAULT_PRIORITY, tags=None, done=False, repeat=None):
        self.title = title
        self.due = due              # "2026-10-24" or None
        self.priority = priority    # "high", "medium" or "low"
        self.tags = tags or []
        self.done = done
        self.repeat = repeat        # None, "daily" or "weekly"

    def to_data(self):
        return {"title": self.title, "due": self.due, "priority": self.priority,
                "tags": self.tags, "done": self.done, "repeat": self.repeat}


def add_task(username, task):
    """Put a new task on the user's list. None when it worked, otherwise why not."""
    problem = check_title(task.title)
    if problem:
        return problem
    tasks = load_tasks(username)
    tasks.append(task.to_data())
    save_tasks(username, tasks)
    return None


def finish_task(username, number):
    """Tick off task `number` (counting from 1). A repeating task gets its next copy."""
    from recurring import next_copy
    tasks = load_tasks(username)
    task = tasks[number - 1]
    task["done"] = True
    task["finished"] = today()
    copy = next_copy(task)
    if copy:
        tasks.append(copy)
    save_tasks(username, tasks)


def delete_task(username, number):
    tasks = load_tasks(username)
    removed = tasks.pop(number - 1)
    save_tasks(username, tasks)
    return removed["title"]
