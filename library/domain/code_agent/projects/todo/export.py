"""Saving the list as a plain text file to print."""

from config import EXPORT_FILE
from priorities import sort_tasks


def task_line(number, task):
    """"3. [x] Buy milk (due 2026-10-24, high) #home"."""
    box = "[x]" if task["done"] else "[ ]"
    details = ", ".join(part for part in (f"due {task['due']}" if task["due"] else "", task["priority"]) if part)
    tags = " ".join("#" + tag for tag in task["tags"])
    return f"{number}. {box} {task['title']} ({details}) {tags}".rstrip()


def export_tasks(tasks, path=EXPORT_FILE):
    """Write every task, most urgent first, one per line. How many were written."""
    ordered = sort_tasks(tasks)
    with open(path, "w") as file:
        for number, task in enumerate(ordered, 1):
            file.write(task_line(number, task) + "\n")
    return len(ordered)
