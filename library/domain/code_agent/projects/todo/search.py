"""Finding tasks by words in their title."""


def search_titles(tasks, words):
    """Tasks whose title contains every one of the words, ignoring capitals."""
    wanted = words.lower().split()
    return [task for task in tasks if all(word in task["title"].lower() for word in wanted)]


def open_tasks(tasks):
    return [task for task in tasks if not task["done"]]


def number_of(tasks, task):
    """The task's number on the list as the user sees it (from 1)."""
    return tasks.index(task) + 1
