"""Tags like "school" or "home" to group tasks."""

from collections import Counter

from config import MAX_TAGS


def add_tag(task, tag):
    """Add a tag (lower case, no spaces). False when the task has it already or has too many."""
    tag = tag.strip().lower().replace(" ", "-")
    if tag in task["tags"] or len(task["tags"]) >= MAX_TAGS:
        return False
    task["tags"].append(tag)
    return True


def tasks_with_tag(tasks, tag):
    return [task for task in tasks if tag.lower() in task["tags"]]


def all_tags(tasks):
    """Every tag in use, the most used first."""
    counts = Counter(tag for task in tasks for tag in task["tags"])
    return [tag for tag, _ in counts.most_common()]
