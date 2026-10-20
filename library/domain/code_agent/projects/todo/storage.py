"""Tasks and users are kept in JSON files."""

import json
import os

from config import DATA_FOLDER, TASKS_FILE, USERS_FILE


def read_data(path, empty):
    if not os.path.exists(path):
        return empty
    with open(path) as file:
        return json.load(file)


def write_data(path, data):
    os.makedirs(DATA_FOLDER, exist_ok=True)
    with open(path, "w") as file:
        json.dump(data, file, indent=2)


def load_tasks(username):
    """The tasks of one user (every user has their own list in tasks.json)."""
    return read_data(TASKS_FILE, {}).get(username, [])


def save_tasks(username, tasks):
    everything = read_data(TASKS_FILE, {})
    everything[username] = tasks
    write_data(TASKS_FILE, everything)


def load_users():
    return read_data(USERS_FILE, {})


def save_users(users):
    write_data(USERS_FILE, users)
