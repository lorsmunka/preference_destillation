"""The garage keeps its state in JSON files."""

import json
import os

from config import DATA_FOLDER, PASSES_FILE, SPOTS_FILE, TICKETS_FILE


def load_file(path, empty):
    if not os.path.exists(path):
        return empty
    with open(path) as file:
        return json.load(file)


def save_file(path, data):
    os.makedirs(DATA_FOLDER, exist_ok=True)
    with open(path, "w") as file:
        json.dump(data, file, indent=2)


def load_spots():
    return load_file(SPOTS_FILE, {})


def save_spots(spots):
    save_file(SPOTS_FILE, spots)


def load_tickets():
    return load_file(TICKETS_FILE, {})


def save_tickets(tickets):
    save_file(TICKETS_FILE, tickets)


def load_passes():
    return load_file(PASSES_FILE, {})


def save_passes(passes):
    save_file(PASSES_FILE, passes)
