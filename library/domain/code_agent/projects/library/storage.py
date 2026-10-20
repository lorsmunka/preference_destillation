"""Reading and writing the library's JSON files."""

import json
import os

from config import BOOKS_FILE, DATA_FOLDER, LOANS_FILE, MEMBERS_FILE, RESERVATIONS_FILE


def read_file(path, empty):
    """The data in a JSON file, or `empty` when there is no file yet."""
    if not os.path.exists(path):
        return empty
    with open(path) as file:
        return json.load(file)


def write_file(path, data):
    os.makedirs(DATA_FOLDER, exist_ok=True)
    with open(path, "w") as file:
        json.dump(data, file, indent=2)


def load_books():
    return read_file(BOOKS_FILE, {})


def save_books(books):
    write_file(BOOKS_FILE, books)


def load_members():
    return read_file(MEMBERS_FILE, {})


def load_loans():
    return read_file(LOANS_FILE, [])


def save_loans(loans):
    write_file(LOANS_FILE, loans)


def load_reservations():
    return read_file(RESERVATIONS_FILE, {})


def save_reservations(reservations):
    write_file(RESERVATIONS_FILE, reservations)
