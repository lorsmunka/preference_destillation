"""The hotel's rooms, guests and bookings are JSON files."""

import json
import os

from config import BOOKINGS_FILE, DATA_FOLDER, GUESTS_FILE, ROOMS_FILE


def get_json(path, empty):
    if not os.path.exists(path):
        return empty
    with open(path) as file:
        return json.load(file)


def put_json(path, data):
    os.makedirs(DATA_FOLDER, exist_ok=True)
    with open(path, "w") as file:
        json.dump(data, file, indent=2)


def load_rooms():
    return get_json(ROOMS_FILE, {})


def save_rooms(rooms):
    put_json(ROOMS_FILE, rooms)


def load_bookings():
    return get_json(BOOKINGS_FILE, [])


def save_bookings(bookings):
    put_json(BOOKINGS_FILE, bookings)


def load_guests():
    return get_json(GUESTS_FILE, {})


def save_guests(guests):
    put_json(GUESTS_FILE, guests)
