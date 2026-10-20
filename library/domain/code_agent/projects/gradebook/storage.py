"""The gradebook keeps everything in a few JSON files."""

import json
import os

from config import ATTENDANCE_FILE, DATA_FOLDER, GRADES_FILE, STUDENTS_FILE, TEACHERS_FILE


def load(path, empty):
    if not os.path.exists(path):
        return empty
    with open(path) as file:
        return json.load(file)


def store(path, data):
    """Write `data` to a JSON file, making the data folder first if needed."""
    os.makedirs(DATA_FOLDER, exist_ok=True)
    with open(path, "w") as file:
        json.dump(data, file, indent=2)


def load_students():
    return load(STUDENTS_FILE, {})


def save_students(students):
    store(STUDENTS_FILE, students)


def load_grades():
    return load(GRADES_FILE, [])


def save_grades(grades):
    store(GRADES_FILE, grades)


def load_attendance():
    return load(ATTENDANCE_FILE, {})


def save_attendance(attendance):
    store(ATTENDANCE_FILE, attendance)


def load_teachers():
    return load(TEACHERS_FILE, {})
